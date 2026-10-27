#include "convnd.h"

#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/detail/IntegerDivider.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <limits>

namespace torch_convnd {
namespace {

constexpr int kThreads = 256;

int blocks_for(int64_t elements) {
  return static_cast<int>(std::min<int64_t>((elements + kThreads - 1) / kThreads, 65535));
}

using FastDivider = at::cuda::detail::IntDivider<unsigned int>;

template <int Rank>
struct FastGeometry {
  FastDivider output[Rank];
  FastDivider kernel[Rank];
  FastDivider kernel_volume;
  int input[Rank];
  int input_stride[Rank];
  int stride[Rank];
  int padding[Rank];
  int dilation[Rank];
  int input_volume;
};

template <int Rank>
FastGeometry<Rank> make_fast_geometry(const Geometry& geometry) {
  FastGeometry<Rank> result;
  result.kernel_volume = FastDivider(static_cast<unsigned int>(geometry.kernel_volume));
  result.input_volume = static_cast<int>(geometry.input_volume);
  int multiplier = 1;
  for (int axis = Rank - 1; axis >= 0; --axis) {
    result.output[axis] = FastDivider(static_cast<unsigned int>(geometry.output[axis]));
    result.kernel[axis] = FastDivider(static_cast<unsigned int>(geometry.kernel[axis]));
    result.input[axis] = static_cast<int>(geometry.input[axis]);
    result.input_stride[axis] = multiplier;
    result.stride[axis] = static_cast<int>(geometry.stride[axis]);
    result.padding[axis] = static_cast<int>(geometry.padding[axis]);
    result.dilation[axis] = static_cast<int>(geometry.dilation[axis]);
    multiplier *= result.input[axis];
  }
  return result;
}

bool can_use_fast_geometry(const Geometry& geometry, int64_t image_elements,
                           int64_t column_elements) {
  constexpr int64_t limit = std::numeric_limits<int>::max();
  if (geometry.rank > 8 || image_elements > limit || column_elements > limit ||
      geometry.output_volume > limit || geometry.kernel_volume > limit) return false;
  for (int64_t axis = 0; axis < geometry.rank; ++axis) {
    if (geometry.input[axis] > limit || geometry.kernel[axis] > limit ||
        geometry.output[axis] > limit || geometry.stride[axis] > limit ||
        geometry.padding[axis] > limit || geometry.dilation[axis] > limit ||
        (geometry.output[axis] - 1) * geometry.stride[axis] +
            (geometry.kernel[axis] - 1) * geometry.dilation[axis] > limit) return false;
  }
  return true;
}

template <int Rank>
__device__ __forceinline__ int fast_input_offset(
    unsigned int row, unsigned int position, const FastGeometry<Rank>& geometry) {
  const auto channel_tap = geometry.kernel_volume.divmod(row);
  unsigned int tap = channel_tap.mod;
  int offset = static_cast<int>(channel_tap.div) * geometry.input_volume;
#pragma unroll
  for (int axis = Rank - 1; axis >= 0; --axis) {
    const auto spatial = geometry.output[axis].divmod(position);
    const auto kernel = geometry.kernel[axis].divmod(tap);
    const int coordinate = static_cast<int>(spatial.mod) * geometry.stride[axis] -
        geometry.padding[axis] + static_cast<int>(kernel.mod) * geometry.dilation[axis];
    if (coordinate < 0 || coordinate >= geometry.input[axis]) return -1;
    offset += coordinate * geometry.input_stride[axis];
    position = spatial.div;
    tap = kernel.div;
  }
  return offset;
}

template <typename scalar_t, int Rank, bool Fold>
__global__ void fast_patch_kernel(const scalar_t* input, scalar_t* output,
                                  FastGeometry<Rank> geometry, FastDivider length_divider,
                                  int start, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const auto row_position = length_divider.divmod(static_cast<unsigned int>(index));
    const int offset = fast_input_offset<Rank>(
        row_position.div, start + row_position.mod, geometry);
    if constexpr (Fold) {
      if (offset >= 0) gpuAtomicAdd(output + offset, input[index]);
    } else {
      output[index] = offset < 0 ? scalar_t(0) : input[offset];
    }
  }
}

template <typename scalar_t, int Rank, bool Fold>
void launch_fast_patch(const scalar_t* input, scalar_t* output, const Geometry& geometry,
                       int64_t start, int64_t length, int64_t elements,
                       cudaStream_t stream) {
  fast_patch_kernel<scalar_t, Rank, Fold><<<blocks_for(elements), kThreads, 0, stream>>>(
      input, output, make_fast_geometry<Rank>(geometry),
      FastDivider(static_cast<unsigned int>(length)), static_cast<int>(start), elements);
}

template <typename scalar_t, bool Fold>
void dispatch_fast_patch(const scalar_t* input, scalar_t* output, const Geometry& geometry,
                         int64_t start, int64_t length, int64_t elements,
                         cudaStream_t stream) {
#define TORCH_CONVND_RANK_CASE(rank) \
  case rank: launch_fast_patch<scalar_t, rank, Fold>( \
      input, output, geometry, start, length, elements, stream); break
  switch (geometry.rank) {
    TORCH_CONVND_RANK_CASE(1);
    TORCH_CONVND_RANK_CASE(2);
    TORCH_CONVND_RANK_CASE(3);
    TORCH_CONVND_RANK_CASE(4);
    TORCH_CONVND_RANK_CASE(5);
    TORCH_CONVND_RANK_CASE(6);
    TORCH_CONVND_RANK_CASE(7);
    TORCH_CONVND_RANK_CASE(8);
  }
#undef TORCH_CONVND_RANK_CASE
}

template <typename scalar_t>
__global__ void unfold_kernel(const scalar_t* input, scalar_t* columns, Geometry geometry,
                              int64_t start, int64_t length, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t offset = column_input_offset(index / length, start + index % length, geometry);
    columns[index] = offset < 0 ? scalar_t(0) : input[offset];
  }
}

template <typename scalar_t>
__global__ void fold_atomic_kernel(const scalar_t* columns, scalar_t* output, Geometry geometry,
                                   int64_t start, int64_t length, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t offset = column_input_offset(index / length, start + index % length, geometry);
    if (offset >= 0) gpuAtomicAdd(output + offset, columns[index]);
  }
}

template <typename scalar_t>
__global__ void fold_gather_kernel(const scalar_t* columns, scalar_t* output, Geometry geometry,
                                   int64_t start, int64_t length, int64_t elements) {
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    scalar_t sum = 0;
    for (int64_t tap = 0; tap < geometry.kernel_volume; ++tap) {
      const int64_t offset = input_column_offset(index, tap, geometry, start, length);
      if (offset >= 0) sum += columns[offset];
    }
    output[index] += sum;
  }
}

}  // namespace

void unfold_cuda(const at::Tensor& input, at::Tensor& columns, const Geometry& geometry,
                 int64_t start, int64_t length) {
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(),
                                "unfold_nd_cuda", [&] {
    if (can_use_fast_geometry(geometry, input.numel(), columns.numel())) {
      dispatch_fast_patch<scalar_t, false>(
          input.const_data_ptr<scalar_t>(), columns.mutable_data_ptr<scalar_t>(), geometry,
          start, length, columns.numel(), stream);
    } else {
      unfold_kernel<<<blocks_for(columns.numel()), kThreads, 0, stream>>>(
          input.const_data_ptr<scalar_t>(), columns.mutable_data_ptr<scalar_t>(), geometry,
          start, length, columns.numel());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fold_cuda(const at::Tensor& columns, at::Tensor& output, const Geometry& geometry,
               int64_t start, int64_t length) {
  const auto stream = c10::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "fold_nd_cuda", [&] {
    if (at::globalContext().deterministicAlgorithms()) {
      fold_gather_kernel<<<blocks_for(output.numel()), kThreads, 0, stream>>>(
          columns.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(), geometry,
          start, length, output.numel());
    } else if (can_use_fast_geometry(geometry, output.numel(), columns.numel())) {
      dispatch_fast_patch<scalar_t, true>(
          columns.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(), geometry,
          start, length, columns.numel(), stream);
    } else {
      fold_atomic_kernel<<<blocks_for(columns.numel()), kThreads, 0, stream>>>(
          columns.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(), geometry,
          start, length, columns.numel());
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace torch_convnd
