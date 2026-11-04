#include "convnd.h"

#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <tuple>
#include <vector>

namespace torch_convnd {
namespace {

int64_t checked_product(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0 &&
                  (right == 0 || left <= std::numeric_limits<int64_t>::max() / right),
              label, " overflows int64");
  return left * right;
}

int64_t checked_sum(int64_t left, int64_t right, const char* label) {
  TORCH_CHECK(left >= 0 && right >= 0 &&
                  left <= std::numeric_limits<int64_t>::max() - right,
              label, " overflows int64");
  return left + right;
}

Geometry make_geometry(const at::Tensor& input, const at::Tensor& weight,
                       at::IntArrayRef stride, at::IntArrayRef padding,
                       at::IntArrayRef dilation, int64_t groups) {
  TORCH_CHECK(input.layout() == at::kStrided && weight.layout() == at::kStrided,
              "conv_nd requires strided tensors");
  TORCH_CHECK(input.device().is_cpu() || input.device().is_cuda(),
              "conv_nd supports CPU and CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble ||
                  input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
              "native conv_nd supports float16, bfloat16, float32 and float64");
  TORCH_CHECK(weight.device() == input.device() && weight.scalar_type() == input.scalar_type(),
              "input and weight must have the same device and dtype");
  TORCH_CHECK(input.dim() >= 3 && input.dim() <= kMaxSpatialDims + 2,
              "conv_nd requires 1 to 62 spatial dimensions");
  TORCH_CHECK(weight.dim() == input.dim(), "input and weight ranks must match");
  Geometry geometry;
  geometry.rank = input.dim() - 2;
  TORCH_CHECK(static_cast<int64_t>(stride.size()) == geometry.rank &&
                  static_cast<int64_t>(padding.size()) == geometry.rank &&
                  static_cast<int64_t>(dilation.size()) == geometry.rank,
              "stride, padding and dilation must match the spatial rank");
  TORCH_CHECK(groups > 0, "groups must be positive");
  geometry.channels = input.size(1);
  geometry.output_channels = weight.size(0);
  geometry.groups = groups;
  TORCH_CHECK(geometry.channels > 0 && geometry.output_channels > 0 &&
                  geometry.channels % groups == 0 && geometry.output_channels % groups == 0 &&
                  weight.size(1) == geometry.channels / groups,
              "input/weight channels must match and be divisible by groups");
  for (int64_t axis = 0; axis < geometry.rank; ++axis) {
    TORCH_CHECK(input.size(axis + 2) > 0 && weight.size(axis + 2) > 0,
                "spatial and kernel dimensions must be positive");
    TORCH_CHECK(stride[axis] > 0 && dilation[axis] > 0 && padding[axis] >= 0,
                "stride and dilation must be positive; padding must be nonnegative");
    geometry.input[axis] = input.size(axis + 2);
    geometry.kernel[axis] = weight.size(axis + 2);
    geometry.stride[axis] = stride[axis];
    geometry.padding[axis] = padding[axis];
    geometry.dilation[axis] = dilation[axis];
    const int64_t effective_kernel = checked_sum(
        checked_product(dilation[axis], geometry.kernel[axis] - 1, "dilated kernel"),
        1, "dilated kernel");
    const int64_t padded_input = checked_sum(
        geometry.input[axis], checked_product(2, padding[axis], "padding"), "padded input");
    TORCH_CHECK(padded_input >= effective_kernel, "kernel exceeds padded input");
    geometry.output[axis] = (padded_input - effective_kernel) / stride[axis] + 1;
    geometry.input_volume = checked_product(
        geometry.input_volume, geometry.input[axis], "input spatial volume");
    geometry.output_volume = checked_product(
        geometry.output_volume, geometry.output[axis], "output spatial volume");
    geometry.kernel_volume = checked_product(
        geometry.kernel_volume, geometry.kernel[axis], "kernel volume");
  }
  return geometry;
}

int64_t tile_size(const Geometry& geometry, int64_t element_size, int64_t workspace_bytes) {
  // One patch buffer is reused for forward, weight gradients and input gradients.
  const int64_t column_rows = checked_product(
      geometry.channels, geometry.kernel_volume, "patch rows");
  // Low-precision backward accumulates in FP32 and uses the same workspace budget.
  const int64_t bytes_per_position = checked_product(
      std::max<int64_t>(element_size, 4), column_rows, "workspace bytes per position");
  TORCH_CHECK(workspace_bytes >= bytes_per_position,
              "workspace_bytes is too small for one output position; requires at least ",
              bytes_per_position);
  return std::min(geometry.output_volume, workspace_bytes / bytes_per_position);
}

void unfold(const at::Tensor& input, at::Tensor& columns, const Geometry& geometry,
            int64_t start, int64_t length) {
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    unfold_cuda(input, columns, geometry, start, length);
#else
    TORCH_CHECK(false, "conv_nd was built without CUDA; rebuild with with_cuda=True");
#endif
  } else {
    unfold_cpu(input, columns, geometry, start, length);
  }
}

void fold(const at::Tensor& columns, at::Tensor& output, const Geometry& geometry,
          int64_t start, int64_t length) {
  if (output.is_cuda()) {
#ifdef WITH_CUDA
    fold_cuda(columns, output, geometry, start, length);
#else
    TORCH_CHECK(false, "conv_nd was built without CUDA; rebuild with with_cuda=True");
#endif
  } else {
    fold_cpu(columns, output, geometry, start, length);
  }
}

std::vector<int64_t> output_shape(const at::Tensor& input, const Geometry& geometry) {
  std::vector<int64_t> result = {input.size(0), geometry.output_channels};
  result.insert(result.end(), geometry.output, geometry.output + geometry.rank);
  return result;
}

// Row-major coordinates of a flat index.
void decode(int64_t index, const int64_t* sizes, int64_t rank, int64_t* coordinates) {
  for (int64_t axis = rank - 1; axis >= 0; --axis) {
    coordinates[axis] = index % sizes[axis];
    index /= sizes[axis];
  }
}

// Steps row-major coordinates from `axis` outward; true when they wrap around to zero.
bool advance(const int64_t* sizes, int64_t axis, int64_t* coordinates) {
  for (; axis >= 0; --axis) {
    if (++coordinates[axis] < sizes[axis]) return false;
    coordinates[axis] = 0;
  }
  return true;
}

// Runs body(begin, end) over ranges of a contiguous tensor's elements in parallel. With
// PyTorch's default OpenMP backend, at::parallel_for in an extension built without OpenMP
// flags runs serially; TensorIterator's loop runs on PyTorch's own threads instead and
// follows torch.set_num_threads.
template <typename Body>
void parallel_ranges(const at::Tensor& tensor, int64_t grain_size, const Body& body) {
  const auto flat = tensor.view({-1});
  auto iterator = at::TensorIteratorConfig()
                      .add_output(flat)
                      .resize_outputs(false)
                      .check_all_same_dtype(false)
                      .build();
  const char* base = static_cast<const char*>(flat.const_data_ptr());
  const int64_t element_size = flat.element_size();
  iterator.for_each(
      [&](char** data, const int64_t* strides, int64_t length, int64_t rows) {
        for (int64_t row = 0; row < rows; ++row) {
          const int64_t begin = (data[0] + row * strides[1] - base) / element_size;
          body(begin, begin + length);
        }
      },
      grain_size);
}

}  // namespace

// Decoding each element's position divides on every axis, so each range decodes its
// first element and then steps through positions and taps in order.
void unfold_cpu(const at::Tensor& input, at::Tensor& columns, const Geometry& full_geometry,
                int64_t start, int64_t length) {
  const Geometry geometry = without_unit_axes(full_geometry);
  const int64_t last = geometry.rank - 1;
  std::array<int64_t, kMaxSpatialDims> multiplier{};
  multiplier[last] = 1;
  for (int64_t axis = last - 1; axis >= 0; --axis) {
    multiplier[axis] = multiplier[axis + 1] * geometry.input[axis + 1];
  }
  std::array<int64_t, kMaxSpatialDims> first{};
  decode(start, geometry.output, geometry.rank, first.data());
  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(),
                                "unfold_nd_cpu", [&] {
    const scalar_t* source = input.const_data_ptr<scalar_t>();
    scalar_t* target = columns.mutable_data_ptr<scalar_t>();
    parallel_ranges(columns, 4096, [&](int64_t begin, int64_t end) {
      int64_t column = begin % length;
      int64_t channel = begin / length / geometry.kernel_volume;
      std::array<int64_t, kMaxSpatialDims> tap{};
      std::array<int64_t, kMaxSpatialDims> position{};
      decode(begin / length % geometry.kernel_volume, geometry.kernel, geometry.rank, tap.data());
      decode(start + column, geometry.output, geometry.rank, position.data());
      for (int64_t index = begin; index < end;) {
        // Every axis but the last is constant along a run of the last one.
        int64_t offset = channel * geometry.input_volume;
        bool inside = true;
        for (int64_t axis = 0; axis < last; ++axis) {
          const int64_t coordinate = position[axis] * geometry.stride[axis] -
              geometry.padding[axis] + tap[axis] * geometry.dilation[axis];
          inside = inside && coordinate >= 0 && coordinate < geometry.input[axis];
          offset += coordinate * multiplier[axis];
        }
        const int64_t origin = tap[last] * geometry.dilation[last] - geometry.padding[last];
        const int64_t run =
            std::min({geometry.output[last] - position[last], length - column, end - index});
        for (int64_t step = 0; step < run; ++step) {
          const int64_t coordinate = (position[last] + step) * geometry.stride[last] + origin;
          target[index + step] = inside && coordinate >= 0 && coordinate < geometry.input[last]
                                     ? source[offset + coordinate]
                                     : scalar_t(0);
        }
        index += run;
        column += run;
        position[last] += run;
        if (column == length) {
          // The next row holds the next tap, or the next channel, from the tile's start.
          column = 0;
          position = first;
          if (advance(geometry.kernel, last, tap.data())) ++channel;
        } else if (position[last] == geometry.output[last]) {
          position[last] = 0;
          advance(geometry.output, last - 1, position.data());
        }
      }
    });
  });
}

void fold_cpu(const at::Tensor& columns, at::Tensor& output, const Geometry& full_geometry,
              int64_t start, int64_t length) {
  const Geometry geometry = without_unit_axes(full_geometry);
  const int64_t last = geometry.rank - 1;
  std::array<int64_t, kMaxSpatialDims> multiplier{};
  multiplier[last] = 1;
  for (int64_t axis = last - 1; axis >= 0; --axis) {
    multiplier[axis] = multiplier[axis + 1] * geometry.output[axis + 1];
  }
  // Where each axis's taps start in the per-element table below.
  std::array<int64_t, kMaxSpatialDims> first_tap{};
  int64_t tap_count = 0;
  for (int64_t axis = 0; axis < geometry.rank; ++axis) {
    first_tap[axis] = tap_count;
    tap_count += geometry.kernel[axis];
  }
  AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "fold_nd_cpu", [&] {
    const scalar_t* source = columns.const_data_ptr<scalar_t>();
    scalar_t* target = output.mutable_data_ptr<scalar_t>();
    parallel_ranges(output, 512, [&](int64_t begin, int64_t end) {
      int64_t channel = begin / geometry.input_volume;
      std::array<int64_t, kMaxSpatialDims> element{};
      decode(begin % geometry.input_volume, geometry.input, geometry.rank, element.data());
      // Per axis and tap: the output coordinate whose window puts that tap on the
      // element, or a negative value when there is none.
      std::vector<int64_t> reached(tap_count);
      for (int64_t index = begin; index < end; ++index) {
        for (int64_t axis = 0; axis < geometry.rank; ++axis) {
          for (int64_t step = 0; step < geometry.kernel[axis]; ++step) {
            int64_t coordinate =
                element[axis] + geometry.padding[axis] - step * geometry.dilation[axis];
            if (coordinate >= 0 && geometry.stride[axis] != 1) {
              coordinate = coordinate % geometry.stride[axis] == 0
                               ? coordinate / geometry.stride[axis]
                               : -1;
            }
            reached[first_tap[axis] + step] = coordinate < geometry.output[axis] ? coordinate : -1;
          }
        }
        // Taps in the per-element map's order, so the sum rounds identically.
        scalar_t sum = 0;
        std::array<int64_t, kMaxSpatialDims> tap{};
        for (int64_t row = channel * geometry.kernel_volume,
                     rows_end = row + geometry.kernel_volume;
             row < rows_end;) {
          int64_t position = 0;
          bool inside = true;
          for (int64_t axis = 0; axis < last; ++axis) {
            const int64_t coordinate = reached[first_tap[axis] + tap[axis]];
            inside = inside && coordinate >= 0;
            position += coordinate * multiplier[axis];
          }
          for (int64_t step = 0; step < geometry.kernel[last]; ++step, ++row) {
            const int64_t coordinate = reached[first_tap[last] + step];
            const int64_t column = position + coordinate - start;
            if (inside && coordinate >= 0 && column >= 0 && column < length) {
              sum += source[row * length + column];
            }
          }
          advance(geometry.kernel, last - 1, tap.data());
        }
        target[index] += sum;
        if (advance(geometry.input, last, element.data())) ++channel;
      }
    });
  });
}

at::Tensor conv_nd(const at::Tensor& input, const at::Tensor& weight,
                   const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                   at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                   int64_t workspace_bytes) {
  const auto geometry = make_geometry(input, weight, stride, padding, dilation, groups);
  if (bias.has_value()) {
    TORCH_CHECK(bias->layout() == at::kStrided && bias->dim() == 1 &&
                    bias->size(0) == geometry.output_channels &&
                    bias->device() == input.device() && bias->scalar_type() == input.scalar_type(),
                "bias must match output channels, device and dtype");
  }
  const c10::OptionalDeviceGuard guard(input.device());
  const auto contiguous_input = input.contiguous();
  const auto flat_weight = weight.contiguous().view(
      {groups, geometry.output_channels / groups, -1});
  auto output = at::empty(output_shape(input, geometry), input.options());
  auto flat_output = output.view({input.size(0), groups, geometry.output_channels / groups,
                                 geometry.output_volume});
  const int64_t tile = tile_size(geometry, input.element_size(), workspace_bytes);
  const int64_t rows = geometry.channels / groups * geometry.kernel_volume;
  if (input.size(0) == 0) return output;
  const auto patch_storage = at::empty({groups * rows * tile}, input.options());
  for (int64_t batch = 0; batch < input.size(0); ++batch) {
    const auto image = contiguous_input.select(0, batch);
    for (int64_t start = 0; start < geometry.output_volume; start += tile) {
      const int64_t length = std::min(tile, geometry.output_volume - start);
      auto columns = patch_storage.narrow(0, 0, groups * rows * length).view({groups, rows, length});
      unfold(image, columns, geometry, start, length);
      auto output_tile = flat_output.select(0, batch).slice(2, start, start + length);
      at::bmm_out(output_tile, flat_weight, columns);
    }
  }
  if (bias.has_value()) {
    std::vector<int64_t> bias_shape(input.dim(), 1);
    bias_shape[1] = geometry.output_channels;
    output.add_(bias->reshape(bias_shape));
  }
  return output;
}

std::tuple<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>>
conv_nd_backward(const at::Tensor& grad_output, const at::Tensor& input,
                 const at::Tensor& weight, at::IntArrayRef stride, at::IntArrayRef padding,
                 at::IntArrayRef dilation, int64_t groups, int64_t workspace_bytes,
                 std::array<bool, 3> output_mask) {
  const auto geometry = make_geometry(input, weight, stride, padding, dilation, groups);
  TORCH_CHECK(grad_output.sizes().vec() == output_shape(input, geometry) &&
                  grad_output.device() == input.device() &&
                  grad_output.scalar_type() == input.scalar_type() &&
                  grad_output.layout() == at::kStrided,
              "grad_output must match the convolution output shape, device and dtype");
  const c10::OptionalDeviceGuard guard(input.device());
  if (input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16) {
#if defined(WITH_CUDA) && defined(TORCH_CONVND_FP32_OUTPUT_GEMM)
    if (input.is_cuda()) {
      return mixed_backward_cuda(grad_output, input, weight, geometry,
                                 tile_size(geometry, 4, workspace_bytes), output_mask);
    }
#endif
    const auto promoted = conv_nd_backward(
        grad_output.to(at::kFloat), input.to(at::kFloat), weight.to(at::kFloat),
        stride, padding, dilation, groups, workspace_bytes, output_mask);
    const auto cast_gradient = [&](const std::optional<at::Tensor>& gradient)
        -> std::optional<at::Tensor> {
      if (gradient.has_value()) return gradient->to(input.scalar_type());
      return std::nullopt;
    };
    return {cast_gradient(std::get<0>(promoted)), cast_gradient(std::get<1>(promoted)),
            cast_gradient(std::get<2>(promoted))};
  }
  const int64_t tile = tile_size(geometry, input.element_size(), workspace_bytes);
  const int64_t rows = geometry.channels / groups * geometry.kernel_volume;
  const auto contiguous_input = input.contiguous();
  const auto flat_weight = weight.contiguous().view(
      {groups, geometry.output_channels / groups, rows});
  const auto transposed_weight = flat_weight.transpose(1, 2);
  const auto flat_grad = grad_output.contiguous().view(
      {input.size(0), groups, geometry.output_channels / groups, geometry.output_volume});
  std::optional<at::Tensor> grad_input;
  std::optional<at::Tensor> grad_weight;
  std::optional<at::Tensor> grad_bias;
  if (output_mask[0]) grad_input = at::zeros(input.sizes(), input.options());
  if (output_mask[1]) grad_weight = at::zeros(weight.sizes(), weight.options());
  if (output_mask[2]) {
    grad_bias = flat_grad.sum(at::IntArrayRef({0, 3})).reshape({geometry.output_channels});
  }
  if ((!output_mask[0] && !output_mask[1]) || input.size(0) == 0) {
    return {grad_input, grad_weight, grad_bias};
  }

  const auto patch_storage = at::empty({groups * rows * tile}, input.options());
  for (int64_t batch = 0; batch < input.size(0); ++batch) {
    const auto image = contiguous_input.select(0, batch);
    for (int64_t start = 0; start < geometry.output_volume; start += tile) {
      const int64_t length = std::min(tile, geometry.output_volume - start);
      const auto tile_grad = flat_grad.select(0, batch).slice(2, start, start + length);
      auto columns = patch_storage.narrow(0, 0, groups * rows * length).view({groups, rows, length});
      if (output_mask[1]) {
        unfold(image, columns, geometry, start, length);
        grad_weight->view({groups, geometry.output_channels / groups, rows}).baddbmm_(
            tile_grad, columns.transpose(1, 2));
      }
      if (output_mask[0]) {
        at::bmm_out(columns, transposed_weight, tile_grad);
        auto image_grad = grad_input->select(0, batch);
        fold(columns, image_grad, geometry, start, length);
      }
    }
  }
  return {grad_input, grad_weight, grad_bias};
}

template <c10::DeviceType DeviceType>
at::Tensor autocast_conv_nd(const at::Tensor& input, const at::Tensor& weight,
                            const std::optional<at::Tensor>& bias, at::IntArrayRef stride,
                            at::IntArrayRef padding, at::IntArrayRef dilation, int64_t groups,
                            int64_t workspace_bytes) {
  const c10::impl::ExcludeDispatchKeyGuard guard(
      at::autocast::get_autocast_dispatch_key_from_device_type(DeviceType));
  const auto dtype = at::autocast::get_autocast_dtype(DeviceType);
  static const auto operation = c10::Dispatcher::singleton()
      .findSchemaOrThrow("torch_convnd::conv_nd", "").typed<decltype(conv_nd)>();
  return operation.call(at::autocast::cached_cast(dtype, input, DeviceType),
                        at::autocast::cached_cast(dtype, weight, DeviceType),
                        at::autocast::cached_cast(dtype, bias, DeviceType),
                        stride, padding, dilation, groups, workspace_bytes);
}

}  // namespace torch_convnd

TORCH_LIBRARY_IMPL(torch_convnd, CPU, library) {
  library.impl("conv_nd", &torch_convnd::conv_nd);
  library.impl("conv_nd_backward", &torch_convnd::conv_nd_backward);
}

TORCH_LIBRARY_IMPL(torch_convnd, AutocastCPU, library) {
  library.impl("conv_nd", &torch_convnd::autocast_conv_nd<c10::DeviceType::CPU>);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(torch_convnd, CUDA, library) {
  library.impl("conv_nd", &torch_convnd::conv_nd);
  library.impl("conv_nd_backward", &torch_convnd::conv_nd_backward);
}
TORCH_LIBRARY_IMPL(torch_convnd, AutocastCUDA, library) {
  library.impl("conv_nd", &torch_convnd::autocast_conv_nd<c10::DeviceType::CUDA>);
}
#endif
