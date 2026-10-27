#include "convnd.h"

#include <ATen/ops/baddbmm.h>
#include <ATen/ops/bmm.h>
#include <algorithm>

namespace torch_convnd {

// Half/BF16 operands use tensor-core GEMM with FP32 outputs. Only gradients,
// not the complete input, weights and output cotangent, need FP32 storage.
std::tuple<std::optional<at::Tensor>, std::optional<at::Tensor>, std::optional<at::Tensor>>
mixed_backward_cuda(const at::Tensor& grad_output, const at::Tensor& input,
                    const at::Tensor& weight, const Geometry& geometry, int64_t tile,
                    std::array<bool, 3> output_mask) {
  const auto options = input.options().dtype(at::kFloat);
  const int64_t groups = geometry.groups;
  const int64_t rows = geometry.channels / groups * geometry.kernel_volume;
  const int64_t output_channels = geometry.output_channels / groups;
  const auto source = input.contiguous();
  const auto matrix = weight.contiguous().view({groups, output_channels, rows});
  const auto transpose = matrix.transpose(1, 2);
  const auto gradient = grad_output.contiguous().view(
      {input.size(0), groups, output_channels, geometry.output_volume});
  std::optional<at::Tensor> grad_input, grad_weight, grad_bias;
  if (output_mask[0]) grad_input = at::zeros(input.sizes(), options);
  if (output_mask[1]) grad_weight = at::zeros(weight.sizes(), options);
  if (output_mask[2]) {
    grad_bias = gradient.sum(at::IntArrayRef({0, 3}), false, at::kFloat)
        .reshape({geometry.output_channels});
  }
  if (input.size(0) > 0 && (output_mask[0] || output_mask[1])) {
    // One buffer: low-precision patches for dW, then FP32 columns for dX.
    const auto storage = at::empty({groups * rows * tile}, options);
    for (int64_t batch = 0; batch < input.size(0); ++batch) {
      const auto image = source.select(0, batch);
      for (int64_t start = 0; start < geometry.output_volume; start += tile) {
        const int64_t length = std::min(tile, geometry.output_volume - start);
        const auto tile_grad = gradient.select(0, batch).slice(2, start, start + length);
        if (output_mask[1]) {
          auto patches = storage.view(input.scalar_type()).narrow(0, 0, groups * rows * length)
              .view({groups, rows, length});
          unfold_cuda(image, patches, geometry, start, length);
          auto target = grad_weight->view({groups, output_channels, rows});
          at::baddbmm_out(target, target, tile_grad, patches.transpose(1, 2), at::kFloat);
        }
        if (output_mask[0]) {
          auto columns = storage.narrow(0, 0, groups * rows * length).view({groups, rows, length});
          at::bmm_out(columns, transpose, tile_grad, at::kFloat);
          auto target = grad_input->select(0, batch);
          fold_cuda(columns, target, geometry, start, length);
        }
      }
    }
  }
  const auto cast = [&](const std::optional<at::Tensor>& value) -> std::optional<at::Tensor> {
    if (value.has_value()) return value->to(input.scalar_type());
    return std::nullopt;
  };
  return {cast(grad_input), cast(grad_weight), cast(grad_bias)};
}

}  // namespace torch_convnd
