"""Map two input channels to eight output features across six grid axes.

Run from a checkout after installing the package: python examples/conv6d.py
"""

import torch

from torch_convnd import ConvNd


def main() -> None:
    """Run compiler-free CPU convolution and backpropagation on a small 6D grid."""
    torch.manual_seed(6)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    layer = ConvNd(
        in_channels=2,  # Two input feature values at every grid location.
        out_channels=8,  # Learn eight output feature maps on the same grid.
        kernel_size=3,  # Three positions along each of the six grid axes.
        dimensions=6,  # Batch and channels are additional, separate axes.
        padding=1,  # Preserve spatial sizes with default stride=1 and dilation=1.
        backend="torch",
        device="cpu",
    )
    # [batch, in_channels, D1, D2, D3, D4, D5, D6].
    observations = torch.randn(1, 2, 3, 3, 3, 3, 3, 3, device="cpu", requires_grad=True)
    features = layer(observations)
    loss = features.square().mean()
    loss.backward()

    assert features.shape == (1, 8, 3, 3, 3, 3, 3, 3)
    assert layer.weight.shape == (8, 2, 3, 3, 3, 3, 3, 3)
    assert observations.grad is not None
    assert layer.weight.grad is not None
    assert torch.isfinite(observations.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    print(f"input  [batch, in_channels, D1..D6]: {tuple(observations.shape)}")
    print(f"weight [out_channels, in_channels, K1..K6]: {tuple(layer.weight.shape)}")
    print(f"output [batch, out_channels, D1..D6]: {tuple(features.shape)}")
    print(f"input gradient: {tuple(observations.grad.shape)}")
    print(f"weight gradient: {tuple(layer.weight.grad.shape)}")


if __name__ == "__main__":
    main()
