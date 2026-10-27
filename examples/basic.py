"""Run a full 4D layer and a backward pass without any external project code."""

import argparse

import torch

from torch_convnd import Conv4d, load_native


def main():
    """Build the selected backend and show a minimal trainable convolution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["auto", "native", "torch"], default="auto")
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.backend == "native":
        load_native(with_cuda=device.type == "cuda")
    layer = Conv4d(2, 4, kernel_size=3, padding=1, device=device, backend=args.backend)
    input = torch.randn(1, 2, 4, 4, 4, 8, device=device, requires_grad=True)
    output = layer(input)
    output.square().mean().backward()
    assert input.grad is not None and torch.isfinite(input.grad).all()
    print(f"input={tuple(input.shape)} output={tuple(output.shape)} backward=OK")


if __name__ == "__main__":
    main()
