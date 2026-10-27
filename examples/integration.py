"""A framework-independent 4D classifier with ordinary PyTorch training."""

import argparse

import torch
from torch import Tensor, nn

from torch_convnd import Conv4d


class EventClassifier(nn.Module):
    """Example composition; replace the synthetic observations with your own data."""

    def __init__(self, backend: str = "auto"):
        super().__init__()
        self.features = nn.Sequential(
            Conv4d(2, 4, 3, padding=1, backend=backend),
            nn.GELU(),
            Conv4d(4, 4, 3, padding=1, backend=backend),
            nn.GELU(),
        )
        self.classifier = nn.Linear(4, 2)

    def forward(self, input: Tensor) -> Tensor:
        features = self.features(input)
        pooled = features.mean(dim=tuple(range(2, features.ndim)))
        return self.classifier(pooled)


def main() -> None:
    """Run a few optimizer steps without a dataset or training framework."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["auto", "native", "torch"], default="auto")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    torch.manual_seed(17)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = EventClassifier(args.backend).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    observations = torch.randn(2, 2, 4, 4, 4, 4, device=args.device)
    labels = torch.tensor([0, 1], device=args.device)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(observations), labels)
        loss.backward()
        optimizer.step()
        print(f"step={step + 1} loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
