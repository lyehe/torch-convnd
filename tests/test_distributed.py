"""Data-parallel training in two processes matches one process on the whole batch."""

import datetime

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from torch_convnd import ConvNd, PartialConvNd


def gloo_device_available():
    """Some builds, such as PyTorch's Windows wheels, ship Gloo without a transport."""
    if not dist.is_available() or not dist.is_gloo_available():
        return False
    try:
        dist.ProcessGroupGloo.create_default_device()
    except RuntimeError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not gloo_device_available(), reason="torch.distributed has no usable Gloo device"
)
WORLD = 2


class Network(nn.Module):
    """A rank-5 convolution feeding a masked partial convolution."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = ConvNd(2, 4, 2, padding=1, dimensions=5, dtype=torch.float64)
        self.partial = PartialConvNd(
            4, 4, 3, padding=1, groups=2, dimensions=5, multi_channel=True, dtype=torch.float64
        )

    def forward(self, input, mask):
        return self.partial(torch.relu(self.conv(input)), mask)


def batch():
    generator = torch.Generator().manual_seed(1)
    input = torch.randn((4, 2) + (3,) * 5, generator=generator, dtype=torch.float64)
    mask = (torch.rand((4, 4) + (4,) * 5, generator=generator) > 0.3).double()
    return input, mask


def train(rank, store, path):
    """One process: its half of the batch, with gradients averaged across processes."""
    dist.init_process_group(
        "gloo",
        init_method=store,
        rank=rank,
        world_size=WORLD,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        model = nn.parallel.DistributedDataParallel(Network())
        input, mask = batch()
        half = slice(rank * 2, rank * 2 + 2)
        model(input[half], mask[half]).square().mean().backward()
        if rank == 0:
            gradients = {name: value.grad for name, value in model.module.named_parameters()}
            torch.save(gradients, path)
    finally:
        dist.destroy_process_group()


def test_data_parallel_gradients_match_one_process(tmp_path):
    """DDP's averaged gradients through native and partial layers equal one process's."""
    path = tmp_path / "gradients.pt"
    mp.spawn(train, args=((tmp_path / "store").as_uri(), str(path)), nprocs=WORLD, join=True)
    distributed = torch.load(path)
    model = Network()
    model(*batch()).square().mean().backward()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(distributed[name], parameter.grad, rtol=1e-12, atol=1e-12)
