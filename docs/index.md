# torch-convnd

Add full N-dimensional dense convolution to an ordinary PyTorch project:

```python
import torch
from torch_convnd import Conv4d

layer = Conv4d(2, 8, 3, padding=1, backend="torch")
input = torch.randn(1, 2, 4, 4, 4, 8, requires_grad=True)
output = layer(input)
output.square().mean().backward()
```

Every spatial axis has independent kernel weights. The rank-4 portable path sums
batched Conv3d taps; it preserves the full kernel rather than learning a separable
approximation. The native addon extracts tiled patches and calls GEMM.

Start with [installation](installation.md), then [integrate a model](integration.md).
The [API](api.md) follows stock convolution constructors where ranks overlap.
The [support matrix](support.md) records limits and verification status.

For developers, read [testing](development.md) and [release automation](releasing.md).
Historical [correctness](PARITY.md) and [performance](PERFORMANCE.md) reports
retain the raw measurements and source hashes.
