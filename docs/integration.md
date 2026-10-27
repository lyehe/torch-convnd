# Integration guide

## Shapes and composition

`ConvNd` accepts 1-62 spatial axes. Inputs are `[N,C,*spatial]` or unbatched
`[C,*spatial]`. Weights are `[Cout,Cin/groups,*kernel]`. Kernel, stride, dilation
and numeric padding accept an integer or a tuple with one entry per spatial axis.

```python
import torch
from torch import nn
from torch_convnd import Conv4d

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Conv4d(2, 8, 3, padding="same", backend="torch"),
            nn.GELU(),
            Conv4d(8, 4, 1, backend="torch"),
        )

    def forward(self, x):
        return self.features(x)

model = Model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(2, 2, 4, 4, 4, 8)
optimizer.zero_grad(set_to_none=True)
loss = model(x).square().mean()
loss.backward()
optimizer.step()
```

Use the same parameters and optimizers as other PyTorch modules. No callbacks,
training base classes or data format adapters are required. For distributed
training, the layer exposes ordinary parameters; multi-process DDP/FSDP behavior
is not yet part of the validated matrix.

## Existing convolution parameters

`Conv1d`, `Conv2d` and `Conv3d` use the corresponding stock positional constructor
arguments and state keys. Transfer an existing state dictionary directly:

```python
from torch import nn
from torch_convnd import Conv3d

stock = nn.Conv3d(4, 8, 3, padding=1, groups=2)
layer = Conv3d(4, 8, 3, padding=1, groups=2)
layer.load_state_dict(stock.state_dict())
```

Do not reinterpret a 3D checkpoint as a 4D kernel: the added axis has independent
weights and requires an explicit initialization/conversion policy.

Choose surrounding layers that accept the tensor's rank: elementwise activations,
GroupNorm and explicit spatial reductions compose with 4D convolution. Stock
BatchNorm3d and MaxPool3d retain their own input-rank requirements; this package
does not replace every higher-dimensional network layer.

## Mixed precision and memory

Move the layer and inputs to the same device. Use ordinary `torch.autocast` and
`torch.amp.GradScaler`; CUDA FP16/BF16 native gradients accumulate in FP32.

```python
model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cuda")
x = torch.randn(2, 2, 4, 4, 4, 8, device="cuda")
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.float16):
    loss = model(x).square().mean()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

For memory-heavy networks, non-reentrant `torch.utils.checkpoint.checkpoint`
works with recomputation and parameter updates. `workspace_bytes` limits one
native patch buffer, not the model's peak memory. The portable high-rank path
also retains indexing/autograd intermediates. Profile a representative full model.

## Compilation, transforms and saving

```python
compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
torch.save(model.state_dict(), "weights.pt")
```

The default Inductor backend is also exercised on CPU/CUDA. Native kernels load
lazily at a compiled graph's first actual execution. Explicit `load_native` can
move build time out of that call. Missing native prerequisites select a portable
automatic path; Inductor's own compiler requirements still apply.

Where provided by the framework, JVP/VJP, vmap, gradients and Jacobians preserve
differentiable operations. Script/trace models also retain tangents after save/reload. Import
`torch_convnd` before loading a serialized TorchScript or `torch.export` graph
in a fresh process so its custom operator metadata is registered.

## ONNX

```python
from torch_convnd import Conv4d, export_onnx

layer = Conv4d(2, 4, 3, padding=1, backend="torch").eval()
example = torch.randn(1, 2, 4, 4, 4, 8)
export_onnx(layer, (example,), "layer.onnx", external_data=False)
```

The helper uses standard real `Conv` nodes and forwards PyTorch exporter options
such as names and `dynamic_shapes`. No custom ONNX runtime plugin is required.
On modern exporters, eager export defaults to opset 22; legacy export chooses an
opset supported by the installed framework, up to 20. BF16 needs opset 22 and a
capable exporter. Runtime provider/rank/dtype support is distinct
from a valid export; see [compatibility](support.md).

## Functional use

```python
from torch_convnd import conv_nd

x = torch.randn(1, 2, 4, 4, 4, 4, requires_grad=True)
weight = torch.randn(8, 2, 3, 3, 3, 3, requires_grad=True)
y = conv_nd(x, weight, padding=1, backend="torch")
```

The public package ships typing metadata and stubs, so consuming projects can
check constructors, backend names and functional signatures without inspecting
private dispatcher or build modules.
