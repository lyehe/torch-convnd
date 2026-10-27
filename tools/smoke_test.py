"""Exercise an installed package from any working directory, including Python -I."""

import argparse
import importlib.metadata
import io
import json
from pathlib import Path

import torch

import torch_convnd
from torch_convnd import Conv2d, Conv3d, Conv4d, __version__, _build


def compare_training(layer, reference, input, tolerance=2e-5):
    reference.load_state_dict(layer.state_dict())
    actual, expected = layer(input), reference(input)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    cotangent = torch.randn_like(actual) / actual.numel() ** 0.5
    first = torch.autograd.grad(actual, (input, *layer.parameters()), cotangent)
    second = torch.autograd.grad(expected, (input, *reference.parameters()), cotangent)
    for value, target in zip(first, second, strict=True):
        torch.testing.assert_close(value, target, atol=tolerance, rtol=tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["auto", "native", "torch"], default="torch")
    parser.add_argument(
        "--installed", action="store_true", help="require a noneditable wheel import"
    )
    parser.add_argument("--onnx", action="store_true", help="also require ONNX CPU inference")
    args = parser.parse_args()
    package = Path(torch_convnd.__file__).resolve()
    if args.installed and package.parent.parent.name != "site-packages":
        raise AssertionError(f"Imported source instead of installed wheel: {package}")
    assert __version__ == importlib.metadata.version("torch-convnd")
    torch.manual_seed(913)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for rank, factory in [(2, Conv2d), (3, Conv3d)]:
        layer = factory(2, 4, 2, backend=args.backend, device=args.device)
        reference = getattr(torch.nn, f"Conv{rank}d")(2, 4, 2, device=args.device)
        input = torch.randn((1, 2) + (4,) * rank, device=args.device, requires_grad=True)
        compare_training(layer, reference, input)
    layer = Conv4d(2, 4, 2, backend=args.backend, device=args.device)
    reference = Conv4d(2, 4, 2, backend="torch", device=args.device)
    input = torch.randn(1, 2, 3, 3, 3, 3, device=args.device, requires_grad=True)
    compare_training(layer, reference, input)
    tangent = torch.randn_like(input)
    for mode in ("script", "trace"):
        compiled = torch.jit.script(layer) if mode == "script" else torch.jit.trace(layer, input)
        buffer = io.BytesIO()
        torch.jit.save(compiled, buffer)
        buffer.seek(0)
        restored = torch.jit.load(buffer, map_location=args.device)
        if hasattr(torch, "func"):
            actual = torch.func.jvp(restored, (input,), (tangent,))
            expected = torch.func.jvp(reference, (input,), (tangent,))
        else:
            with torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(input, tangent)
                actual = torch.autograd.forward_ad.unpack_dual(restored(dual))
                expected = torch.autograd.forward_ad.unpack_dual(reference(dual))
        assert actual[1].norm() > 0
        for value, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, target, atol=2e-5, rtol=2e-5)
    if args.onnx:
        import tempfile

        import onnxruntime as ort

        from torch_convnd import export_onnx

        model = Conv4d(2, 3, 2, backend=args.backend).eval()
        input = torch.randn(1, 2, 3, 3, 3, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            export_onnx(model, (input,), path, external_data=False)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            actual = session.run(None, {session.get_inputs()[0].name: input.numpy()})[0]
            torch.testing.assert_close(torch.from_numpy(actual), model(input), atol=2e-5, rtol=2e-5)
    if args.backend == "torch":
        assert not _build._LOADED, "portable smoke test unexpectedly built native kernels"
    print(json.dumps({"version": __version__, "package": str(package), "checks": "passed"}))


if __name__ == "__main__":
    main()
