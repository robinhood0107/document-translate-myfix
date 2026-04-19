from __future__ import annotations

import ctypes
import json
import os
import pathlib
import site
import sys
import tempfile
import traceback

import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
import tensorrt as trt


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _try_load_dll(name: str) -> None:
    try:
        ctypes.WinDLL(name)
        print(f"OK   {name}")
    except Exception as exc:
        print(f"FAIL {name}: {exc}")


def _make_test_model(model_path: pathlib.Path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph([node], "relu_graph", [x], [y])
    model = helper.make_model(
        graph,
        producer_name="comic-translate-cuda13-check",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    onnx.save(model, model_path)


def _session_check(model_path: str, providers) -> None:
    provider_name = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        active = session.get_providers()
        print(f"OK   session via {provider_name}: {active}")
    except Exception as exc:
        print(f"FAIL session via {provider_name}: {exc}")
        traceback.print_exc()


def main() -> int:
    _print_header("Environment")
    print(f"python: {sys.executable}")
    print(f"platform: {sys.platform}")
    print(f"PATH entries: {len(os.environ.get('PATH', '').split(os.pathsep))}")

    site_packages = [pathlib.Path(p) for p in site.getsitepackages()]
    venv_root = pathlib.Path(sys.prefix)
    site_root = next((p for p in site_packages if p.name == "site-packages"), site_packages[-1])
    tensorrt_libs_dir = site_root / "tensorrt_libs"
    cudnn_dir = site_root / "nvidia" / "cudnn" / "bin"
    cublas_dir = site_root / "nvidia" / "cublas" / "bin"

    cuda_candidates = [
        pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"),
        pathlib.Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin"),
    ]
    cuda_dir = next((path for path in cuda_candidates if path.exists()), None)

    print(f"venv root: {venv_root}")
    print(f"site-packages: {site_root}")
    print(f"TensorRT libs dir: {tensorrt_libs_dir} exists={tensorrt_libs_dir.exists()}")
    print(f"cuDNN dir: {cudnn_dir} exists={cudnn_dir.exists()}")
    print(f"cuBLAS site dir: {cublas_dir} exists={cublas_dir.exists()}")
    print(f"CUDA dir: {cuda_dir} exists={bool(cuda_dir and cuda_dir.exists())}")

    _print_header("DLL Directories")
    dll_dirs: list[str] = []
    if cuda_dir and cuda_dir.exists():
        os.add_dll_directory(str(cuda_dir))
        dll_dirs.append(str(cuda_dir))
    if tensorrt_libs_dir.exists():
        os.add_dll_directory(str(tensorrt_libs_dir))
        dll_dirs.append(str(tensorrt_libs_dir))
    if cudnn_dir.exists():
        os.add_dll_directory(str(cudnn_dir))
        dll_dirs.append(str(cudnn_dir))
    if cublas_dir.exists():
        os.add_dll_directory(str(cublas_dir))
        dll_dirs.append(str(cublas_dir))
    print(json.dumps(dll_dirs, indent=2))

    _print_header("ORT Build Info")
    try:
        import onnxruntime.capi.build_and_package_info as build_info

        print(f"onnxruntime: {ort.__version__}")
        print(f"cuda_version: {getattr(build_info, 'cuda_version', 'unknown')}")
    except Exception as exc:
        print(f"Unable to read ORT build info: {exc}")

    try:
        if cuda_dir and cuda_dir.exists():
            ort.preload_dlls(cuda=True, cudnn=False, directory=str(cuda_dir))
        ort.preload_dlls(cuda=False, cudnn=True, directory="")
        ort.print_debug_info()
    except Exception as exc:
        print(f"ORT debug info unavailable: {exc}")

    _print_header("Direct DLL Load")
    for dll_name in [
        "cudart64_13.dll",
        "cublas64_13.dll",
        "cublasLt64_13.dll",
        "cudnn64_9.dll",
        "nvinfer_10.dll",
        "nvinfer_plugin_10.dll",
    ]:
        _try_load_dll(dll_name)

    _print_header("TensorRT")
    print(f"tensorrt: {trt.__version__}")
    builder = trt.Builder(trt.Logger())
    print(f"builder ok: {bool(builder)}")

    _print_header("ORT Providers")
    print(ort.get_available_providers())

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = pathlib.Path(temp_dir) / "relu.onnx"
        _make_test_model(model_path)

        _print_header("Session Checks")
        _session_check(
            str(model_path),
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _session_check(
            str(model_path),
            [
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_engine_cache_enable": False,
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
