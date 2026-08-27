from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from modules.inpainting.runtime_contract import (
    INPAINT_RETRY_POLICY_VERSION,
    InpaintingCudaOOMError,
    InpaintingRuntimeContractError,
    bounded_retry_roi,
    inspect_learned_inpainter_runtime,
    validate_learned_inpaint_runtime,
)
from modules.inpainting.source_lama_blockwise import SourceLaMaLarge
from modules.utils.inpaint_evidence import SourceLamaBlockwiseResult
from modules.inpainting.ffc_torch import FourierUnit
from pipeline.inpainting import InpaintingHandler


def _fake_cuda_model() -> SimpleNamespace:
    return SimpleNamespace(
        generator=SimpleNamespace(
            parameters=lambda: iter(
                [
                    SimpleNamespace(
                        device="cuda:0",
                        dtype=torch.float32,
                    )
                ]
            ),
        )
    )


def test_learned_inpainting_rejects_cpu_without_fallback() -> None:
    with pytest.raises(
        InpaintingRuntimeContractError,
        match="requires CUDA",
    ):
        validate_learned_inpaint_runtime(
            inpainter_key="lama_large_512px",
            device="cpu",
            precision="fp32",
        )


def test_fp32_cuda_is_promotion_eligible_but_bf16_is_not() -> None:
    fp32 = validate_learned_inpaint_runtime(
        inpainter_key="lama_large_512px",
        device="cuda",
        precision="fp32",
    )
    bf16 = validate_learned_inpaint_runtime(
        inpainter_key="lama_large_512px",
        device="cuda",
        precision="bf16",
    )

    assert fp32["fp32_promotion_eligible"] is True
    assert bf16["fp32_promotion_eligible"] is False
    assert fp32["cpu_fallback_used"] is False


def test_runtime_inspection_rejects_model_left_on_cpu() -> None:
    inpainter = SimpleNamespace(
        runtime_device="cuda",
        precision="fp32",
        model=torch.nn.Linear(2, 2),
    )

    with pytest.raises(
        InpaintingRuntimeContractError,
        match="requires CUDA",
    ):
        inspect_learned_inpainter_runtime(
            inpainter,
            inpainter_key="lama_large_512px",
            requested_device="cuda",
            requested_precision="fp32",
        )


def test_runtime_inspection_verifies_nested_lama_generator_on_cuda() -> None:
    nested_generator = SimpleNamespace(
        parameters=lambda: iter(
            [
                SimpleNamespace(
                    device="cuda:0",
                    dtype=torch.float32,
                )
            ]
        ),
    )
    inpainter = SimpleNamespace(
        runtime_device="cuda",
        precision="fp32",
        model=SimpleNamespace(generator=nested_generator),
    )

    report = inspect_learned_inpainter_runtime(
        inpainter,
        inpainter_key="lama_large_512px",
        requested_device="cuda",
        requested_precision="fp32",
    )

    assert report["model_parameter_device"] == "cuda:0"
    assert report["device_verified_from_model"] is True


def test_runtime_inspection_rejects_cpu_only_onnx_session() -> None:
    inpainter = SimpleNamespace(
        runtime_device="cuda",
        precision="fp32",
        model=None,
        session=SimpleNamespace(
            get_providers=lambda: ["CPUExecutionProvider"]
        ),
    )

    with pytest.raises(
        InpaintingRuntimeContractError,
        match="requires CUDA",
    ):
        inspect_learned_inpainter_runtime(
            inpainter,
            inpainter_key="AOT",
            requested_device="cuda",
            requested_precision="fp32",
        )


@pytest.mark.parametrize("fft_name", ["rfftn", "irfftn"])
def test_fourier_unit_never_falls_back_to_cpu(
    monkeypatch,
    fft_name: str,
) -> None:
    unit = FourierUnit(1, 1)
    source = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

    monkeypatch.setattr(
        torch.fft,
        fft_name,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic FFT failure")
        ),
    )

    with pytest.raises(
        InpaintingRuntimeContractError,
        match="CPU fallback is disabled",
    ):
        unit(source)


def test_bounded_retry_roi_is_smaller_and_contains_the_mask() -> None:
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:120, 130:170] = 255

    roi = bounded_retry_roi(mask, (200, 300, 3))

    assert roi is not None
    assert roi.area < 200 * 300
    assert roi.x1 <= 130 < 170 <= roi.x2
    assert roi.y1 <= 80 < 120 <= roi.y2


def test_source_lama_oom_retries_once_on_tighter_cuda_roi(
    monkeypatch,
) -> None:
    inpainter = object.__new__(SourceLaMaLarge)
    inpainter.device = "cuda"
    inpainter.precision = "fp32"
    inpainter.inpaint_size = 2048
    inpainter.model = _fake_cuda_model()
    inpainter.run_diagnostics = []
    inpainter.ensure_loaded = lambda: None
    calls: list[tuple[int, int]] = []

    def fake_inpaint(image, mask, _blocks):
        calls.append(image.shape[:2])
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        output = image.copy()
        output[mask > 0] = 77
        return output

    inpainter._inpaint = fake_inpaint
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:120, 130:170] = 255

    result = inpainter.memory_safe_inpaint(image, mask)

    assert len(calls) == 2
    assert calls[1][0] < calls[0][0]
    assert calls[1][1] < calls[0][1]
    assert inpainter.device == "cuda"
    assert inpainter.precision == "fp32"
    assert np.all(result[mask > 0] == 77)
    assert np.count_nonzero(result[mask <= 0]) == 0
    assert inpainter.run_diagnostics[-1]["oom_retry_count"] == 1
    assert (
        inpainter.run_diagnostics[-1]["retry_policy"]
        == INPAINT_RETRY_POLICY_VERSION
    )
    assert inpainter.run_diagnostics[-1]["cpu_fallback_used"] is False


def test_source_lama_oom_failure_exposes_retry_diagnostics(
    monkeypatch,
) -> None:
    inpainter = object.__new__(SourceLaMaLarge)
    inpainter.device = "cuda"
    inpainter.precision = "fp32"
    inpainter.inpaint_size = 2048
    inpainter.model = _fake_cuda_model()
    inpainter.run_diagnostics = []
    inpainter.ensure_loaded = lambda: None
    inpainter._inpaint = lambda *_args: (_ for _ in ()).throw(
        torch.cuda.OutOfMemoryError("CUDA out of memory")
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:120, 130:170] = 255

    with pytest.raises(InpaintingCudaOOMError) as raised:
        inpainter.memory_safe_inpaint(image, mask)

    assert raised.value.diagnostics["oom_retry_count"] == 1
    assert raised.value.diagnostics["status"] == "failed_after_roi_retry"
    assert raised.value.diagnostics["cpu_fallback_used"] is False


class _FakeInpainter:
    def __init__(
        self,
        _device,
        *,
        backend,
        runtime_device,
        inpaint_size,
        precision,
    ) -> None:
        self.backend = backend
        self.runtime_device = runtime_device
        self.inpaint_size = inpaint_size
        self.precision = precision
        self.model = _fake_cuda_model()


def test_handler_reloads_same_model_when_runtime_profile_changes() -> None:
    runtime = {
        "key": "lama_large_512px",
        "backend": "torch",
        "device": "cuda",
        "inpaint_size": 1536,
        "precision": "bf16",
    }
    settings_page = SimpleNamespace(is_gpu_enabled=lambda: True)
    handler = InpaintingHandler(
        SimpleNamespace(settings_page=settings_page)
    )
    release_report = {
        "vram_release_gate": {
            "required": True,
            "observed": True,
            "status": "observed",
        }
    }

    with mock.patch(
        "pipeline.inpainting.get_inpainter_runtime",
        side_effect=lambda _settings: dict(runtime),
    ), mock.patch(
        "pipeline.inpainting.resolve_device",
        return_value="cuda",
    ), mock.patch(
        "pipeline.inpainting.query_cuda_handoff_metrics",
        return_value={},
    ), mock.patch.dict(
        "pipeline.inpainting.inpaint_map",
        {"lama_large_512px": _FakeInpainter},
    ), mock.patch.object(
        handler,
        "release_inpainter_resources",
        return_value=release_report,
    ) as release_inpainter:
        first = handler._ensure_inpainter()
        runtime["precision"] = "fp32"
        second = handler._ensure_inpainter()

    release_inpainter.assert_called_once_with()
    assert handler.last_profile_change_release == release_report
    assert first is not second
    assert first.precision == "bf16"
    assert second.precision == "fp32"


def test_handler_refuses_profile_reload_without_confirmed_vram_release() -> None:
    runtime = {
        "key": "lama_large_512px",
        "backend": "torch",
        "device": "cuda",
        "inpaint_size": 2048,
        "precision": "fp32",
    }
    settings_page = SimpleNamespace(is_gpu_enabled=lambda: True)
    handler = InpaintingHandler(
        SimpleNamespace(settings_page=settings_page)
    )
    handler.inpainter_cache = _FakeInpainter(
        "cuda",
        backend="torch",
        runtime_device="cuda",
        inpaint_size=1536,
        precision="bf16",
    )
    handler.cached_inpainter_runtime_signature = (
        "lama_large_512px",
        "torch",
        "cuda",
        1536,
        "bf16",
    )

    with mock.patch(
        "pipeline.inpainting.get_inpainter_runtime",
        return_value=runtime,
    ), mock.patch.object(
        handler,
        "release_inpainter_resources",
        return_value={
            "vram_release_gate": {
                "required": True,
                "observed": False,
                "status": "timeout",
            }
        },
    ):
        with pytest.raises(
            InpaintingRuntimeContractError,
            match="VRAM release was not confirmed",
        ):
            handler._ensure_inpainter()


def test_handler_losslessly_restores_every_pixel_outside_edit_mask() -> None:
    handler = InpaintingHandler(SimpleNamespace(settings_page=object()))
    handler.cached_inpainter_key = "lama_large_512px"
    handler.inpainter_cache = SimpleNamespace(
        runtime_device="cuda",
        precision="fp32",
        inpaint_size=2048,
    )
    handler._ensure_inpainter = lambda: handler.inpainter_cache
    image = np.full((8, 8, 3), 10, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3:5, 3:5] = 255
    unsafe_result = np.full_like(image, 200)

    with mock.patch(
        "pipeline.inpainting.source_lama_blockwise_inpaint_result",
        return_value=SourceLamaBlockwiseResult(
            image=unsafe_result,
            edit_mask=mask.copy(),
            diagnostics=[],
        ),
    ) as run_lama:
        result = handler.inpaint_with_blocks(
            image,
            mask,
            [],
            config=object(),
        )

    assert run_lama.call_args.kwargs["check_need_inpaint"] is True
    np.testing.assert_array_equal(result[mask <= 0], image[mask <= 0])
    np.testing.assert_array_equal(result[mask > 0], unsafe_result[mask > 0])
    assert (
        handler.last_inpaint_diagnostics[
            "outside_mask_changed_pixel_count"
        ]
        == 0
    )
    assert handler.last_inpaint_diagnostics["cpu_fallback_used"] is False
