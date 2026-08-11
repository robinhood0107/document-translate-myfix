"""인페인팅 후처리의 순수 계산부.

인페인팅 스윕은 페이지마다 세 구간을 완전히 직렬로 돈다. 366장 실측:

    mask_generation        1.335s/page   GPU (CTD 신경망)
    model_forward          1.309s/page   GPU (LaMa)
    cleanup_and_composite  0.991s/page   CPU

앞의 둘은 같은 GPU를 쓰므로 겹쳐도 이득이 없다. 반면 후처리는 순수 CPU 이고,
같은 스윕에서 도는 렌더 워커는 페이지당 0.370초만 쓰고 거의 놀고 있다(큐 대기
0.0002초, 최종 drain 0.015초). 그래서 후처리를 파이프라인 스레드에서 들어내면
GPU 뒤에 숨는다.

여기서는 **값만 받아 값만 돌려준다.** `main_page` 도, 시그널도, 잠금도 만지지
않는다. 그래야 워커 스레드에서 안전하게 돌고, 순수 함수라 결과 동등성을 그대로
검증할 수 있다. 파이프라인 스레드가 해야 하는 일(진행 보고, 체크포인트 기록,
바깥 픽셀 오염 검사)은 호출부에 남는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import imkit as imk
import numpy as np

from modules.utils.inpaint_cleanup import (
    apply_duplicate_bubble_inner_fill,
)
from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
    normalize_edit_mask,
)
from modules.utils.inpaint_evidence import BlockInpaintEvidence


@dataclass
class InpaintCleanupInput:
    """후처리 한 페이지분 입력. 전부 값이다."""

    image: np.ndarray
    inpaint_input_img: np.ndarray
    mask: np.ndarray
    mask_details: dict[str, Any]
    inpaint_blocks: list[Any]
    config: Any
    page_label: str
    # 모델이 실제로 고친 영역. 없으면 마스크를 그대로 쓴다.
    inpaint_edit_mask: np.ndarray | None = None
    routing_evidence: tuple[BlockInpaintEvidence, ...] = ()


@dataclass
class InpaintCleanupResult:
    inpaint_input_img: np.ndarray
    mask: np.ndarray
    cleanup_stats: dict[str, Any] = field(default_factory=dict)
    # 복원 전후로 마스크 **바깥**이 몇 픽셀 바뀌었는지. 0 이 아니면 인페인팅이
    # 마스크를 넘어 번진 것이므로 호출부가 그 페이지를 실패시킨다.
    outside_before_restore: int = 0
    outside_after_restore: int = 0
    worker_seconds: float = 0.0


def run_inpaint_cleanup(job: InpaintCleanupInput) -> InpaintCleanupResult:
    """인페인팅 결과를 정리하고 마스크 안으로만 합성한다.

    순서는 원래 스윕 안에 있던 것과 글자 그대로 같다. 순서가 결과를 바꾸므로
    바꾸지 않는다.
    """

    import time

    started = time.monotonic()

    inpainted = imk.convert_scale_abs(job.inpaint_input_img)
    mask = job.mask
    if job.inpaint_edit_mask is not None:
        mask = np.where(
            (mask > 0) | (job.inpaint_edit_mask > 0),
            255,
            0,
        ).astype(np.uint8)

    cleanup_stats = {"autonomous_residue_cleanup": "disabled"}
    inpainted, mask, cleanup_stats = apply_duplicate_bubble_inner_fill(
        inpainted,
        mask,
        job.mask_details,
        cleanup_stats,
    )
    protected_corner_mask = normalize_edit_mask(
        job.mask_details.get("protected_corner_mask"),
        job.image.shape,
    )
    if np.any(protected_corner_mask):
        mask = np.where(
            (normalize_edit_mask(mask, job.image.shape) > 0)
            & (protected_corner_mask <= 0),
            255,
            0,
        ).astype(np.uint8)

    outside_before = count_changed_outside_edit_mask(job.image, inpainted, mask)
    inpainted = composite_with_edit_mask(job.image, inpainted, mask)
    outside_after = count_changed_outside_edit_mask(job.image, inpainted, mask)

    return InpaintCleanupResult(
        inpaint_input_img=np.ascontiguousarray(inpainted, dtype=np.uint8),
        mask=np.ascontiguousarray(mask, dtype=np.uint8),
        cleanup_stats=cleanup_stats,
        outside_before_restore=int(outside_before),
        outside_after_restore=int(outside_after),
        worker_seconds=max(0.0, time.monotonic() - started),
    )
