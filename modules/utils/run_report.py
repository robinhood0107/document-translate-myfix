"""배치 한 번의 실측 소요시간과 페이지별 결과를 파일로 남긴다.

지금까지 앱은 실행 기록을 아무 데도 쓰지 않았다. 그래서 두 가지를 할 수 없었다.

* "이전보다 빨라졌는가"에 답할 수 없다. 비교할 과거 실행 기록이 없다.
* 366장을 넣고 347장이 나왔을 때, 사라진 19장이 왜 사라졌는지 알 수 없다.

여기서는 추정이 아니라 **실측값만** 쓴다. 단계별 시간은 파이프라인 텔레메트리가
이미 재고 있던 값이고, 총 시간은 사용자가 버튼을 누른 순간부터 잰다.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Mapping

logger = logging.getLogger(__name__)

RUN_REPORT_SCHEMA_VERSION = 1

# 사람이 읽는 표에서 쓰는 단계 이름.
STAGE_LABELS: dict[str, str] = {
    "pipeline": "전체",
    "detect": "말풍선 감지",
    "ocr": "OCR",
    "translate": "번역",
    "inpaint": "인페인팅",
    "render": "렌더",
}


def format_hms(seconds: float) -> str:
    total = int(round(max(float(seconds), 0.0)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_minutes(seconds: float) -> str:
    return f"{max(float(seconds), 0.0) / 60.0:.1f}분"


def build_run_report(
    *,
    telemetry: Mapping[str, Any],
    total_wall_sec: float,
    page_outcomes: list[Mapping[str, Any]],
    output_summary: Mapping[str, Any],
    started_at_local: str = "",
) -> dict[str, Any]:
    """기계가 읽을 실행 리포트 한 덩어리."""

    stages = telemetry.get("stages")
    stage_rows: list[dict[str, Any]] = []
    if isinstance(stages, Mapping):
        for name, values in stages.items():
            if not isinstance(values, Mapping):
                continue
            wall_ms = float(values.get("wall_ms", 0.0) or 0.0)
            stage_rows.append(
                {
                    "stage": str(name),
                    "label": STAGE_LABELS.get(str(name), str(name)),
                    "seconds": round(wall_ms / 1000.0, 3),
                    "count": int(values.get("count", 0) or 0),
                }
            )
    stage_rows.sort(key=lambda row: row["seconds"], reverse=True)

    page_count = int(output_summary.get("input_count", 0) or 0)
    return {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "started_at_local": started_at_local,
        "total_wall_sec": round(float(total_wall_sec), 3),
        "total_wall_text": format_hms(total_wall_sec),
        "page_count": page_count,
        "seconds_per_page": (
            round(float(total_wall_sec) / page_count, 3) if page_count else None
        ),
        "stages": stage_rows,
        "output": dict(output_summary),
        "pages": [dict(item) for item in page_outcomes],
    }


def render_run_report_text(report: Mapping[str, Any]) -> str:
    """사람이 바로 읽는 요약. 숫자는 리포트에 있는 값을 그대로 쓴다."""

    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("자동 번역 실행 리포트")
    lines.append("=" * 60)
    if report.get("started_at_local"):
        lines.append(f"시작: {report['started_at_local']}")
    lines.append(
        f"전체 소요: {report.get('total_wall_text', '-')} "
        f"({format_minutes(float(report.get('total_wall_sec', 0.0) or 0.0))})"
    )
    page_count = int(report.get("page_count", 0) or 0)
    lines.append(f"페이지: {page_count}장")
    per_page = report.get("seconds_per_page")
    if isinstance(per_page, (int, float)):
        lines.append(f"페이지당: {float(per_page):.2f}초")

    output = report.get("output") or {}
    if isinstance(output, Mapping):
        lines.append("")
        lines.append(
            f"출력: {output.get('output_count', 0)}장 / 입력 {output.get('input_count', 0)}장"
        )
        fallbacks = output.get("fallbacks") or []
        if fallbacks:
            lines.append(f"폴백 저장: {len(fallbacks)}장")
            for item in fallbacks:
                if not isinstance(item, Mapping):
                    continue
                lines.append(
                    f"  - {item.get('image_name', '?')}"
                    f" [{item.get('kind', '?')}]"
                    f" {item.get('failed_stage', '')}: {item.get('reason', '')}".rstrip()
                )
        missing = output.get("missing") or []
        if missing:
            lines.append(f"누락(저장 실패): {len(missing)}장 -> {', '.join(missing)}")

    stages = report.get("stages") or []
    if stages:
        lines.append("")
        lines.append("단계별 실측 소요 (긴 순서)")
        lines.append("-" * 60)
        total = float(report.get("total_wall_sec", 0.0) or 0.0)
        for row in stages:
            if not isinstance(row, Mapping):
                continue
            seconds = float(row.get("seconds", 0.0) or 0.0)
            share = f"{(seconds / total * 100.0):5.1f}%" if total > 0 else "    -"
            lines.append(
                f"  {str(row.get('label', row.get('stage', '?'))):<12}"
                f" {format_hms(seconds):>9}"
                f" {format_minutes(seconds):>9}"
                f" {share}"
            )
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def write_run_report(report: Mapping[str, Any], *, log_dir: str) -> str:
    """리포트를 ``log_dir`` 에 JSON 과 텍스트로 쓴다. 경로를 돌려준다.

    쓰기에 실패해도 배치를 실패시키지 않는다. 리포트는 진단 자료이지 결과물이
    아니다.
    """

    try:
        os.makedirs(log_dir, exist_ok=True)
        stamp = str(report.get("started_at_local") or "").replace(":", "-").replace(" ", "_")
        if not stamp:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        json_path = os.path.join(log_dir, f"run_{stamp}.json")
        text_path = os.path.join(log_dir, f"run_{stamp}.txt")
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(render_run_report_text(report))
        logger.info("Wrote the run report to %s", text_path)
        return text_path
    except Exception:
        logger.warning("Could not write the run report.", exc_info=True)
        return ""
