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
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

RUN_REPORT_SCHEMA_VERSION = 2

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


# 한 단계가 실제로 점유한 벽시계 구간을 재는 연산 이름. 나머지 연산은 페이지마다
# 여러 번, 서로 겹치게 측정되므로 더하면 실행 시간을 훌쩍 넘는다.
STAGE_WINDOW_OPERATION = "stage_window"


def _stage_rows(telemetry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """단계별 벽시계 소요. ``stage_window`` 만 쓴다.

    ``telemetry["stages"]`` 는 그 단계에서 일어난 **모든** 측정의 합이다. 중첩되고
    동시에 도는 페이지별 측정까지 전부 더해지므로 실행 시간과 비교할 수 없다.
    실측 예: 41분 실행에서 ``page`` 가 202시간, ``inpaint`` 가 64시간으로 찍혔다.
    단계가 실제로 점유한 구간은 ``stage_details[stage]["stage_window"]`` 하나뿐이다.
    """

    details = telemetry.get("stage_details")
    rows: list[dict[str, Any]] = []
    if isinstance(details, Mapping):
        for name, operations in details.items():
            if not isinstance(operations, Mapping):
                continue
            window = operations.get(STAGE_WINDOW_OPERATION)
            if not isinstance(window, Mapping):
                continue
            wall_ms = float(window.get("wall_ms", 0.0) or 0.0)
            rows.append(
                {
                    "stage": str(name),
                    "label": STAGE_LABELS.get(str(name), str(name)),
                    "seconds": round(wall_ms / 1000.0, 3),
                    "count": int(window.get("count", 0) or 0),
                }
            )
    rows.sort(key=lambda row: row["seconds"], reverse=True)
    return rows


def _stage_work_rows(telemetry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """단계별 **작업량** 합계. 그 단계의 세부 연산을 더한 값이다.

    벽시계 구간과 나란히 놓으면 동시성이 얼마나 먹히는지 드러난다. 작업 합계가
    벽시계 구간보다 크면 그만큼 겹쳐서 숨겼다는 뜻이다.

    ``telemetry["stages"]`` 를 쓰지 않는 이유는 그 값이 여러 출처를 한 통에
    담기 때문이다. 실측으로 42분 실행에서 ``inpaint`` 가 4030분으로 찍혔는데,
    그 단계의 세부 연산을 다 더해도 21분이다. 여기서는 세부 연산만 더하므로
    합계가 항상 설명 가능하다.
    """

    details = telemetry.get("stage_details")
    rows: list[dict[str, Any]] = []
    if isinstance(details, Mapping):
        for name, operations in details.items():
            if not isinstance(operations, Mapping):
                continue
            seconds = 0.0
            count = 0
            for operation, metrics in operations.items():
                if operation == STAGE_WINDOW_OPERATION or not isinstance(
                    metrics, Mapping
                ):
                    continue
                seconds += float(metrics.get("wall_ms", 0.0) or 0.0) / 1000.0
                count = max(count, int(metrics.get("count", 0) or 0))
            if seconds <= 0.0:
                continue
            rows.append(
                {
                    "stage": str(name),
                    "label": STAGE_LABELS.get(str(name), str(name)),
                    "work_seconds": round(seconds, 3),
                    # 연산마다 호출 횟수가 다르므로 최대값을 페이지 수 대용으로 쓴다.
                    "count": count,
                    "seconds_per_measurement": (
                        round(seconds / count, 4) if count else None
                    ),
                }
            )
    rows.sort(key=lambda row: row["work_seconds"], reverse=True)
    return rows


def _operation_rows(telemetry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """단계 안의 세부 연산별 합계. 어디에 시간이 가는지 여기서 갈린다.

    ``stage_window`` 는 단계 전체 구간이라 여기서는 뺀다. 남는 것이 실제 작업이다.
    렌더의 ``worker``/``queue_wait``/``tail_drain``, 번역의
    ``inference_and_cache``/``dictionary_and_sanitizer`` 가 이 표에 나온다.
    """

    details = telemetry.get("stage_details")
    rows: list[dict[str, Any]] = []
    if isinstance(details, Mapping):
        for stage, operations in details.items():
            if not isinstance(operations, Mapping):
                continue
            for operation, metrics in operations.items():
                if operation == STAGE_WINDOW_OPERATION or not isinstance(
                    metrics, Mapping
                ):
                    continue
                seconds = float(metrics.get("wall_ms", 0.0) or 0.0) / 1000.0
                count = int(metrics.get("count", 0) or 0)
                rows.append(
                    {
                        "stage": str(stage),
                        "operation": str(operation),
                        "seconds": round(seconds, 3),
                        "count": count,
                        "seconds_each": round(seconds / count, 4) if count else None,
                    }
                )
    rows.sort(key=lambda row: row["seconds"], reverse=True)
    return rows


def build_run_report(
    *,
    telemetry: Mapping[str, Any] | None,
    total_wall_sec: float | None,
    page_outcomes: list[Mapping[str, Any]],
    output_summary: Mapping[str, Any],
    started_at_local: str = "",
    run_id: str = "",
    terminal_status: str = "unknown",
    processing_status: str = "unknown",
    report_status: str = "unknown",
    cleanup_status: str = "unknown",
    pipeline_cleanup_status: str = "unknown",
    application_cleanup_status: str = "unknown",
    archive_status: str = "not_applicable",
    processing_error: Mapping[str, Any] | None = None,
    cleanup_errors: list[Mapping[str, Any]] | None = None,
    telemetry_reason_code: str = "telemetry_snapshot_unavailable",
) -> dict[str, Any]:
    """기계가 읽을 실행 리포트 한 덩어리."""

    page_outcomes = [dict(item) for item in page_outcomes]
    for item in page_outcomes:
        source_units = item.get("source_units")
        source_units = (
            [row for row in source_units if isinstance(row, Mapping)]
            if isinstance(source_units, list)
            else []
        )
        if any(row.get("status") == "failed" for row in source_units):
            item["quality_status"] = "failed"
        elif item.get("quality_status") not in {"failed", "review_required"} and any(
            row.get("status") == "unjudged"
            and row.get("reason_code")
            in {"review_required", "ocr_review_required", "ocr_rejected_empty"}
            for row in source_units
        ):
            item["quality_status"] = "review_required"
        elif item.get("quality_status") not in {"failed", "review_required"} and any(
            row.get("status") == "unjudged" for row in source_units
        ):
            item["quality_status"] = "incomplete"

    telemetry_data = dict(telemetry) if telemetry is not None else {}
    stage_rows = _stage_rows(telemetry_data)
    input_count = output_summary.get("input_count")
    input_count_measured = isinstance(input_count, int) and not isinstance(
        input_count, bool
    )
    page_count = max(0, input_count) if input_count_measured else None
    review_page_count = (
        sum(
            item.get("quality_status") == "review_required"
            or any(row.get("review_required") for row in item.get("block_failure_ledger", []))
            for item in page_outcomes
        )
        if page_outcomes
        else None
    )
    missing = output_summary.get("missing")
    missing_count = len(missing) if isinstance(missing, list) else None
    page_indexes_by_name: dict[str, list[int]] = {}
    failed_units: set[int | tuple[str, int]] = set()
    for index, item in enumerate(page_outcomes):
        source_index = item.get("source_unit_index", index)
        if isinstance(source_index, int) and not isinstance(source_index, bool):
            page_indexes_by_name.setdefault(str(item.get("image_name") or ""), []).append(source_index)
            if bool(item.get("failed_stage")) or item.get("quality_status") == "failed":
                failed_units.add(source_index)
    missing_indexes = output_summary.get("missing_source_unit_indices")
    if isinstance(missing_indexes, list):
        failed_units.update(
            index
            for index in missing_indexes
            if isinstance(index, int) and not isinstance(index, bool)
        )
    elif isinstance(missing, list):
        for index, item in enumerate(missing):
            if isinstance(item, Mapping) and isinstance(item.get("source_unit_index"), int):
                failed_units.add(item["source_unit_index"])
                continue
            name = str(item)
            matches = page_indexes_by_name.get(name, [])
            failed_units.add(matches[0] if len(matches) == 1 else ("missing", index))
    failed_unit_count = (
        len(failed_units)
        if page_outcomes or missing_count is not None or isinstance(missing_indexes, list)
        else None
    )
    fallbacks = output_summary.get("fallbacks")
    source_fallback_indexes: set[int] = set()
    source_fallback_names: set[str] = set()
    source_fallback_unindexed_count = 0
    for index, item in enumerate(fallbacks if isinstance(fallbacks, list) else []):
        if not isinstance(item, Mapping) or item.get("kind") not in {"source", "source-reloaded"}:
            continue
        source_index = item.get("source_unit_index")
        if isinstance(source_index, int) and not isinstance(source_index, bool):
            source_fallback_indexes.add(source_index)
            continue
        name = str(item.get("image_name") or "")
        matches = page_indexes_by_name.get(name, [])
        if len(matches) == 1:
            source_fallback_indexes.add(matches[0])
        else:
            source_fallback_names.add(name)
            source_fallback_unindexed_count += 1
    source_fallback_count = (
        len(source_fallback_indexes) + source_fallback_unindexed_count
        if isinstance(fallbacks, list)
        else None
    )
    completed_page_count = (
        sum(
            item.get("quality_status") == "completed"
            and item.get("output_status", "measured" if item.get("output_path") else "unmeasured")
            == "measured"
            and item.get("source_unit_index", index) not in source_fallback_indexes
            and item.get("source_unit_index", index) not in failed_units
            and str(item.get("image_name") or "") not in source_fallback_names
            for index, item in enumerate(page_outcomes)
        )
        if page_outcomes
        else None
    )
    output_count = output_summary.get("output_count")
    output_count_measured = isinstance(output_count, int) and not isinstance(
        output_count, bool
    )
    output = dict(output_summary)
    output["measurement"] = {
        "value": output_count if output_count_measured else None,
        "unit": "source_pages",
        "status": "measured" if output_count_measured else "unavailable",
        "reason_code": None if output_count_measured else "output_reconciliation_not_observed",
    }
    output["measurement_status"] = output["measurement"]["status"]
    output["measurement_reason_code"] = output["measurement"]["reason_code"]
    unjudged_count = (
        sum(
            item.get("quality_status") not in {"completed", "failed", "review_required"}
            for item in page_outcomes
        )
        if page_count is not None
        else None
    )
    row_output_unmeasured = sum(
        item.get("output_status", "measured" if item.get("output_path") else "unmeasured")
        != "measured"
        for item in page_outcomes
    )
    unmeasured_count = (
        max(page_count - len(page_outcomes), 0) + row_output_unmeasured
        if page_count is not None
        else None
    )
    wall_time = (
        round(float(total_wall_sec), 3)
        if isinstance(total_wall_sec, (int, float)) and not isinstance(total_wall_sec, bool)
        else None
    )
    source_unit_mapping_by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    source_unit_mapping_unidentified: list[dict[str, Any]] = []
    source_unit_mapping_pages_observed = 0
    source_unit_mapping_pages_partial = 0
    output_path_by_image_path = {
        str(page.get("image_path")): page.get("output_path") or None
        for page in page_outcomes
        if isinstance(page.get("image_path"), str) and page.get("image_path")
    }
    source_unit_statuses = {
        "translated",
        "preserved",
        "intentionally_skipped",
        "failed",
        "unjudged",
    }
    for page in page_outcomes:
        units = page.get("source_units")
        if (
            page.get("source_unit_mapping_status")
            not in {"observed", "partially_observed"}
            or not isinstance(units, list)
        ):
            continue
        source_unit_mapping_pages_observed += 1
        if page.get("source_unit_mapping_status") == "partially_observed":
            source_unit_mapping_pages_partial += 1
        for unit in units:
            if not isinstance(unit, Mapping):
                source_unit_mapping_unidentified.append(
                    {
                        "source_image_path": page.get("image_path") or None,
                        "source_image_name": page.get("image_name"),
                        "source_unit_id": None,
                        "status": "unjudged",
                        "reason_code": "source_unit_record_invalid",
                        "output_unit_ids": [],
                        "output_group_ids": [],
                        "output_units": [],
                        "output_relation_status": "unavailable",
                        "output_relation_reason_code": "source_unit_record_invalid",
                        "semantic_quality_status": "unjudged",
                    }
                )
                continue
            source_unit_id = unit.get("source_unit_id")
            source_unit_id = (
                source_unit_id
                if isinstance(source_unit_id, str) and source_unit_id
                else None
            )
            source_image_path = unit.get("source_image_path")
            if "source_image_path" not in unit:
                source_image_path = page.get("image_path")
            elif not isinstance(source_image_path, str) or not source_image_path:
                source_image_path = None
            source_image_name = unit.get("source_image_name")
            source_image_name = (
                source_image_name
                if isinstance(source_image_name, str) and source_image_name
                else os.path.basename(source_image_path)
                if isinstance(source_image_path, str) and source_image_path
                else None
            )
            status = unit.get("status")
            if status not in source_unit_statuses:
                status = "unjudged"
            reason_code = unit.get("reason_code")
            reason_code = (
                reason_code
                if isinstance(reason_code, str) and reason_code
                else None
            )
            output_unit_ids = unit.get("output_unit_ids")
            output_unit_ids = (
                [value for value in output_unit_ids if isinstance(value, str) and value]
                if isinstance(output_unit_ids, list)
                else []
            )
            relation_status = unit.get("output_relation_status")
            if relation_status not in {"observed", "unavailable"}:
                relation_status = "observed" if output_unit_ids else "unavailable"
            relation_reason = unit.get("output_relation_reason_code")
            relation_reason = (
                relation_reason
                if isinstance(relation_reason, str) and relation_reason
                else None
            )
            if relation_status == "unavailable" and relation_reason is None:
                relation_reason = "output_relation_not_observed"
            provided_output_units = unit.get("output_units")
            output_units = [
                {
                    "output_unit_id": output["output_unit_id"],
                    "output_page_image_path": output.get(
                        "output_page_image_path"
                    ) or None,
                    "output_page_path": output.get("output_page_path")
                    or output_path_by_image_path.get(
                        str(output.get("output_page_image_path") or "")
                    ),
                    "output_fallback_kind": output.get("output_fallback_kind") or None,
                }
                for output in list(provided_output_units or [])
                if isinstance(output, Mapping)
                and isinstance(output.get("output_unit_id"), str)
                and output.get("output_unit_id")
            ] if isinstance(provided_output_units, list) and provided_output_units else [
                {
                    "output_unit_id": output_id,
                    "output_page_image_path": page.get("image_path") or None,
                    "output_page_path": page.get("output_path") or None,
                    "output_fallback_kind": page.get("fallback_kind") or None,
                }
                for output_id in output_unit_ids
            ]
            output_unit_ids = list(
                dict.fromkeys(
                    output_unit_ids
                    + [output["output_unit_id"] for output in output_units]
                )
            )
            output_group_ids = unit.get("output_group_ids")
            output_group_ids = (
                list(
                    dict.fromkeys(
                        value
                        for value in output_group_ids
                        if isinstance(value, str) and value
                    )
                )
                if isinstance(output_group_ids, list)
                else []
            )
            entry = {
                "source_image_path": source_image_path or None,
                "source_image_name": source_image_name,
                "source_unit_id": source_unit_id,
                "status": status,
                "reason_code": reason_code,
                "output_unit_ids": output_unit_ids,
                "output_group_ids": output_group_ids,
                "output_units": output_units,
                "output_relation_status": relation_status,
                "output_relation_reason_code": relation_reason,
                "semantic_quality_status": "unjudged",
            }
            if source_unit_id is None:
                source_unit_mapping_unidentified.append(entry)
                continue
            identity = (source_image_path, source_unit_id)
            existing = source_unit_mapping_by_identity.get(identity)
            if existing is None:
                source_unit_mapping_by_identity[identity] = entry
                continue
            for output in output_units:
                if output not in existing["output_units"]:
                    existing["output_units"].append(output)
                if output["output_unit_id"] not in existing["output_unit_ids"]:
                    existing["output_unit_ids"].append(output["output_unit_id"])
            for group_id in output_group_ids:
                if group_id not in existing["output_group_ids"]:
                    existing["output_group_ids"].append(group_id)
            if existing["status"] != status:
                existing["status"] = "unjudged"
                existing["reason_code"] = "source_unit_status_conflict"
            elif existing["reason_code"] != reason_code:
                existing["reason_code"] = "source_unit_reason_conflict"
            if relation_status == "observed":
                existing["output_relation_status"] = "observed"
                existing["output_relation_reason_code"] = None
    source_unit_mapping_entries = [
        *source_unit_mapping_by_identity.values(),
        *source_unit_mapping_unidentified,
    ]
    source_unit_mapping_status = (
        "unavailable"
        if not page_outcomes or source_unit_mapping_pages_observed == 0
        else "observed"
        if source_unit_mapping_pages_observed == len(page_outcomes)
        and source_unit_mapping_pages_partial == 0
        else "partially_observed"
    )
    quality_status = (
        "failed"
        if failed_unit_count or processing_status == "failed"
        else "review_required"
        if review_page_count
        else "incomplete"
        if processing_status in {"running", "cancelled", "interrupted", "paused"}
        or page_count is None
        or not output_count_measured
        or bool(unjudged_count)
        or bool(unmeasured_count)
        or bool(source_fallback_count)
        else "completed"
    )
    overall_terminal_status = terminal_status
    if terminal_status == "completed":
        overall_terminal_status = (
            "failed"
            if quality_status == "failed"
            else "completed"
            if quality_status == "completed"
            else "incomplete"
        )
    return {
        "schema_version": RUN_REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "terminal_status": overall_terminal_status,
        "processing_status": processing_status,
        "report_status": report_status,
        "cleanup_status": cleanup_status,
        "pipeline_cleanup_status": pipeline_cleanup_status,
        "application_cleanup_status": application_cleanup_status,
        "archive_status": archive_status,
        "processing_error": dict(processing_error) if processing_error else None,
        "cleanup_errors": [dict(item) for item in cleanup_errors or []],
        "started_at_local": started_at_local,
        "total_wall_sec": wall_time,
        "total_wall_measurement": {
            "value": wall_time,
            "unit": "seconds",
            "status": "measured" if wall_time is not None else "unavailable",
            "reason_code": None if wall_time is not None else "wall_time_not_observed",
        },
        "total_wall_text": format_hms(wall_time) if wall_time is not None else "미측정",
        "page_count": page_count,
        "seconds_per_page": (
            round(wall_time / page_count, 3)
            if page_count and wall_time is not None
            else None
        ),
        "telemetry": {
            "value": telemetry_data if telemetry is not None else None,
            "status": "collected" if telemetry is not None else "unavailable",
            "reason_code": None if telemetry is not None else telemetry_reason_code,
        },
        "resource_collection": {
            "status": (
                "collected"
                if isinstance(telemetry_data.get("resources"), Mapping)
                and telemetry_data["resources"].get("sampling_enabled")
                else "not_enabled"
                if isinstance(telemetry_data.get("resources"), Mapping)
                else "unavailable"
            ),
            "reason_code": (
                "resource_sampling_disabled"
                if isinstance(telemetry_data.get("resources"), Mapping)
                and not telemetry_data["resources"].get("sampling_enabled")
                else "resource_snapshot_unavailable"
                if not isinstance(telemetry_data.get("resources"), Mapping)
                else None
            ),
        },
        "stages": stage_rows,
        "stage_work": _stage_work_rows(telemetry_data),
        "operations": _operation_rows(telemetry_data),
        "output": output,
        "source_unit_mapping": {
            "status": source_unit_mapping_status,
            "reason_code": (
                None
                if source_unit_mapping_status == "observed"
                else "source_unit_mapping_not_fully_observed"
            ),
            "unit": "detected_text_blocks",
            "semantic_quality_status": "unjudged",
            "entries": source_unit_mapping_entries,
        },
        "quality": {
            "status": quality_status,
            "review_page_count": review_page_count,
            "failed_page_count": failed_unit_count,
            "required_units": {
                "value": page_count,
                "unit": "source_pages",
                "status": "measured" if input_count_measured else "unavailable",
                "reason_code": None if input_count_measured else "input_count_not_observed",
            },
            "completed_units": {
                "value": completed_page_count,
                "unit": "source_pages",
                "status": "measured" if page_outcomes else "unavailable",
                "reason_code": None if page_outcomes else "page_outcomes_not_observed",
            },
            "review_units": {
                "value": review_page_count,
                "unit": "source_pages",
                "status": "measured" if page_outcomes else "unavailable",
                "reason_code": None if page_outcomes else "page_outcomes_not_observed",
            },
            "failed_units": {
                "value": failed_unit_count,
                "unit": "source_pages",
                "status": "measured" if failed_unit_count is not None else "unavailable",
                "reason_code": None if failed_unit_count is not None else "page_outcomes_not_observed",
            },
            "unjudged_units": {
                "value": unjudged_count,
                "unit": "source_pages",
                "status": "measured" if unjudged_count is not None else "unavailable",
                "reason_code": None if unjudged_count is not None else "input_count_not_observed",
            },
            "unmeasured_units": {
                "value": unmeasured_count,
                "unit": "source_pages",
                "status": "measured" if unmeasured_count is not None else "unavailable",
                "reason_code": None if unmeasured_count is not None else "input_count_not_observed",
            },
            "source_fallback_units": {
                "value": source_fallback_count,
                "unit": "source_pages",
                "status": "measured" if source_fallback_count is not None else "unavailable",
                "reason_code": None if source_fallback_count is not None else "fallbacks_not_observed",
            },
            "output_mapping": [
                {
                    "source_unit_index": item.get("source_unit_index", index),
                    "image_name": item.get("image_name"),
                    "output_path": item.get("output_path") or None,
                    "status": item.get("quality_status", "unjudged"),
                    "output_status": item.get(
                        "output_status",
                        "measured" if item.get("output_path") else "unmeasured",
                    ),
                    "fallback_kind": item.get("fallback_kind") or None,
                }
                for index, item in enumerate(page_outcomes)
            ],
            "output_mapping_status": "measured" if page_outcomes else "unavailable",
            "output_mapping_reason_code": None if page_outcomes else "page_outcomes_not_observed",
        },
        "pages": [dict(item) for item in page_outcomes],
    }


def _report_base_name(report: Mapping[str, Any], run_id: str | None = None) -> str:
    stamp = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(report.get("started_at_local") or ""),
    ).strip("_")[:64]
    if not stamp:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    selected_run_id = str(run_id or report.get("run_id") or "").strip()
    if selected_run_id and not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", selected_run_id):
        raise ValueError("Invalid run report identifier.")
    return f"run_{stamp}_{selected_run_id}" if selected_run_id else f"run_{stamp}"


def _atomic_write(path: str, content: str) -> None:
    directory = os.path.dirname(path) or "."
    descriptor, temporary = tempfile.mkstemp(prefix=".run-report-", dir=directory)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def start_run_report(
    *,
    log_dir: str,
    run_id: str,
    started_at_local: str,
    input_count: int,
) -> str:
    """시작 기록을 먼저 저장해 강제 종료된 실행을 성공으로 오인하지 않는다."""

    report = build_run_report(
        telemetry=None,
        total_wall_sec=None,
        page_outcomes=[],
        output_summary={
            "input_count": input_count,
            "output_count": None,
            "fallback_count": None,
            "fallbacks": None,
            "missing": None,
        },
        started_at_local=started_at_local,
        run_id=run_id,
        terminal_status="running",
        processing_status="running",
        report_status="running",
        cleanup_status="not_started",
        telemetry_reason_code="run_in_progress",
    )
    try:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"{_report_base_name(report, run_id)}.json")
        _atomic_write(path, json.dumps(report, ensure_ascii=False, indent=2))
        return path
    except Exception:
        logger.warning("Could not write the run report start record.", exc_info=True)
        return ""


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
        + (
            f"({format_minutes(float(report['total_wall_sec']))})"
            if isinstance(report.get("total_wall_sec"), (int, float))
            else ""
        )
    )
    lines.append(
        "종료: "
        f"{report.get('terminal_status', 'unknown')} "
        f"(처리 {report.get('processing_status', 'unknown')}, "
        f"보고 {report.get('report_status', 'unknown')}, "
        f"정리 {report.get('cleanup_status', 'unknown')})"
    )
    if "auto_export_status" in report:
        completed_targets = report.get("auto_export_completed_targets") or []
        selected_targets = report.get("auto_export_targets") or []
        lines.append(
            "TXT/MD 자동 내보내기: "
            f"{report.get('auto_export_status', 'unknown')} "
            f"({len(completed_targets)}/{len(selected_targets)})"
            + (
                f", 실패 {report['auto_export_failed_target']}"
                if report.get("auto_export_failed_target")
                else ""
            )
        )
    page_count = report.get("page_count")
    page_count_text = (
        f"{page_count}장"
        if isinstance(page_count, int) and not isinstance(page_count, bool)
        else "미측정"
    )
    lines.append(f"페이지: {page_count_text}")
    quality = report.get("quality") or {}
    if quality.get("status") != "completed" or quality.get("review_page_count") or quality.get("failed_page_count"):
        failed_count = quality.get("failed_page_count")
        review_count = quality.get("review_page_count")
        unjudged_count = quality.get("unjudged_units", {}).get("value")
        unmeasured_count = quality.get("unmeasured_units", {}).get("value")
        source_fallback_count = quality.get("source_fallback_units", {}).get("value")
        lines.append(
            f"품질 확인: 검토 필요 {review_count if review_count is not None else '미측정'}장, "
            f"실패 {failed_count if failed_count is not None else '미측정'}장, "
            f"미판정 {unjudged_count if unjudged_count is not None else '미측정'}장, "
            f"미측정 {unmeasured_count if unmeasured_count is not None else '미측정'}장, "
            f"원본 대체 {source_fallback_count if source_fallback_count is not None else '미측정'}장"
        )
    per_page = report.get("seconds_per_page")
    if isinstance(per_page, (int, float)):
        lines.append(f"페이지당: {float(per_page):.2f}초")

    output = report.get("output") or {}
    if isinstance(output, Mapping):
        lines.append("")
        output_count_text = (
            f"{output.get('output_count')}장"
            if isinstance(output.get("output_count"), int)
            and not isinstance(output.get("output_count"), bool)
            else "미측정"
        )
        input_count = output.get("input_count")
        input_count_text = (
            f"{input_count}장"
            if isinstance(input_count, int) and not isinstance(input_count, bool)
            else "미측정"
        )
        lines.append(
            f"출력: {output_count_text} / 입력 {input_count_text}"
        )
        fallbacks = output.get("fallbacks") or []
        if fallbacks:
            lines.append(f"폴백 저장: {len(fallbacks)}장")
            for item in fallbacks:
                if not isinstance(item, Mapping):
                    continue
                cause = str(
                    item.get("cause")
                    or f"{item.get('failed_stage', '')}: {item.get('reason', '')}"
                ).strip(": ").strip()
                lines.append(
                    f"  - {item.get('image_name', '?')}"
                    f" [{item.get('kind', '?')}]"
                    f" {cause}".rstrip()
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
    work = report.get("stage_work") or []
    if work:
        lines.append("")
        lines.append("단계별 작업량 합계 (겹쳐 돈 양까지 포함)")
        lines.append("-" * 60)
        windows = {
            str(row.get("stage")): float(row.get("seconds", 0.0) or 0.0)
            for row in (report.get("stages") or [])
            if isinstance(row, Mapping)
        }
        for row in work:
            if not isinstance(row, Mapping):
                continue
            stage = str(row.get("stage", "?"))
            work_seconds = float(row.get("work_seconds", 0.0) or 0.0)
            per = row.get("seconds_per_measurement")
            window = windows.get(stage)
            # 작업 합계가 벽시계 구간보다 크면 그 차이가 동시성으로 숨긴 양이다.
            hidden = (
                f" 숨김 {format_minutes(work_seconds - window)}"
                if window is not None and work_seconds > window
                else ""
            )
            lines.append(
                f"  {str(row.get('label', stage)):<12}"
                f" {format_minutes(work_seconds):>9}"
                f" x{int(row.get('count', 0) or 0):<5}"
                f" {f'{float(per):.2f}s/회' if isinstance(per, (int, float)) else '':>10}"
                f"{hidden}"
            )
    operations = report.get("operations") or []
    if operations:
        lines.append("")
        lines.append("세부 연산 (긴 순서, 상위 10개)")
        lines.append("-" * 60)
        for row in operations[:10]:
            if not isinstance(row, Mapping):
                continue
            each = row.get("seconds_each")
            lines.append(
                f"  {str(row.get('stage', '?')):<10}"
                f" {str(row.get('operation', '?')):<22}"
                f" {format_minutes(float(row.get('seconds', 0.0) or 0.0)):>9}"
                f" x{int(row.get('count', 0) or 0):<5}"
                f" {f'{float(each):.3f}s/회' if isinstance(each, (int, float)) else ''}"
            )
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def write_run_report(
    report: Mapping[str, Any],
    *,
    log_dir: str,
    run_id: str | None = None,
) -> str:
    """리포트를 ``log_dir`` 에 JSON 과 텍스트로 쓴다. 경로를 돌려준다.

    쓰기에 실패해도 배치를 실패시키지 않는다. 리포트는 진단 자료이지 결과물이
    아니다.
    """

    try:
        os.makedirs(log_dir, exist_ok=True)
        base_name = _report_base_name(report, run_id)
        json_path = os.path.join(log_dir, f"{base_name}.json")
        text_path = os.path.join(log_dir, f"{base_name}.txt")
        previous_files = {
            path: Path(path).read_bytes() if os.path.exists(path) else None
            for path in (text_path, json_path)
        }
        try:
            _atomic_write(text_path, render_run_report_text(report))
            _atomic_write(json_path, json.dumps(report, ensure_ascii=False, indent=2))
        except BaseException:
            for path, previous in previous_files.items():
                try:
                    if previous is None:
                        if os.path.exists(path):
                            os.unlink(path)
                    else:
                        _atomic_write(path, previous.decode("utf-8"))
                except BaseException:
                    logger.error("Could not roll back partial run report at %s.", path, exc_info=True)
            raise
        logger.info("Wrote the run report to %s", text_path)
        return text_path
    except Exception:
        logger.warning("Could not write the run report.", exc_info=True)
        return ""
