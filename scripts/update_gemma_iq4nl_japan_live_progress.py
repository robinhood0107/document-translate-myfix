from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_SUITE_DIR = Path(
    "/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/"
    "banchmark_result_log/gemma_iq4nl_japan/"
    "20260411_171639_gemma_iq4nl_japan_fullgpu_suite"
)
DEFAULT_OUTPUT = Path(
    "/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate/"
    "docs/benchmark/gemma-iq4nl-japan/live-progress-ko.md"
)
REPO_ROOT = Path(
    "/mnt/c/Users/pjjpj/Desktop/openai_manga_translater/comic-translate"
)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return "./" + str(path.relative_to(repo_root)).replace("\\", "/")
    except ValueError:
        return str(path)


def summarize_attempt(attempt_dir: Path) -> dict[str, Any]:
    summary = load_json(attempt_dir / "summary.json")
    if summary is not None:
        return {
            "attempt": attempt_dir.name,
            "kind": "summary",
            "elapsed_sec": summary.get("elapsed_sec"),
            "page_done_count": summary.get("page_done_count"),
            "page_failed_count": summary.get("page_failed_count"),
            "gpu_peak_used_mb": summary.get("gpu_peak_used_mb"),
            "gpu_floor_free_mb": summary.get("gpu_floor_free_mb"),
            "detect_median_sec": summary.get("detect_median_sec"),
            "ocr_median_sec": summary.get("ocr_median_sec"),
            "translate_median_sec": summary.get("translate_median_sec"),
            "inpaint_median_sec": summary.get("inpaint_median_sec"),
            "gemma_json_retry_count": summary.get("gemma_json_retry_count"),
            "gemma_truncated_count": summary.get("gemma_truncated_count"),
            "gemma_empty_content_count": summary.get("gemma_empty_content_count"),
            "gemma_missing_key_count": summary.get("gemma_missing_key_count"),
            "gemma_schema_validation_fail_count": summary.get(
                "gemma_schema_validation_fail_count"
            ),
        }

    metrics_path = attempt_dir / "metrics.jsonl"
    if metrics_path.exists():
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        page_done = [row for row in rows if row.get("tag") == "page_done"]
        last_row = rows[-1] if rows else {}
        return {
            "attempt": attempt_dir.name,
            "kind": "metrics",
            "metrics_rows": len(rows),
            "page_done_count": len(page_done),
            "last_tag": last_row.get("tag"),
            "last_image": last_row.get("image_name"),
        }

    return {"attempt": attempt_dir.name, "kind": "empty"}


def collect_completed_stage1(stage_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not stage_dir.exists():
        return rows

    for candidate_dir in sorted(p for p in stage_dir.iterdir() if p.is_dir()):
        attempts = sorted(p for p in candidate_dir.iterdir() if p.is_dir())
        for attempt_dir in attempts:
            summary = load_json(attempt_dir / "summary.json")
            if summary is None:
                continue
            rows.append(
                {
                    "candidate": candidate_dir.name,
                    "attempt": attempt_dir.name,
                    "elapsed_sec": summary.get("elapsed_sec"),
                    "page_done_count": summary.get("page_done_count"),
                    "page_failed_count": summary.get("page_failed_count"),
                    "gpu_floor_free_mb": summary.get("gpu_floor_free_mb"),
                    "translate_median_sec": summary.get("translate_median_sec"),
                    "gemma_truncated_count": summary.get("gemma_truncated_count"),
                    "gemma_empty_content_count": summary.get(
                        "gemma_empty_content_count"
                    ),
                    "gemma_missing_key_count": summary.get(
                        "gemma_missing_key_count"
                    ),
                    "gemma_schema_validation_fail_count": summary.get(
                        "gemma_schema_validation_fail_count"
                    ),
                }
            )

    rows.sort(
        key=lambda item: (
            item["page_failed_count"],
            item["elapsed_sec"] if item["elapsed_sec"] is not None else 10**18,
        )
    )
    return rows


def format_completed_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_아직 완료된 stage1 후보가 없습니다._"

    header = [
        "| candidate | attempt | elapsed_sec | page_failed | translate_median_sec | gpu_floor_free_mb | note |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    body = []
    for row in rows:
        note_parts = []
        if row["page_failed_count"]:
            note_parts.append("failed")
        if row["gemma_truncated_count"]:
            note_parts.append("truncated")
        if row["gemma_empty_content_count"]:
            note_parts.append("empty")
        if row["gemma_missing_key_count"]:
            note_parts.append("missing-key")
        if row["gemma_schema_validation_fail_count"]:
            note_parts.append("schema-fail")
        note = ", ".join(note_parts) if note_parts else "passed"
        body.append(
            "| {candidate} | {attempt} | {elapsed_sec:.3f} | {page_failed_count} | "
            "{translate_median_sec:.3f} | {gpu_floor_free_mb} | {note} |".format(
                candidate=row["candidate"],
                attempt=row["attempt"],
                elapsed_sec=row["elapsed_sec"],
                page_failed_count=row["page_failed_count"],
                translate_median_sec=row["translate_median_sec"],
                gpu_floor_free_mb=row["gpu_floor_free_mb"],
                note=note,
            )
        )
    return "\n".join(header + body)


def build_markdown(
    suite_dir: Path,
    state: dict[str, Any],
    active_attempt_dir: Path | None,
    active_attempt_summary: dict[str, Any] | None,
    completed_stage1: list[dict[str, Any]],
) -> str:
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines: list[str] = []
    lines.append("# Gemma IQ4_NL Japan Full-GPU Live Progress")
    lines.append("")
    lines.append("## 현재 상태")
    lines.append("")
    lines.append(f"- 업데이트 시각: `{now}`")
    lines.append(f"- 공식 suite: `{relative_to_repo(suite_dir, REPO_ROOT)}`")
    lines.append(f"- suite 상태: `{state.get('status', '')}`")
    lines.append(f"- 현재 stage: `{state.get('current_stage', '')}`")
    lines.append(f"- 현재 candidate: `{state.get('current_candidate', '')}`")
    lines.append(f"- 마지막 heartbeat: `{state.get('last_heartbeat_at', '')}`")
    lines.append(f"- infra retry count: `{state.get('infra_retry_count', 0)}`")
    if state.get("last_failure_kind") or state.get("last_failure_reason"):
        lines.append(f"- last failure kind: `{state.get('last_failure_kind', '')}`")
        lines.append(
            f"- last failure reason: `{state.get('last_failure_reason', '')}`"
        )
    lines.append("")
    lines.append("## 고정 파이프라인")
    lines.append("")
    lines.append("- translator: `Custom Local Server(Gemma)`")
    lines.append("- ocr: `PaddleOCR VL`")
    lines.append("- detector: `RT-DETR-v2 ONNX + CUDAExecutionProvider`")
    lines.append("- inpainter: `lama_large_512px`")
    lines.append("- mask refiner: `ctd`")
    lines.append("- use_gpu: `true`")
    lines.append("- OCR / detector / CTD / inpainter: 모두 `cuda`")
    lines.append("- corpus: `Sample/japan` 전체 22장")
    lines.append("")
    lines.append("## 진행 중인 candidate")
    lines.append("")
    if active_attempt_dir is None or active_attempt_summary is None:
        lines.append("_현재 활성 attempt 정보를 찾지 못했습니다._")
    else:
        lines.append(
            f"- active attempt: `{relative_to_repo(active_attempt_dir, REPO_ROOT)}`"
        )
        if active_attempt_summary["kind"] == "summary":
            lines.append(
                f"- 상태: 완료 (`page_done={active_attempt_summary['page_done_count']}`, "
                f"`page_failed={active_attempt_summary['page_failed_count']}`)"
            )
            lines.append(
                f"- elapsed_sec: `{active_attempt_summary['elapsed_sec']}`"
            )
        elif active_attempt_summary["kind"] == "metrics":
            lines.append(
                f"- 상태: 실행 중 (`page_done={active_attempt_summary['page_done_count']}`)"
            )
            lines.append(
                f"- 마지막 태그: `{active_attempt_summary.get('last_tag', '')}`"
            )
            lines.append(
                f"- 마지막 이미지: `{active_attempt_summary.get('last_image', '')}`"
            )
            lines.append(
                f"- metrics rows: `{active_attempt_summary.get('metrics_rows', 0)}`"
            )
        else:
            lines.append("- 상태: 초기화 중")
    lines.append("")
    lines.append("## stage1 완료 후보 요약")
    lines.append("")
    lines.append(format_completed_table(completed_stage1))
    lines.append("")
    lines.append("## 현재까지의 해석")
    lines.append("")
    if completed_stage1:
        best = completed_stage1[0]
        lines.append(
            f"- 현재까지 가장 빠른 통과 후보는 `{best['candidate']} / {best['attempt']}`이며 "
            f"`elapsed_sec={best['elapsed_sec']:.3f}`, "
            f"`translate_median_sec={best['translate_median_sec']:.3f}` 입니다."
        )
        if best["candidate"] == "ov068-ngl16":
            lines.append(
                "- `ov068-ngl16`은 속도는 가장 좋지만 `gpu_floor_free_mb=79`라서 "
                "VRAM headroom이 매우 낮습니다."
            )
        rescue_rows = [
            row for row in completed_stage1 if row["attempt"] != "attempt01_t07_infra01"
        ]
        if rescue_rows:
            rescue = rescue_rows[0]
            lines.append(
                f"- 현재까지 rescue가 필요했던 후보는 `{rescue['candidate']}`이며, "
                f"`{rescue['attempt']}`에서 통과했습니다."
            )
    else:
        lines.append("- 아직 stage1 완료 후보가 없어 해석을 보류합니다.")
    lines.append("")
    lines.append("## 메모")
    lines.append("")
    lines.append(
        "- 이 문서는 벤치가 도는 동안의 live progress 문서입니다. "
        "최종 채택값은 suite 완료 후 최종 보고서에 다시 정리합니다."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    state = load_json(args.suite_dir / "suite_state.json")
    if state is None:
        raise FileNotFoundError(f"missing suite_state.json under {args.suite_dir}")

    active_attempt_dir: Path | None = None
    active_attempt_summary: dict[str, Any] | None = None
    stage = state.get("current_stage")
    candidate = state.get("current_candidate")
    if stage and candidate:
        candidate_dir = args.suite_dir / stage / candidate
        attempts = (
            sorted(p for p in candidate_dir.iterdir() if p.is_dir())
            if candidate_dir.exists()
            else []
        )
        if attempts:
            active_attempt_dir = attempts[-1]
            active_attempt_summary = summarize_attempt(active_attempt_dir)

    completed_stage1 = collect_completed_stage1(args.suite_dir / "stage1")
    markdown = build_markdown(
        suite_dir=args.suite_dir,
        state=state,
        active_attempt_dir=active_attempt_dir,
        active_attempt_summary=active_attempt_summary,
        completed_stage1=completed_stage1,
    )
    args.output.write_text(markdown, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
