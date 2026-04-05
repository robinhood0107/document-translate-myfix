#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_summary_files(root: Path) -> list[dict]:
    payloads: list[dict] = []
    for summary_path in sorted(root.rglob("summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["_summary_path"] = str(summary_path)
        payload["_run_dir"] = str(summary_path.parent)
        payloads.append(payload)
    return payloads


def _render_markdown(payloads: list[dict]) -> str:
    lines = [
        "# Benchmark Aggregation",
        "",
        "| run_dir | mode | elapsed_sec | page_done | page_failed | gpu_floor_free_mb | gpu_peak_used_mb |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for payload in payloads:
        run_dir = str(payload.get("_run_dir", ""))
        try:
            run_dir = "./" + str(Path(run_dir).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        except Exception:
            run_dir = run_dir.replace("\\", "/")
        lines.append(
            "| {run_dir} | {mode} | {elapsed} | {done} | {failed} | {free} | {used} |".format(
                run_dir=run_dir,
                mode=payload.get("mode", ""),
                elapsed=payload.get("elapsed_sec", ""),
                done=payload.get("page_done_count", ""),
                failed=payload.get("page_failed_count", ""),
                free=payload.get("gpu_floor_free_mb", ""),
                used=payload.get("gpu_peak_used_mb", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize benchmark run directories.")
    parser.add_argument("--input", required=True, help="Benchmark root directory")
    parser.add_argument("--output", default="", help="Optional markdown output path")
    args = parser.parse_args()

    root = Path(args.input)
    payloads = _load_summary_files(root)
    markdown = _render_markdown(payloads)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
