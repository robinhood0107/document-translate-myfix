#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_common import create_run_dir, repo_relative_str, write_json
from benchmark_stage_batched_archive_pipeline import SPEED_PROFILES
from modules.utils.automatic_output import OUTPUT_ARCHIVE_FORMAT_CBZ, OUTPUT_IMAGE_FORMAT_PNG


DEFAULT_MATRIX_PROFILES = [
    "baseline-safe",
    "ctx3072-fast",
    "ctx2560-aggressive",
    "ctx2560-fast-archive",
    "ctx2048-gpu23-fast",
    "ctx1792-gpu23-extreme",
    "ctx1536-gpu23-shadow",
    "ctx1280-gpu23-shadow",
    "ctx3072-fast-archive",
    "warm-reuse",
    "ctx3072-gpu24-extreme",
    "ctx2560-gpu24-extreme",
    "ctx2560-gpu25-danger",
    "ctx2048-gpu24-extreme",
    "ctx1792-gpu24-shadow",
    "ctx2048-gpu25-danger",
    "ctx2048-gpu26-danger",
    "ctx2048-threads14",
    "ctx2048-batch1024",
    "ctx2048-batch2048",
    "ctx2048-flash-attn",
    "ctx2048-q8-kv",
    "ctx2048-no-warmup",
    "ctx2048-np2",
    "ctx3072-threads12",
    "danger-shadow",
]
ARCHIVE_RUNNER = SCRIPT_DIR / "benchmark_stage_batched_archive_pipeline.py"
SAFE_SUMMARY_KEYS = (
    "mode",
    "workflow_mode",
    "resident_ocr_mode",
    "page_done_count",
    "page_failed_count",
    "final_archive_page_count",
    "archive_format",
    "archive_image_format",
    "archive_compression_level",
    "runtime_reuse_mode",
    "gemma_runtime_overrides",
    "tiny_font_item_count",
    "min_render_font_size",
    "total_elapsed_sec",
    "elapsed_sec",
    "event_counts",
    "llama_cpp_runtime",
)
RAW_PATH_KEYS = {
    "image_paths",
    "final_archive_path",
    "final_archive_root",
    "render_fit_summary_path",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()


def anonymize_input_paths(input_paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path_hash": _sha256_text(str(path.resolve())),
            "suffix": path.suffix.lower(),
            "exists": path.exists(),
        }
        for path in input_paths
    ]


def safe_summary(summary: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in SAFE_SUMMARY_KEYS:
        if key in summary and key not in RAW_PATH_KEYS:
            cleaned[key] = summary[key]
    return cleaned


def load_safe_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return safe_summary(payload)


def profile_status(*, returncode: int, summary: dict[str, Any], allow_failure: bool) -> str:
    page_failed_count = int(summary.get("page_failed_count", 0) or 0) if isinstance(summary, dict) else 0
    hard_failed = int(returncode) != 0 or page_failed_count > 0
    if hard_failed:
        return "shadow_failed" if allow_failure else "failed"
    return "passed"


def build_profile_command(
    *,
    profile: str,
    input_paths: list[Path],
    run_dir: Path,
    source_lang: str,
    target_lang: str,
    ocr_mode: str,
    archive_format: str,
    archive_image_format: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ARCHIVE_RUNNER),
        "--speed-profile",
        profile,
        "--ocr-mode",
        ocr_mode,
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--archive-format",
        archive_format,
        "--archive-image-format",
        archive_image_format,
        "--output-dir",
        str(run_dir),
        "--input",
        *[str(path) for path in input_paths],
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def run_profile(command: list[str], *, cwd: Path, stdout_path: Path, stderr_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd), text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    return int(completed.returncode), elapsed


def render_matrix_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage-Batched Gemma Speed Matrix",
        "",
        "| profile | status | elapsed_sec | page_failed | archive_pages | ctx | gpu_layers | threads | n_parallel | compression | reuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in results:
        summary = item.get("summary", {}) if isinstance(item.get("summary"), dict) else {}
        overrides = summary.get("gemma_runtime_overrides", {}) if isinstance(summary.get("gemma_runtime_overrides"), dict) else {}
        lines.append(
            "| {profile} | {status} | {elapsed:.3f} | {failed} | {pages} | {ctx} | {gpu_layers} | {threads} | {n_parallel} | {compression} | {reuse} |".format(
                profile=item.get("profile", ""),
                status=item.get("status", ""),
                elapsed=float(item.get("elapsed_sec", 0.0) or 0.0),
                failed=summary.get("page_failed_count", ""),
                pages=summary.get("final_archive_page_count", ""),
                ctx=overrides.get("context_size", ""),
                gpu_layers=overrides.get("n_gpu_layers", ""),
                threads=overrides.get("threads", ""),
                n_parallel=overrides.get("n_parallel", ""),
                compression=summary.get("archive_compression_level", ""),
                reuse=summary.get("runtime_reuse_mode", ""),
            )
        )
    lines.append("")
    lines.append("Raw OCR text, translations, local source paths, images, and archives are intentionally omitted.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage-Batched archive benchmarks across Gemma speed profiles.")
    parser.add_argument("--input", required=True, nargs="+", help="Archive or image input path(s)")
    parser.add_argument("--profiles", nargs="+", choices=tuple(SPEED_PROFILES.keys()), default=DEFAULT_MATRIX_PROFILES)
    parser.add_argument("--source-lang", default="English")
    parser.add_argument("--target-lang", default="Korean")
    parser.add_argument("--ocr-mode", default="optimal-plus")
    parser.add_argument("--archive-format", default=OUTPUT_ARCHIVE_FORMAT_CBZ)
    parser.add_argument("--archive-image-format", default=OUTPUT_IMAGE_FORMAT_PNG)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--write-next-to-source", action="store_true")
    parser.add_argument("--discard-staging", action="store_true")
    args = parser.parse_args()

    input_paths = [Path(value).expanduser().resolve() for value in args.input]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input path(s) not found: {missing}")

    suite_dir = Path(args.output_dir).resolve() if args.output_dir else create_run_dir(
        "stage_batched_speed_matrix",
        root=Path(args.output_root).resolve() if args.output_root else None,
    )
    suite_dir.mkdir(parents=True, exist_ok=True)

    extra_args: list[str] = []
    if args.write_next_to_source:
        extra_args.append("--write-next-to-source")
    if args.discard_staging:
        extra_args.append("--discard-staging")

    results: list[dict[str, Any]] = []
    for profile in args.profiles:
        run_dir = suite_dir / profile
        run_dir.mkdir(parents=True, exist_ok=True)
        command = build_profile_command(
            profile=profile,
            input_paths=input_paths,
            run_dir=run_dir,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            ocr_mode=args.ocr_mode,
            archive_format=args.archive_format,
            archive_image_format=args.archive_image_format,
            extra_args=extra_args,
        )
        stdout_path = run_dir / "stdout.txt"
        stderr_path = run_dir / "stderr.txt"
        returncode, elapsed_sec = run_profile(command, cwd=ROOT, stdout_path=stdout_path, stderr_path=stderr_path)
        allow_failure = bool(SPEED_PROFILES.get(profile, {}).get("allow_failure", False))
        summary = load_safe_summary(run_dir / "summary.json")
        status = profile_status(returncode=returncode, summary=summary, allow_failure=allow_failure)
        results.append(
            {
                "profile": profile,
                "status": status,
                "returncode": returncode,
                "elapsed_sec": round(elapsed_sec, 3),
                "stdout_path": repo_relative_str(stdout_path),
                "stderr_path": repo_relative_str(stderr_path),
                "summary": summary,
            }
        )
        write_json(
            suite_dir / "matrix_summary.json",
            {
                "input_count": len(input_paths),
                "inputs": anonymize_input_paths(input_paths),
                "profiles": list(args.profiles),
                "results": results,
            },
        )
        (suite_dir / "matrix_report.md").write_text(render_matrix_report(results), encoding="utf-8")
        if status == "failed":
            break

    final_payload = {
        "input_count": len(input_paths),
        "inputs": anonymize_input_paths(input_paths),
        "profiles": list(args.profiles),
        "results": results,
    }
    write_json(suite_dir / "matrix_summary.json", final_payload)
    (suite_dir / "matrix_report.md").write_text(render_matrix_report(results), encoding="utf-8")
    return 1 if any(item["status"] == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
