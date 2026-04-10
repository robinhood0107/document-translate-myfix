#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import (
    ALL_BENCHMARK_CONTAINER_NAMES,
    create_run_dir,
    load_preset,
    remove_containers,
    repo_relative_str,
    resolve_runtime_health_urls,
    run_command,
    stage_runtime_files,
    write_json,
)

FAMILY_NAME = "inpaint_ctd"
FAMILY_OUTPUT_ROOT_NAME = "inpaint_ctd"
LAST_SUITE_RECORD = "last_inpaint_ctd_suite.json"
REPORT_MANIFEST_NAME = "inpaint_ctd_report_manifest.yaml"
DEFAULT_SCOPE = "suite"

CASES = [
    {"slug": "legacy-bbox-aot", "label": "legacy_bbox + AOT", "protect": False},
    {"slug": "ctd-aot", "label": "ctd + AOT", "protect": False},
    {"slug": "ctd-protect-aot", "label": "ctd + protect + AOT", "protect": True},
    {"slug": "ctd-protect-lama-large-512px", "label": "ctd + protect + lama_large_512px", "protect": True},
    {"slug": "ctd-protect-lama-mpe", "label": "ctd + protect + lama_mpe", "protect": True},
]
CASE_BY_SLUG = {case["slug"]: case for case in CASES}
CORPORA = {
    "china": {
        "sample_subdir": "China",
        "sample_count": 8,
        "source_lang": "Chinese",
        "target_lang": "Korean",
        "ocr": "HunyuanOCR + Gemma",
        "spotlight_file": "0094_0093.jpg",
    },
    "japan": {
        "sample_subdir": "japan",
        "sample_count": 22,
        "source_lang": "Japanese",
        "target_lang": "Korean",
        "ocr": "PaddleOCR VL + Gemma",
        "spotlight_file": "094.png",
    },
}
DOCKER_LOG_CONTAINERS = [
    "gemma-local-server",
    "paddleocr-server",
    "paddleocr-vllm",
    "hunyuanocr-local-server",
]


def _log(message: str) -> None:
    print(f"[inpaint-ctd] {message}", flush=True)


def family_output_root() -> Path:
    env_root = os.getenv("CT_BENCH_OUTPUT_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
        if root.name != FAMILY_OUTPUT_ROOT_NAME:
            root = root / FAMILY_OUTPUT_ROOT_NAME
    else:
        root = ROOT / "banchmark_result_log" / FAMILY_OUTPUT_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _repo_root_results_root() -> str:
    try:
        return "./" + str(family_output_root().resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(family_output_root()).replace("\\", "/")


def _current_git_sha(ref: str = "HEAD") -> str:
    completed = run_command(["git", "rev-parse", ref], cwd=ROOT, check=False)
    return (completed.stdout or "").strip()


def _wait_for_url(url: str, timeout_sec: int = 240) -> None:
    import urllib.request

    started = time.time()
    while time.time() - started < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _stage_spotlight_input(suite_dir: Path, corpus_name: str, source_path: Path) -> Path:
    target_dir = suite_dir / "_spotlight_inputs" / corpus_name
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_dir / source_path.name)
    return target_dir


def _compose_up(runtime_dir: Path, preset_ref: str) -> None:
    preset, _ = load_preset(preset_ref)
    staged = stage_runtime_files(preset, runtime_dir)
    remove_containers(ALL_BENCHMARK_CONTAINER_NAMES)
    if staged.get("gemma"):
        run_command(
            ["docker", "compose", "-f", staged["gemma"]["compose_path"], "up", "-d", "--force-recreate"],
            cwd=runtime_dir / "gemma",
        )
    if staged["ocr"].get("kind") != "internal":
        run_command(
            ["docker", "compose", "-f", staged["ocr"]["compose_path"], "up", "-d", "--force-recreate"],
            cwd=runtime_dir / "ocr",
        )
    for url in resolve_runtime_health_urls(preset, "full"):
        _wait_for_url(url)


def _write_container_logs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in DOCKER_LOG_CONTAINERS:
        completed = run_command(["docker", "logs", "--tail", "400", name], check=False)
        (output_dir / f"{name}.log").write_text((completed.stdout or "") + (completed.stderr or ""), encoding="utf-8")


def _run_pipeline_step(
    *,
    preset: str,
    mode: str,
    sample_dir: Path,
    sample_count: int,
    output_dir: Path,
    source_lang: str,
    target_lang: str,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        preset,
        "--mode",
        mode,
        "--repeat",
        "1",
        "--runtime-mode",
        "attach-running",
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(sample_count),
        "--output-dir",
        str(output_dir),
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--export-page-snapshots",
        "--clear-app-caches",
    ]
    env = os.environ.copy()
    env["CT_BENCH_OUTPUT_ROOT"] = str(family_output_root())
    _log("run: " + " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "command_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (output_dir / "command_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"benchmark_pipeline failed: preset={preset} mode={mode} code={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _find_export_root(run_dir: Path) -> Path | None:
    corpus_dir = run_dir / "corpus"
    candidates = sorted(path for path in corpus_dir.glob("comic_translate_*") if path.is_dir())
    return candidates[-1] if candidates else None


def _find_artifact(export_root: Path | None, folder: str, stem: str, suffix: str) -> str:
    if export_root is None:
        return ""
    base_dir = export_root / folder
    if not base_dir.exists():
        return ""
    matches = sorted(base_dir.rglob(f"{stem}_{suffix}.*"))
    return repo_relative_str(matches[0]) if matches else ""


def _load_debug_cleanup_count(export_root: Path | None) -> int:
    if export_root is None:
        return 0
    debug_dir = export_root / "debug_metadata"
    if not debug_dir.exists():
        return 0
    count = 0
    for path in debug_dir.rglob("*_debug.json"):
        try:
            payload = _load_json(path)
        except Exception:
            continue
        if bool(payload.get("cleanup_applied", False)):
            count += 1
    return count


def _load_page_snapshots(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "page_snapshots.json"
    return _load_json(path) if path.is_file() else {"pages": []}


def _snapshot_signature(snapshot: dict[str, Any]) -> dict[str, Any]:
    pages = snapshot.get("pages", []) if isinstance(snapshot.get("pages"), list) else []
    signature: dict[str, Any] = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        image_name = str(page.get("image_name", ""))
        blocks = page.get("blocks", []) if isinstance(page.get("blocks"), list) else []
        texts = [str(block.get("normalized_text", "")) for block in blocks if isinstance(block, dict)]
        signature[image_name] = {
            "page_failed": bool(page.get("page_failed", False)),
            "block_count": int((page.get("ocr_quality") or {}).get("block_count", 0) if isinstance(page.get("ocr_quality"), dict) else 0),
            "texts": texts,
        }
    return signature


def _build_run_record(*, corpus_name: str, scope: str, case_slug: str, run_dir: Path, spotlight_file: str | None = None) -> dict[str, Any]:
    summary = _load_json(run_dir / "summary.json")
    export_root = _find_export_root(run_dir)
    page_snapshots = _load_page_snapshots(run_dir)
    record: dict[str, Any] = {
        "corpus": corpus_name,
        "scope": scope,
        "case_slug": case_slug,
        "case_label": CASE_BY_SLUG[case_slug]["label"],
        "run_dir": repo_relative_str(run_dir),
        "export_root": repo_relative_str(export_root) if export_root else "",
        "summary": summary,
        "page_snapshots": page_snapshots,
        "cleanup_applied_count": _load_debug_cleanup_count(export_root),
        "artifacts": {},
    }
    if spotlight_file:
        stem = Path(spotlight_file).stem
        record["artifacts"] = {
            "source": repo_relative_str(run_dir / "corpus" / spotlight_file),
            "detector_overlay": _find_artifact(export_root, "detector_overlays", stem, "detector_overlay"),
            "raw_mask": _find_artifact(export_root, "raw_masks", stem, "raw_mask"),
            "mask_overlay": _find_artifact(export_root, "mask_overlays", stem, "mask_overlay"),
            "cleanup_delta": _find_artifact(export_root, "cleanup_mask_delta", stem, "cleanup_delta"),
            "cleaned": _find_artifact(export_root, "cleaned_images", stem, "cleaned"),
            "translated": _find_artifact(export_root, "translated_images", stem, "translated"),
            "debug_metadata": _find_artifact(export_root, "debug_metadata", stem, "debug"),
        }
    return record


def _ocr_invariance(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"status": "SKIP", "mismatches": []}
    baseline = records[0]
    baseline_sig = _snapshot_signature(baseline.get("page_snapshots", {}))
    mismatches: list[dict[str, Any]] = []
    for record in records[1:]:
        current_sig = _snapshot_signature(record.get("page_snapshots", {}))
        all_pages = sorted(set(baseline_sig) | set(current_sig))
        for page_name in all_pages:
            if baseline_sig.get(page_name) != current_sig.get(page_name):
                mismatches.append(
                    {
                        "case": record["case_slug"],
                        "page": page_name,
                        "baseline": baseline_sig.get(page_name),
                        "current": current_sig.get(page_name),
                    }
                )
                if len(mismatches) >= 20:
                    break
        if len(mismatches) >= 20:
            break
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "baseline_case": baseline["case_slug"],
        "mismatches": mismatches,
    }


def _corpus_case_preset(corpus_name: str, case_slug: str) -> str:
    return f"inpaint-ctd-{corpus_name}-{case_slug}"


def _run_scope_for_corpus(*, suite_dir: Path, sample_root: Path, corpus_name: str, scope: str, case_slugs: list[str]) -> list[dict[str, Any]]:
    corpus_cfg = CORPORA[corpus_name]
    records: list[dict[str, Any]] = []
    if scope == "spotlight":
        spotlight_name = str(corpus_cfg["spotlight_file"])
        spotlight_source = sample_root / str(corpus_cfg["sample_subdir"]) / spotlight_name
        if not spotlight_source.is_file():
            raise FileNotFoundError(f"Spotlight image not found: {spotlight_source}")
        sample_dir = _stage_spotlight_input(suite_dir, corpus_name, spotlight_source)
        sample_count = 1
        mode = "one-page"
    else:
        sample_dir = sample_root / str(corpus_cfg["sample_subdir"])
        sample_count = int(corpus_cfg["sample_count"])
        spotlight_name = None
        mode = "batch"

    for case_slug in case_slugs:
        output_dir = suite_dir / corpus_name / scope / case_slug
        output_dir.mkdir(parents=True, exist_ok=True)
        _run_pipeline_step(
            preset=_corpus_case_preset(corpus_name, case_slug),
            mode=mode,
            sample_dir=sample_dir,
            sample_count=sample_count,
            output_dir=output_dir,
            source_lang=str(corpus_cfg["source_lang"]),
            target_lang=str(corpus_cfg["target_lang"]),
        )
        records.append(
            _build_run_record(
                corpus_name=corpus_name,
                scope=scope,
                case_slug=case_slug,
                run_dir=output_dir,
                spotlight_file=spotlight_name,
            )
        )
    return records


def _suite_payload(
    *,
    suite_dir: Path,
    sample_root: Path,
    spotlight: bool,
    full: bool,
    case_slugs: list[str],
    corpus_names: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "family_name": FAMILY_NAME,
        "results_root": _repo_root_results_root(),
        "suite_dir": repo_relative_str(suite_dir),
        "git_sha": _current_git_sha(),
        "execution_scope": "full-pipeline",
        "speed_score_scope": "full-pipeline elapsed",
        "quality_gate_scope": "OCR invariance + manual inpaint review",
        "fixed_runtime": {
            "detector": "RT-DETR-v2 (CPU pinned in CUDA13 benchmark family)",
            "china": "HunyuanOCR + Gemma",
            "japan": "PaddleOCR VL + Gemma",
            "default_recommendation": "ctd + protect + AOT",
            "quality_mode": "ctd + protect + lama_large_512px",
            "offline_review_mode": "ctd + protect + lama_mpe",
        },
        "corpora": {},
    }
    for corpus_name in corpus_names:
        corpus_cfg = CORPORA[corpus_name]
        runtime_dir = suite_dir / "_runtime" / corpus_name
        _log(f"runtime start: corpus={corpus_name}")
        _compose_up(runtime_dir, _corpus_case_preset(corpus_name, case_slugs[0]))
        corpus_payload: dict[str, Any] = {
            "corpus": corpus_name,
            "sample_subdir": corpus_cfg["sample_subdir"],
            "sample_count": corpus_cfg["sample_count"],
            "spotlight_file": corpus_cfg["spotlight_file"],
            "ocr_runtime": corpus_cfg["ocr"],
            "spotlight_runs": [],
            "full_runs": [],
        }
        try:
            if spotlight:
                corpus_payload["spotlight_runs"] = _run_scope_for_corpus(
                    suite_dir=suite_dir,
                    sample_root=sample_root,
                    corpus_name=corpus_name,
                    scope="spotlight",
                    case_slugs=case_slugs,
                )
                corpus_payload["spotlight_ocr_invariance"] = _ocr_invariance(corpus_payload["spotlight_runs"])
            if full:
                corpus_payload["full_runs"] = _run_scope_for_corpus(
                    suite_dir=suite_dir,
                    sample_root=sample_root,
                    corpus_name=corpus_name,
                    scope="full",
                    case_slugs=case_slugs,
                )
                corpus_payload["full_ocr_invariance"] = _ocr_invariance(corpus_payload["full_runs"])
        finally:
            _write_container_logs(suite_dir / "docker_logs" / corpus_name)
            remove_containers(ALL_BENCHMARK_CONTAINER_NAMES)
        payload["corpora"][corpus_name] = corpus_payload
    return payload


def _write_manifest(suite_dir: Path, payload: dict[str, Any]) -> Path:
    path = suite_dir / REPORT_MANIFEST_NAME
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def run_suite(*, sample_root: Path = ROOT / "Sample") -> int:
    suite_dir = create_run_dir("inpaint_ctd_suite", root=family_output_root())
    payload = _suite_payload(
        suite_dir=suite_dir,
        sample_root=sample_root,
        spotlight=True,
        full=True,
        case_slugs=[case["slug"] for case in CASES],
        corpus_names=list(CORPORA.keys()),
    )
    manifest_path = _write_manifest(suite_dir, payload)
    write_json(suite_dir / "suite_payload.json", payload)
    write_json(family_output_root() / LAST_SUITE_RECORD, payload)
    completed = subprocess.run(
        [sys.executable, "-u", str(ROOT / "scripts" / "generate_inpaint_ctd_report.py"), "--manifest", str(manifest_path)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    (suite_dir / "report_stdout.txt").write_text(completed.stdout or "", encoding="utf-8")
    (suite_dir / "report_stderr.txt").write_text(completed.stderr or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"report generation failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    print(suite_dir)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inpaint CTD benchmark family.")
    parser.add_argument("--scope", default=DEFAULT_SCOPE, choices=("suite", "spotlight", "full"))
    parser.add_argument("--corpus", default="all", choices=("all", "china", "japan"))
    parser.add_argument("--case", default="all", choices=("all", *CASE_BY_SLUG.keys()))
    parser.add_argument("--sample-root", default=str(ROOT / "Sample"))
    args = parser.parse_args()

    sample_root = Path(args.sample_root)
    case_slugs = [args.case] if args.case != "all" else [case["slug"] for case in CASES]
    corpus_names = [args.corpus] if args.corpus != "all" else list(CORPORA.keys())

    if args.scope == "suite":
        return run_suite(sample_root=sample_root)

    suite_dir = create_run_dir(f"inpaint_ctd_{args.scope}", root=family_output_root())
    payload = _suite_payload(
        suite_dir=suite_dir,
        sample_root=sample_root,
        spotlight=args.scope == "spotlight",
        full=args.scope == "full",
        case_slugs=case_slugs,
        corpus_names=corpus_names,
    )
    manifest_path = _write_manifest(suite_dir, payload)
    write_json(suite_dir / "suite_payload.json", payload)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
