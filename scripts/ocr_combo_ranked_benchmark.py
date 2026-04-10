#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_common import (
    create_run_dir,
    load_preset,
    repo_relative_str,
    resolve_runtime_container_names,
    run_command,
    write_json,
)

from modules.utils.llama_cpp_runtime import DEFAULT_LLAMA_CPP_IMAGE

FAMILY_NAME = "ocr_combo_ranked"
FAMILY_OUTPUT_ROOT_NAME = "ocr_combo_ranked"
LAST_SUITE_RECORD = "last_ocr_combo_ranked_suite.json"
REPORT_MANIFEST_NAME = "ocr_combo_ranked_report_manifest.yaml"
STRICT_GOLD_ROOT = ROOT / "benchmarks" / "ocr_combo" / "gold"
FROZEN_CHINA_MANIFEST = ROOT / "benchmarks" / FAMILY_NAME / "frozen" / "china_winner.json"
QUALITY_BAND_RANK = {
    "catastrophic": 0,
    "hold": 1,
    "conditional": 2,
    "ready": 3,
}

JAPAN_CORPUS = {
    "name": "japan",
    "sample_subdir": "japan",
    "sample_count": 22,
    "source_lang": "Japanese",
    "target_lang": "Korean",
    "default_candidates": [
        {
            "engine": "MangaOCR + Gemma",
            "preset": "ocr-combo-ranked-japan-mangaocr-gemma",
            "tunable": True,
            "tuning_steps": [
                {
                    "name": "expansion_percentage",
                    "candidates": [
                        {"suffix": "exp3", "updates": {"ocr_generic": {"manga_expansion_percentage": 3}}},
                        {"suffix": "exp5", "updates": {"ocr_generic": {"manga_expansion_percentage": 5}}},
                        {"suffix": "exp7", "updates": {"ocr_generic": {"manga_expansion_percentage": 7}}},
                    ],
                }
            ],
        },
        {
            "engine": "PaddleOCR VL + Gemma",
            "preset": "ocr-combo-ranked-japan-paddleocr-vl-gemma",
            "tunable": True,
            "tuning_steps": [
                {
                    "name": "parallel_workers",
                    "candidates": [
                        {"suffix": "pw4", "updates": {"ocr_client": {"parallel_workers": 4}}},
                        {"suffix": "pw8", "updates": {"ocr_client": {"parallel_workers": 8}}},
                    ],
                },
                {
                    "name": "max_new_tokens",
                    "candidates": [
                        {"suffix": "mnt512", "updates": {"ocr_client": {"max_new_tokens": 512}}},
                        {"suffix": "mnt1024", "updates": {"ocr_client": {"max_new_tokens": 1024}}},
                    ],
                },
                {
                    "name": "max_concurrency",
                    "candidates": [
                        {"suffix": "conc16", "updates": {"ocr_runtime": {"max_concurrency": 16}}},
                        {"suffix": "conc32", "updates": {"ocr_runtime": {"max_concurrency": 32}}},
                    ],
                },
                {
                    "name": "gpu_memory_utilization",
                    "candidates": [
                        {"suffix": "vram080", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.80}}},
                        {"suffix": "vram084", "updates": {"ocr_runtime": {"gpu_memory_utilization": 0.84}}},
                    ],
                },
                {
                    "name": "crop_padding_ratio",
                    "candidates": [
                        {"suffix": "crop002", "updates": {"ocr_generic": {"crop_padding_ratio": 0.02}}},
                        {"suffix": "crop005", "updates": {"ocr_generic": {"crop_padding_ratio": 0.05}}},
                    ],
                },
            ],
        },
        {
            "engine": "HunyuanOCR + Gemma",
            "preset": "ocr-combo-ranked-japan-hunyuanocr-gemma",
            "tunable": True,
            "tuning_steps": [
                {
                    "name": "parallel_workers",
                    "candidates": [
                        {"suffix": "pw1", "updates": {"hunyuan_ocr_client": {"parallel_workers": 1}}},
                        {"suffix": "pw2", "updates": {"hunyuan_ocr_client": {"parallel_workers": 2}}},
                        {"suffix": "pw4", "updates": {"hunyuan_ocr_client": {"parallel_workers": 4}}},
                    ],
                },
                {
                    "name": "max_completion_tokens",
                    "candidates": [
                        {"suffix": "mct128", "updates": {"hunyuan_ocr_client": {"max_completion_tokens": 128}}},
                        {"suffix": "mct256", "updates": {"hunyuan_ocr_client": {"max_completion_tokens": 256}}},
                    ],
                },
                {
                    "name": "n_gpu_layers",
                    "candidates": [
                        {"suffix": "ngl80", "updates": {"ocr_runtime": {"n_gpu_layers": 80}}},
                        {"suffix": "ngl99", "updates": {"ocr_runtime": {"n_gpu_layers": 99}}},
                    ],
                },
                {
                    "name": "crop_padding_ratio",
                    "candidates": [
                        {"suffix": "crop002", "updates": {"ocr_generic": {"crop_padding_ratio": 0.02}}},
                        {"suffix": "crop005", "updates": {"ocr_generic": {"crop_padding_ratio": 0.05}}},
                    ],
                },
            ],
        },
        {
            "engine": "PPOCRv5 + Gemma",
            "preset": "ocr-combo-ranked-japan-ppocrv5-gemma",
            "tunable": True,
            "tuning_steps": [
                {
                    "name": "retry_crop_ratios",
                    "candidates": [
                        {
                            "suffix": "retry0306",
                            "updates": {
                                "ocr_generic": {
                                    "ppocr_retry_crop_ratio_x": 0.03,
                                    "ppocr_retry_crop_ratio_y": 0.06,
                                }
                            },
                        },
                        {
                            "suffix": "retry0610",
                            "updates": {
                                "ocr_generic": {
                                    "ppocr_retry_crop_ratio_x": 0.06,
                                    "ppocr_retry_crop_ratio_y": 0.10,
                                }
                            },
                        },
                    ],
                },
                {
                    "name": "crop_padding_ratio",
                    "candidates": [
                        {"suffix": "crop002", "updates": {"ocr_generic": {"crop_padding_ratio": 0.02}}},
                        {"suffix": "crop005", "updates": {"ocr_generic": {"crop_padding_ratio": 0.05}}},
                    ],
                },
            ],
        },
    ],
}


def _log(message: str) -> None:
    print(f"[ocr-combo-ranked] {message}", flush=True)


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _enqueue_output(stream, queue: Queue[tuple[str, str]]) -> None:
    try:
        for line in iter(stream.readline, ""):
            queue.put(("line", line))
    finally:
        queue.put(("eof", ""))


def _run_command_streaming(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    output_dir: Path,
    step_name: str,
) -> subprocess.CompletedProcess[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "command_stdout.txt"
    stderr_path = output_dir / "command_stderr.txt"

    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stdout is None:
        raise RuntimeError(f"{step_name} stdout pipe could not be created.")

    queue: Queue[tuple[str, str]] = Queue()
    reader = Thread(target=_enqueue_output, args=(process.stdout, queue), daemon=True)
    reader.start()

    combined: list[str] = []
    eof_seen = False
    while not eof_seen or process.poll() is None:
        try:
            kind, payload = queue.get(timeout=0.2)
        except Empty:
            continue
        if kind == "eof":
            eof_seen = True
            continue
        combined.append(payload)
        print(f"[ocr-combo-ranked][{step_name}] {payload.rstrip()}", flush=True)

    return_code = process.wait()
    output_text = "".join(combined)
    stdout_path.write_text(output_text, encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    return subprocess.CompletedProcess(cmd, return_code, stdout=output_text, stderr="")


def _deep_update(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _materialize_generated_preset(
    *,
    base_preset_ref: str | Path,
    output_path: Path,
    name: str,
    description: str,
    updates: dict[str, Any],
) -> Path:
    payload, _ = load_preset(str(base_preset_ref))
    payload = json.loads(json.dumps(payload))
    payload["name"] = name
    payload["description"] = description
    _deep_update(payload, updates)
    write_json(output_path, payload)
    return output_path


def _snapshot_container_inspect(run_dir: Path, preset_ref: str | Path) -> Path:
    preset, _ = load_preset(str(preset_ref))
    container_names = resolve_runtime_container_names(preset, "full")
    inspect_output = run_command(["docker", "inspect", *container_names], cwd=ROOT, check=False)
    output_path = run_dir / "container_inspect.json"
    try:
        payload = json.loads(inspect_output.stdout or "[]")
    except json.JSONDecodeError:
        payload = {
            "error": "docker inspect output was not JSON",
            "stdout": inspect_output.stdout,
            "stderr": inspect_output.stderr,
            "containers": container_names,
        }
    write_json(output_path, payload if isinstance(payload, dict) else {"inspect": payload})
    return output_path


def _run_pipeline_once(
    *,
    preset_ref: str | Path,
    mode: str,
    sample_dir: Path,
    sample_count: int,
    source_lang: str,
    target_lang: str,
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "benchmark_pipeline.py"),
        "--preset",
        str(preset_ref),
        "--mode",
        mode,
        "--repeat",
        "1",
        "--runtime-mode",
        "managed",
        "--runtime-services",
        "full",
        "--stage-ceiling",
        "render",
        "--sample-dir",
        str(sample_dir),
        "--sample-count",
        str(sample_count),
        "--source-lang",
        source_lang,
        "--target-lang",
        target_lang,
        "--output-dir",
        str(output_dir),
        "--label",
        label,
        "--export-page-snapshots",
        "--clear-app-caches",
    ]
    env = os.environ.copy()
    env["CT_BENCH_OUTPUT_ROOT"] = str(family_output_root())
    completed = _run_command_streaming(
        cmd=cmd,
        cwd=ROOT,
        env=env,
        output_dir=output_dir,
        step_name=label,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed (code={completed.returncode})")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    _snapshot_container_inspect(output_dir, preset_ref)
    return {
        "run_dir": output_dir,
        "preset_input": str(preset_ref),
        "summary": summary,
    }


def _find_first_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _locate_translated_image(run_dir: Path, image_stem: str) -> str:
    translated = _find_first_file(run_dir, f"{image_stem}_translated.*")
    return repo_relative_str(translated) if translated else ""


def _locate_ocr_debug(run_dir: Path, image_stem: str) -> str:
    debug = _find_first_file(run_dir, f"{image_stem}_ocr_debug.json")
    return repo_relative_str(debug) if debug else ""


def _load_gold_payload(gold_path: Path) -> dict[str, Any]:
    payload = _load_json(gold_path)
    if str(payload.get("review_status", "") or "") != "locked":
        raise ValueError(f"Gold file is not locked: {gold_path}")
    pages = payload.get("pages", [])
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"Gold file has no pages: {gold_path}")
    return payload


def _run_compare(gold_path: Path, candidate_run_dir: Path) -> dict[str, Any]:
    output_path = candidate_run_dir / "ocr_combo_ranked_compare.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "compare_ocr_combo_ranked.py"),
            "--gold-path",
            str(gold_path),
            "--candidate-run-dir",
            str(candidate_run_dir),
            "--output",
            str(output_path),
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not output_path.is_file():
        raise RuntimeError(
            f"compare_ocr_combo_ranked.py failed for {candidate_run_dir}\n{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _quality_band_rank(compare_payload: dict[str, Any] | None) -> int:
    if not isinstance(compare_payload, dict):
        return 0
    band = str(compare_payload.get("quality_band", "catastrophic") or "catastrophic")
    return QUALITY_BAND_RANK.get(band, 0)


def _result_rank_key(result: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    compare = result.get("compare", {})
    metrics = compare.get("metrics", {}) if isinstance(compare, dict) else {}
    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    return (
        -float(_quality_band_rank(compare)),
        float(summary.get("elapsed_sec") or 10**12),
        float(metrics.get("ocr_char_error_rate") or 10**12),
        float(metrics.get("page_p95_ocr_char_error_rate") or 10**12),
        float(metrics.get("overgenerated_block_rate") or 10**12),
        float(summary.get("page_failed_count") or 10**12),
        float(summary.get("gemma_truncated_count") or 10**12),
    )


def _annotate_candidate_result(
    *,
    engine: str,
    preset_ref: str | Path,
    run_result: dict[str, Any],
    compare_payload: dict[str, Any] | None,
    corpus_name: str,
    stage: str,
    tunable: bool,
) -> dict[str, Any]:
    payload = {
        "engine": engine,
        "preset": str(preset_ref),
        "run_dir": repo_relative_str(run_result["run_dir"]),
        "summary": run_result["summary"],
        "corpus": corpus_name,
        "stage": stage,
        "tunable": tunable,
    }
    if compare_payload is not None:
        payload["compare"] = compare_payload
        payload["quality_gate_pass"] = bool(compare_payload.get("quality_gate_pass", False))
        payload["quality_band"] = compare_payload.get("quality_band", "catastrophic")
    else:
        payload["quality_gate_pass"] = True
        payload["quality_band"] = "ready"
    return payload


def _median(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _run_smoke_suite(*, suite_dir: Path, corpus_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    smoke_dir = suite_dir / "smoke" / corpus_cfg["name"]
    results: list[dict[str, Any]] = []
    sample_dir = ROOT / "Sample" / corpus_cfg["sample_subdir"]
    for candidate in corpus_cfg["default_candidates"]:
        engine_slug = candidate["engine"].split()[0].lower().replace("+", "")
        label = f"{corpus_cfg['name']}-{engine_slug}-smoke"
        run_result = _run_pipeline_once(
            preset_ref=candidate["preset"],
            mode="one-page",
            sample_dir=sample_dir,
            sample_count=1,
            source_lang=corpus_cfg["source_lang"],
            target_lang=corpus_cfg["target_lang"],
            output_dir=smoke_dir / label,
            label=label,
        )
        results.append(
            _annotate_candidate_result(
                engine=candidate["engine"],
                preset_ref=candidate["preset"],
                run_result=run_result,
                compare_payload=None,
                corpus_name=corpus_cfg["name"],
                stage="smoke",
                tunable=bool(candidate.get("tunable", False)),
            )
        )
    return results


def _run_default_compare(*, suite_dir: Path, corpus_cfg: dict[str, Any], gold_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    compare_dir = suite_dir / "default-compare" / corpus_cfg["name"]
    for candidate in corpus_cfg["default_candidates"]:
        engine_slug = candidate["engine"].split()[0].lower().replace("+", "")
        label = f"{corpus_cfg['name']}-{engine_slug}-default"
        run_result = _run_pipeline_once(
            preset_ref=candidate["preset"],
            mode="batch",
            sample_dir=ROOT / "Sample" / corpus_cfg["sample_subdir"],
            sample_count=corpus_cfg["sample_count"],
            source_lang=corpus_cfg["source_lang"],
            target_lang=corpus_cfg["target_lang"],
            output_dir=compare_dir / label,
            label=label,
        )
        compare_payload = _run_compare(gold_path, run_result["run_dir"])
        results.append(
            _annotate_candidate_result(
                engine=candidate["engine"],
                preset_ref=candidate["preset"],
                run_result=run_result,
                compare_payload=compare_payload,
                corpus_name=corpus_cfg["name"],
                stage="default",
                tunable=bool(candidate.get("tunable", False)),
            )
        )
    return results


def _run_tuning_for_candidate(
    *,
    suite_dir: Path,
    corpus_cfg: dict[str, Any],
    gold_path: Path,
    base_result: dict[str, Any],
    candidate_cfg: dict[str, Any],
    generated_dir: Path,
) -> dict[str, Any]:
    steps = candidate_cfg.get("tuning_steps", [])
    if not isinstance(steps, list) or not steps:
        return {"engine": base_result["engine"], "initial": base_result, "steps": [], "best": base_result}

    current_best = base_result
    step_results: list[dict[str, Any]] = []
    for step in steps:
        step_name = str(step.get("name", "") or "")
        candidates: list[dict[str, Any]] = []
        for candidate_spec in step.get("candidates", []):
            suffix = str(candidate_spec.get("suffix", "") or "")
            preset_name = (
                f"{corpus_cfg['name']}-{base_result['engine'].lower().replace(' ', '-').replace('+', '')}-{suffix}"
            )
            preset_path = generated_dir / f"{preset_name}.json"
            generated_preset = _materialize_generated_preset(
                base_preset_ref=current_best["preset"],
                output_path=preset_path,
                name=preset_name,
                description=f"{base_result['engine']} ranked tuning candidate",
                updates=candidate_spec.get("updates", {}),
            )
            label = f"{corpus_cfg['name']}-{suffix}"
            run_result = _run_pipeline_once(
                preset_ref=generated_preset,
                mode="batch",
                sample_dir=ROOT / "Sample" / corpus_cfg["sample_subdir"],
                sample_count=corpus_cfg["sample_count"],
                source_lang=corpus_cfg["source_lang"],
                target_lang=corpus_cfg["target_lang"],
                output_dir=suite_dir / "tuning" / corpus_cfg["name"] / base_result["engine"].replace(" ", "_") / step_name / label,
                label=label,
            )
            compare_payload = _run_compare(gold_path, run_result["run_dir"])
            candidates.append(
                _annotate_candidate_result(
                    engine=base_result["engine"],
                    preset_ref=generated_preset,
                    run_result=run_result,
                    compare_payload=compare_payload,
                    corpus_name=corpus_cfg["name"],
                    stage=f"tuning:{step_name}",
                    tunable=True,
                )
            )
        best_after_step = min([current_best, *candidates], key=_result_rank_key)
        step_results.append(
            {
                "step": step_name,
                "candidates": candidates,
                "best_after_step": best_after_step,
            }
        )
        current_best = best_after_step
    return {"engine": base_result["engine"], "initial": base_result, "steps": step_results, "best": current_best}


def _aggregate_quality_band(runs: list[dict[str, Any]], fallback_band: str) -> str:
    bands = [
        str(((run.get("compare") or {}).get("quality_band", "")) or "")
        for run in runs
        if isinstance(run, dict)
    ]
    if not bands:
        return fallback_band
    return min(bands, key=lambda band: QUALITY_BAND_RANK.get(band, 0))


def _run_final_confirm(
    *,
    suite_dir: Path,
    corpus_cfg: dict[str, Any],
    gold_path: Path,
    winner_result: dict[str, Any],
) -> dict[str, Any]:
    confirm_runs: list[dict[str, Any]] = []
    for index in range(3):
        label = f"{corpus_cfg['name']}-confirm-r{index + 1}"
        run_result = _run_pipeline_once(
            preset_ref=winner_result["preset"],
            mode="batch",
            sample_dir=ROOT / "Sample" / corpus_cfg["sample_subdir"],
            sample_count=corpus_cfg["sample_count"],
            source_lang=corpus_cfg["source_lang"],
            target_lang=corpus_cfg["target_lang"],
            output_dir=suite_dir / "final-confirm" / corpus_cfg["name"] / f"run-{index + 1}",
            label=label,
        )
        compare_payload = _run_compare(gold_path, run_result["run_dir"])
        confirm_runs.append(
            _annotate_candidate_result(
                engine=winner_result["engine"],
                preset_ref=winner_result["preset"],
                run_result=run_result,
                compare_payload=compare_payload,
                corpus_name=corpus_cfg["name"],
                stage="final-confirm",
                tunable=bool(winner_result.get("tunable", False)),
            )
        )
    winner_status = _aggregate_quality_band(confirm_runs, str(winner_result.get("quality_band", "catastrophic")))
    return {
        "engine": winner_result["engine"],
        "preset": winner_result["preset"],
        "runs": confirm_runs,
        "winner_status": winner_status,
        "official_score_elapsed_median_sec": _median(
            [float(item["summary"].get("elapsed_sec") or 0.0) for item in confirm_runs]
        ),
        "tie_breaker_ocr_median_sec": _median(
            [float(item["summary"].get("ocr_median_sec") or 0.0) for item in confirm_runs]
        ),
        "tie_breaker_translate_median_sec": _median(
            [float(item["summary"].get("translate_median_sec") or 0.0) for item in confirm_runs]
        ),
        "tie_breaker_gpu_peak_used_mb": _median(
            [float(item["summary"].get("gpu_peak_used_mb") or 0.0) for item in confirm_runs]
        ),
    }


def _build_visual_examples(
    *,
    gold_payload: dict[str, Any],
    benchmark_winner: dict[str, Any] | None,
    fastest_candidate: dict[str, Any] | None,
    lowest_cer_candidate: dict[str, Any] | None,
    ppocr_candidate: dict[str, Any] | None,
) -> list[dict[str, str]]:
    pages = gold_payload.get("pages", [])
    active_pages = [
        page
        for page in pages
        if isinstance(page, dict) and str(page.get("status", "active") or "active").lower() != "excluded"
    ]
    examples: list[dict[str, str]] = []
    for page in active_pages[:2]:
        image_stem = str(page.get("image_stem", "") or "")
        if not image_stem:
            continue
        examples.append(
            {
                "page": image_stem,
                "source_image": str(page.get("source_image", "") or ""),
                "overlay_image": str(page.get("overlay_image", "") or ""),
                "winner_translated_image": _locate_translated_image(ROOT / str((benchmark_winner or {}).get("run_dir", "")), image_stem)
                if benchmark_winner
                else "",
                "fastest_translated_image": _locate_translated_image(ROOT / str((fastest_candidate or {}).get("run_dir", "")), image_stem)
                if fastest_candidate
                else "",
                "lowest_cer_translated_image": _locate_translated_image(ROOT / str((lowest_cer_candidate or {}).get("run_dir", "")), image_stem)
                if lowest_cer_candidate
                else "",
                "ppocr_translated_image": _locate_translated_image(ROOT / str((ppocr_candidate or {}).get("run_dir", "")), image_stem)
                if ppocr_candidate
                else "",
            }
        )
    return examples


def _build_regression_examples(
    *,
    benchmark_winner: dict[str, Any] | None,
    ppocr_candidate: dict[str, Any] | None,
    paddle_candidate: dict[str, Any] | None,
) -> list[dict[str, str]]:
    page = "p_018"
    examples: list[dict[str, str]] = []
    if any(item for item in (benchmark_winner, ppocr_candidate, paddle_candidate)):
        examples.append(
            {
                "page": page,
                "winner_ocr_debug": _locate_ocr_debug(ROOT / str((benchmark_winner or {}).get("run_dir", "")), page)
                if benchmark_winner
                else "",
                "ppocr_ocr_debug": _locate_ocr_debug(ROOT / str((ppocr_candidate or {}).get("run_dir", "")), page)
                if ppocr_candidate
                else "",
                "paddle_ocr_debug": _locate_ocr_debug(ROOT / str((paddle_candidate or {}).get("run_dir", "")), page)
                if paddle_candidate
                else "",
            }
        )
    return examples


def _load_frozen_china_manifest() -> dict[str, Any]:
    payload = _load_json(FROZEN_CHINA_MANIFEST)
    required = [
        "corpus",
        "winner_engine",
        "winner_preset",
        "official_score_elapsed_median_sec",
        "winner_status",
        "promotion_recommended",
        "source_run_dir",
    ]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing '{key}' in {FROZEN_CHINA_MANIFEST}")
    return payload


def _build_routing_policy(china_frozen: dict[str, Any], japan_result: dict[str, Any]) -> dict[str, str]:
    china = str(china_frozen.get("winner_engine", "") or "no winner")
    japan = str(((japan_result.get("benchmark_winner") or {}).get("engine", "")) or "no winner")
    if china == japan and china != "no winner":
        mixed = f"China/japan 혼합 운영도 `{china}` 단일 OCR로 시작 가능"
    else:
        mixed = (
            f"중국어 페이지는 `{china}`, 일본어 페이지는 `{japan}`로 라우팅하고 "
            "mixed corpus는 source language 판별 뒤 분기 권장"
        )
    return {"china": china, "japan": japan, "mixed": mixed}


def _write_manifest(suite_dir: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = suite_dir / REPORT_MANIFEST_NAME
    manifest_path.write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    family_output_root().mkdir(parents=True, exist_ok=True)
    write_json(
        family_output_root() / LAST_SUITE_RECORD,
        {
            "manifest": repo_relative_str(manifest_path),
            "suite_dir": repo_relative_str(suite_dir),
            "generated_at": time.time(),
        },
    )
    return manifest_path


def _generate_report(manifest_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "generate_ocr_combo_ranked_report.py"),
            "--manifest",
            str(manifest_path),
        ],
        cwd=str(ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"generate_ocr_combo_ranked_report.py failed\n{completed.stdout}\n{completed.stderr}"
        )


def run_suite(*, sample_root: Path) -> int:
    corpus_cfg = JAPAN_CORPUS
    sample_dir = sample_root / corpus_cfg["sample_subdir"]
    if not sample_dir.is_dir():
        raise FileNotFoundError(f"Missing corpus directory: {sample_dir}")

    gold_path = STRICT_GOLD_ROOT / "japan" / "gold.json"
    gold_payload = _load_gold_payload(gold_path)
    china_frozen = _load_frozen_china_manifest()

    suite_dir = create_run_dir("ocr-combo-ranked-runtime_suite", root=family_output_root())
    generated_dir = suite_dir / "_generated_presets"
    generated_dir.mkdir(parents=True, exist_ok=True)
    _log(f"suite output dir: {suite_dir}")

    _log("smoke start: corpus=japan")
    smoke_results = _run_smoke_suite(suite_dir=suite_dir, corpus_cfg=corpus_cfg)

    _log("default compare start: corpus=japan")
    default_results = _run_default_compare(
        suite_dir=suite_dir,
        corpus_cfg=corpus_cfg,
        gold_path=gold_path,
    )

    tuning_results: list[dict[str, Any]] = []
    engine_best_candidates: list[dict[str, Any]] = []
    candidate_cfg_by_engine = {item["engine"]: item for item in corpus_cfg["default_candidates"]}
    for default_result in default_results:
        candidate_cfg = candidate_cfg_by_engine[default_result["engine"]]
        tuning_result = _run_tuning_for_candidate(
            suite_dir=suite_dir,
            corpus_cfg=corpus_cfg,
            gold_path=gold_path,
            base_result=default_result,
            candidate_cfg=candidate_cfg,
            generated_dir=generated_dir,
        )
        tuning_results.append(tuning_result)
        best = tuning_result.get("best")
        if isinstance(best, dict):
            engine_best_candidates.append(best)

    engine_best_candidates.sort(key=_result_rank_key)
    benchmark_winner = engine_best_candidates[0] if engine_best_candidates else default_results[0]
    final_confirm = _run_final_confirm(
        suite_dir=suite_dir,
        corpus_cfg=corpus_cfg,
        gold_path=gold_path,
        winner_result=benchmark_winner,
    )
    winner_status = str(final_confirm.get("winner_status", benchmark_winner.get("quality_band", "catastrophic")) or "catastrophic")
    promotion_recommended = winner_status == "ready"

    fastest_candidate = min(engine_best_candidates, key=lambda item: float(item["summary"].get("elapsed_sec") or 10**12))
    lowest_cer_candidate = min(
        engine_best_candidates,
        key=lambda item: float(((item.get("compare") or {}).get("metrics") or {}).get("ocr_char_error_rate") or 10**12),
    )
    ppocr_candidate = next((item for item in engine_best_candidates if item.get("engine") == "PPOCRv5 + Gemma"), None)
    paddle_candidate = next((item for item in engine_best_candidates if item.get("engine") == "PaddleOCR VL + Gemma"), None)

    japan_result = {
        "corpus": "japan",
        "sample_dir": f"./Sample/{corpus_cfg['sample_subdir']}",
        "sample_count": corpus_cfg["sample_count"],
        "source_lang": corpus_cfg["source_lang"],
        "target_lang": corpus_cfg["target_lang"],
        "gold_path": repo_relative_str(gold_path),
        "gold_review_status": gold_payload.get("review_status", ""),
        "gold_generated_from_run_dir": gold_payload.get("generated_from_run_dir", ""),
        "default_candidates": default_results,
        "tuning_results": tuning_results,
        "engine_best_candidates": engine_best_candidates,
        "benchmark_winner": benchmark_winner,
        "winner_status": winner_status,
        "promotion_recommended": promotion_recommended,
        "final_confirm": final_confirm,
        "visual_examples": _build_visual_examples(
            gold_payload=gold_payload,
            benchmark_winner=benchmark_winner,
            fastest_candidate=fastest_candidate,
            lowest_cer_candidate=lowest_cer_candidate,
            ppocr_candidate=ppocr_candidate,
        ),
        "regression_examples": _build_regression_examples(
            benchmark_winner=benchmark_winner,
            ppocr_candidate=ppocr_candidate,
            paddle_candidate=paddle_candidate,
        ),
    }

    manifest = {
        "status": "benchmark_complete",
        "benchmark": {
            "name": "OCR Combo Ranked Runtime Benchmark",
            "kind": "managed ranked family suite",
            "scope": "Japan full-pipeline OCR+Gemma timing with OCR-only quality bands and always-winner ranking",
            "execution_scope": "full-pipeline",
            "speed_score_scope": "full-pipeline elapsed_sec",
            "quality_gate_scope": "OCR-only",
            "gold_source": "human-reviewed",
            "ocr_normalization": {
                "canonical_small_voiced_kana": True,
                "ignored_chars": "「」『』,，、♡♥",
            },
            "gold_empty_text_policy": "geometry-kept-text-skipped",
            "crop_regression_focus": "xyxy-first OCR crop with bubble clamp; p_018 overread regression",
            "baseline_sha": _current_git_sha(),
            "develop_ref_sha": _current_git_sha("develop"),
            "entrypoint": r"scripts\ocr_combo_ranked_benchmark_suite_cuda13.bat",
            "fixed_gemma": {
                "image": DEFAULT_LLAMA_CPP_IMAGE,
                "response_format_mode": "json_schema",
                "chunk_size": 6,
                "temperature": 0.6,
                "n_gpu_layers": 23,
            },
        },
        "results_root": _repo_root_results_root(),
        "report": {
            "markdown_output": "docs/banchmark_report/ocr-combo-ranked-report-ko.md",
            "assets_dir": "docs/assets/benchmarking/ocr-combo-ranked/latest",
        },
        "china_frozen": china_frozen,
        "smoke_results": smoke_results,
        "corpora": [japan_result],
        "routing_policy": _build_routing_policy(china_frozen, japan_result),
    }
    manifest_path = _write_manifest(suite_dir, manifest)
    _generate_report(manifest_path)
    _log(f"suite manifest written: {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OCR combo ranked Japan benchmark suite.")
    parser.add_argument("--sample-root", default=str(ROOT / "Sample"))
    args = parser.parse_args()
    return run_suite(sample_root=Path(args.sample_root))


if __name__ == "__main__":
    raise SystemExit(main())
