#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
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
from PIL import Image, ImageDraw, ImageFont

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

FAMILY_NAME = "ocr_combo"
FAMILY_OUTPUT_ROOT_NAME = "ocr_combo"
LAST_SUITE_RECORD = "last_ocr_combo_suite.json"
REPORT_MANIFEST_NAME = "ocr_combo_report_manifest.yaml"
GOLD_ROOT_ENV = "CT_OCR_COMBO_GOLD_ROOT"
GOLD_REVIEW_STATUS_DRAFT = "draft"
GOLD_REVIEW_STATUS_LOCKED = "locked"

CORPORA = [
    {
        "name": "china",
        "sample_subdir": "China",
        "sample_count": 8,
        "source_lang": "Chinese",
        "target_lang": "Korean",
        "seed_preset": "ocr-combo-china-reference",
        "default_candidates": [
            {"engine": "PPOCRv5 + Gemma", "preset": "ocr-combo-default-gemma", "tunable": False},
            {"engine": "PaddleOCR VL + Gemma", "preset": "ocr-combo-china-reference", "tunable": True},
            {"engine": "HunyuanOCR + Gemma", "preset": "ocr-combo-hunyuanocr-gemma", "tunable": True},
        ],
    },
    {
        "name": "japan",
        "sample_subdir": "japan",
        "sample_count": 22,
        "source_lang": "Japanese",
        "target_lang": "Korean",
        "seed_preset": "ocr-combo-japan-reference",
        "default_candidates": [
            {"engine": "MangaOCR + Gemma", "preset": "ocr-combo-default-gemma", "tunable": False},
            {"engine": "PaddleOCR VL + Gemma", "preset": "ocr-combo-japan-reference", "tunable": True},
            {"engine": "HunyuanOCR + Gemma", "preset": "ocr-combo-hunyuanocr-gemma", "tunable": True},
        ],
    },
]

PADDLE_TUNING_STEPS = [
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
]

HUNYUAN_TUNING_STEPS = [
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
]


def _log(message: str) -> None:
    print(f"[ocr-combo] {message}", flush=True)


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


def gold_root() -> Path:
    env_root = os.getenv(GOLD_ROOT_ENV, "").strip()
    root = Path(env_root) if env_root else ROOT / "benchmarks" / FAMILY_NAME / "gold"
    root.mkdir(parents=True, exist_ok=True)
    return root


def gold_path_for_corpus(corpus_name: str) -> Path:
    corpus_dir = gold_root() / corpus_name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    return corpus_dir / "gold.json"


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
        print(f"[ocr-combo][{step_name}] {payload.rstrip()}", flush=True)

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
    inspect_output = run_command(
        ["docker", "inspect", *container_names],
        cwd=ROOT,
        check=False,
    )
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


def _load_page_snapshots(run_dir: Path) -> dict[str, Any]:
    return _load_json(run_dir / "page_snapshots.json")


def _normalize_text(text: object) -> str:
    return "".join(str(text or "").split())


def _find_first_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def _load_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def _write_overlay_image(source_image_path: Path, blocks: list[dict[str, Any]], output_path: Path) -> None:
    image = Image.open(source_image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = _load_font()
    for idx, block in enumerate(blocks):
        xyxy = block.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(float(value)) for value in xyxy]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        label = str(idx)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_bbox, fill=(255, 255, 0))
        draw.text((x1, y1), label, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _gold_payload_for_reference(
    *,
    corpus_cfg: dict[str, Any],
    reference_run_dir: Path,
    suite_dir: Path,
) -> dict[str, Any]:
    snapshots = _load_page_snapshots(reference_run_dir)
    review_dir = suite_dir / "gold-review" / corpus_cfg["name"]
    source_dir = review_dir / "source_images"
    overlay_dir = review_dir / "overlay_images"
    translated_dir = review_dir / "translated_images"
    debug_dir = review_dir / "ocr_debugs"
    for path in (source_dir, overlay_dir, translated_dir, debug_dir):
        path.mkdir(parents=True, exist_ok=True)

    pages_payload: list[dict[str, Any]] = []
    review_examples: list[dict[str, Any]] = []
    for page in snapshots.get("pages", []):
        if not isinstance(page, dict):
            continue
        image_stem = str(page.get("image_stem", "") or "")
        image_name = str(page.get("image_name", "") or "")
        source_image_path = Path(str(page.get("image_path", "") or ""))
        if not image_stem or not source_image_path.is_file():
            continue

        copied_source = source_dir / image_name
        shutil.copy2(source_image_path, copied_source)

        blocks = page.get("blocks", []) if isinstance(page.get("blocks"), list) else []
        overlay_path = overlay_dir / f"{image_stem}_overlay.png"
        _write_overlay_image(source_image_path, blocks, overlay_path)

        debug_source = _find_first_file(reference_run_dir, f"{image_stem}_ocr_debug.json")
        copied_debug: Path | None = None
        if debug_source is not None and debug_source.is_file():
            copied_debug = debug_dir / debug_source.name
            shutil.copy2(debug_source, copied_debug)

        translated_source = _find_first_file(reference_run_dir, f"{image_stem}_translated.*")
        copied_translated: Path | None = None
        if translated_source is not None and translated_source.is_file():
            copied_translated = translated_dir / translated_source.name
            shutil.copy2(translated_source, copied_translated)

        gold_blocks = []
        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            gold_blocks.append(
                {
                    "index": idx,
                    "xyxy": block.get("xyxy"),
                    "bubble_xyxy": block.get("bubble_xyxy"),
                    "angle": block.get("angle"),
                    "text_class": block.get("text_class"),
                    "seed_text": str(block.get("text", "") or ""),
                    "gold_text": str(block.get("text", "") or ""),
                    "seed_normalized_text": str(block.get("normalized_text", "") or ""),
                    "seed_translation": str(block.get("translation", "") or ""),
                    "seed_normalized_translation": str(block.get("normalized_translation", "") or ""),
                }
            )

        page_payload = {
            "image_stem": image_stem,
            "image_name": image_name,
            "source_image": repo_relative_str(copied_source),
            "overlay_image": repo_relative_str(overlay_path),
            "ocr_debug": repo_relative_str(copied_debug) if copied_debug else "",
            "translated_image": repo_relative_str(copied_translated) if copied_translated else "",
            "status": "active",
            "exclude_reason": "",
            "blocks": gold_blocks,
        }
        pages_payload.append(page_payload)
        if len(review_examples) < 2:
            review_examples.append(
                {
                    "page": image_stem,
                    "source_image": page_payload["source_image"],
                    "overlay_image": page_payload["overlay_image"],
                    "translated_image": page_payload["translated_image"],
                    "ocr_debug": page_payload["ocr_debug"],
                }
            )

    payload = {
        "schema_version": 1,
        "corpus": corpus_cfg["name"],
        "source_lang": corpus_cfg["source_lang"],
        "target_lang": corpus_cfg["target_lang"],
        "review_status": GOLD_REVIEW_STATUS_DRAFT,
        "generated_at": datetime.now().astimezone().isoformat(),
        "generated_from_run_dir": repo_relative_str(reference_run_dir),
        "review_packet_dir": repo_relative_str(review_dir),
        "pages": pages_payload,
    }

    gold_path = gold_path_for_corpus(corpus_cfg["name"])
    write_json(gold_path, payload)
    write_json(review_dir / "editable_gold.json", payload)
    write_json(
        review_dir / "packet_manifest.json",
        {
            "corpus": corpus_cfg["name"],
            "review_status": GOLD_REVIEW_STATUS_DRAFT,
            "gold_path": repo_relative_str(gold_path),
            "review_packet_dir": repo_relative_str(review_dir),
            "page_count": len(pages_payload),
            "examples": review_examples,
        },
    )
    return {
        "gold_path": repo_relative_str(gold_path),
        "review_packet_dir": repo_relative_str(review_dir),
        "review_status": GOLD_REVIEW_STATUS_DRAFT,
        "generated_from_run_dir": repo_relative_str(reference_run_dir),
        "page_count": len(pages_payload),
        "review_examples": review_examples,
    }


def _load_gold_state(corpus_cfg: dict[str, Any]) -> dict[str, Any]:
    path = gold_path_for_corpus(corpus_cfg["name"])
    if not path.exists():
        return {"status": "missing", "path": path}
    payload = _load_json(path)
    review_status = str(payload.get("review_status", GOLD_REVIEW_STATUS_DRAFT) or GOLD_REVIEW_STATUS_DRAFT)
    return {
        "status": review_status,
        "path": path,
        "payload": payload,
    }


def _validate_gold_payload(gold_path: Path) -> dict[str, Any]:
    payload = _load_json(gold_path)
    review_status = str(payload.get("review_status", "") or "")
    if review_status != GOLD_REVIEW_STATUS_LOCKED:
        raise ValueError(f"Gold file is not locked: {gold_path}")
    pages = payload.get("pages", [])
    if not isinstance(pages, list) or not pages:
        raise ValueError(f"Gold file has no pages: {gold_path}")
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError(f"Invalid page payload in {gold_path}")
        image_stem = str(page.get("image_stem", "") or "")
        status = str(page.get("status", "active") or "active")
        if not image_stem:
            raise ValueError(f"Gold page missing image_stem in {gold_path}")
        if status == "excluded":
            continue
        blocks = page.get("blocks", [])
        if not isinstance(blocks, list):
            raise ValueError(f"Gold page blocks must be a list for {image_stem} in {gold_path}")
        for block in blocks:
            if not isinstance(block, dict):
                raise ValueError(f"Invalid block payload for {image_stem} in {gold_path}")
            if "xyxy" not in block or "gold_text" not in block:
                raise ValueError(f"Gold block missing xyxy/gold_text for {image_stem} in {gold_path}")
    return payload


def _locate_translated_image(run_dir: Path, image_stem: str) -> str:
    translated = _find_first_file(run_dir, f"{image_stem}_translated.*")
    return repo_relative_str(translated) if translated else ""


def _run_compare(gold_path: Path, candidate_run_dir: Path) -> dict[str, Any]:
    output_path = candidate_run_dir / "ocr_combo_compare.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "compare_ocr_combo_reference.py"),
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
            f"compare failed for {candidate_run_dir}\n{completed.stdout}\n{completed.stderr}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _result_rank_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    summary = result.get("summary", {})
    return (
        float(summary.get("elapsed_sec") or 10**12),
        float(summary.get("ocr_median_sec") or 10**12),
        float(summary.get("translate_median_sec") or 10**12),
        float(summary.get("gpu_peak_used_mb") or 10**12),
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
    else:
        payload["quality_gate_pass"] = True
    return payload


def _median(values: list[float]) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _run_smoke_suite(
    *,
    suite_dir: Path,
    corpus_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    smoke_dir = suite_dir / "smoke" / corpus_cfg["name"]
    results: list[dict[str, Any]] = []
    sample_dir = ROOT / "Sample" / corpus_cfg["sample_subdir"]
    for candidate in corpus_cfg["default_candidates"]:
        label = f"{corpus_cfg['name']}-{candidate['engine'].split()[0].lower()}-smoke"
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


def _run_reference_seed(
    *,
    suite_dir: Path,
    corpus_cfg: dict[str, Any],
) -> dict[str, Any]:
    seed_dir = suite_dir / "reference-seed" / corpus_cfg["name"]
    run_result = _run_pipeline_once(
        preset_ref=corpus_cfg["seed_preset"],
        mode="batch",
        sample_dir=ROOT / "Sample" / corpus_cfg["sample_subdir"],
        sample_count=corpus_cfg["sample_count"],
        source_lang=corpus_cfg["source_lang"],
        target_lang=corpus_cfg["target_lang"],
        output_dir=seed_dir,
        label=f"{corpus_cfg['name']}-gold-seed",
    )
    return _annotate_candidate_result(
        engine="PaddleOCR VL + Gemma",
        preset_ref=corpus_cfg["seed_preset"],
        run_result=run_result,
        compare_payload=None,
        corpus_name=corpus_cfg["name"],
        stage="gold-seed",
        tunable=True,
    )


def _run_default_compare(
    *,
    suite_dir: Path,
    corpus_cfg: dict[str, Any],
    gold_path: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    compare_dir = suite_dir / "default-compare" / corpus_cfg["name"]
    for candidate in corpus_cfg["default_candidates"]:
        engine_slug = candidate["engine"].split()[0].lower()
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


def _candidate_tuning_steps(engine: str) -> list[dict[str, Any]]:
    if engine == "PaddleOCR VL + Gemma":
        return PADDLE_TUNING_STEPS
    if engine == "HunyuanOCR + Gemma":
        return HUNYUAN_TUNING_STEPS
    return []


def _run_tuning_for_candidate(
    *,
    suite_dir: Path,
    corpus_cfg: dict[str, Any],
    gold_path: Path,
    base_result: dict[str, Any],
    generated_dir: Path,
) -> dict[str, Any]:
    engine = str(base_result["engine"])
    steps = _candidate_tuning_steps(engine)
    if not steps:
        return {"engine": engine, "initial": base_result, "steps": [], "best": base_result}

    current_best = base_result
    step_results: list[dict[str, Any]] = []
    for step in steps:
        step_name = str(step["name"])
        candidates: list[dict[str, Any]] = []
        for candidate_spec in step["candidates"]:
            preset_name = (
                f"{corpus_cfg['name']}-{engine.lower().replace(' ', '-').replace('+', '')}-{candidate_spec['suffix']}"
            )
            preset_path = generated_dir / f"{preset_name}.json"
            generated_preset = _materialize_generated_preset(
                base_preset_ref=current_best["preset"],
                output_path=preset_path,
                name=preset_name,
                description=f"{engine} {step_name} tuning candidate",
                updates=candidate_spec["updates"],
            )
            label = f"{corpus_cfg['name']}-{candidate_spec['suffix']}"
            run_result = _run_pipeline_once(
                preset_ref=generated_preset,
                mode="batch",
                sample_dir=ROOT / "Sample" / corpus_cfg["sample_subdir"],
                sample_count=corpus_cfg["sample_count"],
                source_lang=corpus_cfg["source_lang"],
                target_lang=corpus_cfg["target_lang"],
                output_dir=suite_dir / "tuning" / corpus_cfg["name"] / engine.replace(" ", "_") / step_name / label,
                label=label,
            )
            compare_payload = _run_compare(gold_path, run_result["run_dir"])
            candidates.append(
                _annotate_candidate_result(
                    engine=engine,
                    preset_ref=generated_preset,
                    run_result=run_result,
                    compare_payload=compare_payload,
                    corpus_name=corpus_cfg["name"],
                    stage=f"tuning:{step_name}",
                    tunable=True,
                )
            )
        passing = [item for item in candidates if item.get("quality_gate_pass")]
        best_after_step = min([current_best, *passing], key=_result_rank_key) if passing else current_best
        step_results.append(
            {
                "step": step_name,
                "candidates": candidates,
                "best_after_step": best_after_step,
            }
        )
        current_best = best_after_step
    return {"engine": engine, "initial": base_result, "steps": step_results, "best": current_best}


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
    all_pass = all(item.get("quality_gate_pass") for item in confirm_runs)
    official_score = _median([float(item["summary"].get("elapsed_sec") or 0.0) for item in confirm_runs])
    ocr_median = _median([float(item["summary"].get("ocr_median_sec") or 0.0) for item in confirm_runs])
    translate_median = _median(
        [float(item["summary"].get("translate_median_sec") or 0.0) for item in confirm_runs]
    )
    gpu_peak = _median([float(item["summary"].get("gpu_peak_used_mb") or 0.0) for item in confirm_runs])
    return {
        "engine": winner_result["engine"],
        "preset": winner_result["preset"],
        "runs": confirm_runs,
        "all_quality_gate_pass": all_pass,
        "official_score_elapsed_median_sec": official_score,
        "tie_breaker_ocr_median_sec": ocr_median,
        "tie_breaker_translate_median_sec": translate_median,
        "tie_breaker_gpu_peak_used_mb": gpu_peak,
    }


def _build_routing_policy(corpus_results: list[dict[str, Any]]) -> dict[str, str]:
    winner_by_corpus = {item["corpus"]: item.get("winner", {}) for item in corpus_results}
    china = str((winner_by_corpus.get("china") or {}).get("engine", "") or "no winner")
    japan = str((winner_by_corpus.get("japan") or {}).get("engine", "") or "no winner")
    if china == japan and china != "no winner":
        mixed = f"China/japan 혼합 운영도 `{china}` 단일 OCR로 시작 가능"
    else:
        mixed = (
            f"중국어 페이지는 `{china}`, 일본어 페이지는 `{japan}`로 라우팅하고 "
            "mixed corpus는 source language 판별 뒤 분기 권장"
        )
    return {
        "china": china,
        "japan": japan,
        "mixed": mixed,
    }


def _build_visual_examples(
    *,
    gold_payload: dict[str, Any],
    winner: dict[str, Any] | None,
    fastest_failed: dict[str, Any] | None,
) -> list[dict[str, str]]:
    pages = gold_payload.get("pages", [])
    examples: list[dict[str, str]] = []
    active_pages = [
        page for page in pages if isinstance(page, dict) and str(page.get("status", "active") or "active") != "excluded"
    ]
    for page in active_pages[:2]:
        image_stem = str(page.get("image_stem", "") or "")
        if not image_stem:
            continue
        winner_image = _locate_translated_image(ROOT / str(winner.get("run_dir", "")), image_stem) if winner else ""
        failed_image = (
            _locate_translated_image(ROOT / str(fastest_failed.get("run_dir", "")), image_stem)
            if fastest_failed
            else ""
        )
        examples.append(
            {
                "page": image_stem,
                "source_image": str(page.get("source_image", "") or ""),
                "overlay_image": str(page.get("overlay_image", "") or ""),
                "winner_translated_image": winner_image,
                "fastest_failed_translated_image": failed_image,
            }
        )
    return examples


def _write_manifest(suite_dir: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = suite_dir / REPORT_MANIFEST_NAME
    manifest_path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
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
            str(ROOT / "scripts" / "generate_ocr_combo_report.py"),
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
            f"generate_ocr_combo_report.py failed\n{completed.stdout}\n{completed.stderr}"
        )


def _bootstrap_manifest(
    *,
    suite_dir: Path,
    corpus_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": "awaiting_gold_review",
        "benchmark": {
            "name": "OCR Combo Runtime Benchmark",
            "kind": "managed family suite",
            "scope": "language-aware OCR+Gemma comparison using benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime",
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
            "entrypoint": r"scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime",
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
            "markdown_output": "docs/banchmark_report/ocr-combo-report-ko.md",
            "assets_dir": "docs/assets/benchmarking/ocr-combo/latest",
        },
        "corpora": corpus_entries,
        "routing_policy": {
            "china": "awaiting_gold_review",
            "japan": "awaiting_gold_review",
            "mixed": "OCR gold review를 먼저 완료해야 mixed corpus 정책을 확정할 수 있습니다.",
        },
    }


def run_suite(*, sample_root: Path) -> int:
    for corpus in CORPORA:
        corpus_dir = sample_root / corpus["sample_subdir"]
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"Missing corpus directory: {corpus_dir}")

    suite_dir = create_run_dir("ocr-combo-runtime_suite", root=family_output_root())
    generated_dir = suite_dir / "_generated_presets"
    generated_dir.mkdir(parents=True, exist_ok=True)
    _log(f"suite output dir: {suite_dir}")

    gold_states = {corpus["name"]: _load_gold_state(corpus) for corpus in CORPORA}
    if not all(state.get("status") == GOLD_REVIEW_STATUS_LOCKED for state in gold_states.values()):
        corpus_entries: list[dict[str, Any]] = []
        for corpus_cfg in CORPORA:
            state = gold_states[corpus_cfg["name"]]
            if state.get("status") == "missing":
                _log(f"gold bootstrap start: corpus={corpus_cfg['name']}")
                seed_result = _run_reference_seed(suite_dir=suite_dir, corpus_cfg=corpus_cfg)
                bootstrap_info = _gold_payload_for_reference(
                    corpus_cfg=corpus_cfg,
                    reference_run_dir=ROOT / seed_result["run_dir"],
                    suite_dir=suite_dir,
                )
                corpus_entries.append(
                    {
                        "corpus": corpus_cfg["name"],
                        "sample_dir": f"./Sample/{corpus_cfg['sample_subdir']}",
                        "sample_count": corpus_cfg["sample_count"],
                        "source_lang": corpus_cfg["source_lang"],
                        "target_lang": corpus_cfg["target_lang"],
                        "gold_path": bootstrap_info["gold_path"],
                        "gold_review_status": bootstrap_info["review_status"],
                        "gold_review_packet_dir": bootstrap_info["review_packet_dir"],
                        "gold_generated_from_run_dir": bootstrap_info["generated_from_run_dir"],
                        "gold_page_count": bootstrap_info["page_count"],
                        "reference_seed": seed_result,
                        "review_examples": bootstrap_info["review_examples"],
                    }
                )
            else:
                payload = state.get("payload", {})
                pages = payload.get("pages", []) if isinstance(payload.get("pages"), list) else []
                review_examples = []
                for page in pages[:2]:
                    if not isinstance(page, dict):
                        continue
                    review_examples.append(
                        {
                            "page": page.get("image_stem", ""),
                            "source_image": page.get("source_image", ""),
                            "overlay_image": page.get("overlay_image", ""),
                            "translated_image": page.get("translated_image", ""),
                            "ocr_debug": page.get("ocr_debug", ""),
                        }
                    )
                corpus_entries.append(
                    {
                        "corpus": corpus_cfg["name"],
                        "sample_dir": f"./Sample/{corpus_cfg['sample_subdir']}",
                        "sample_count": corpus_cfg["sample_count"],
                        "source_lang": corpus_cfg["source_lang"],
                        "target_lang": corpus_cfg["target_lang"],
                        "gold_path": repo_relative_str(state["path"]),
                        "gold_review_status": state["status"],
                        "gold_review_packet_dir": payload.get("review_packet_dir", ""),
                        "gold_generated_from_run_dir": payload.get("generated_from_run_dir", ""),
                        "gold_page_count": len(pages),
                        "reference_seed": {},
                        "review_examples": review_examples,
                    }
                )

        manifest = _bootstrap_manifest(suite_dir=suite_dir, corpus_entries=corpus_entries)
        manifest_path = _write_manifest(suite_dir, manifest)
        _generate_report(manifest_path)
        _log(f"gold review pending: {manifest_path}")
        return 0

    smoke_results: list[dict[str, Any]] = []
    corpus_results: list[dict[str, Any]] = []

    for corpus_cfg in CORPORA:
        gold_payload = _validate_gold_payload(gold_states[corpus_cfg["name"]]["path"])
        gold_path = gold_states[corpus_cfg["name"]]["path"]

        _log(f"smoke start: corpus={corpus_cfg['name']}")
        smoke_results.extend(_run_smoke_suite(suite_dir=suite_dir, corpus_cfg=corpus_cfg))

        _log(f"default compare start: corpus={corpus_cfg['name']}")
        default_results = _run_default_compare(
            suite_dir=suite_dir,
            corpus_cfg=corpus_cfg,
            gold_path=gold_path,
        )
        passing_defaults = [item for item in default_results if item.get("quality_gate_pass")]
        passing_defaults.sort(key=_result_rank_key)
        tuning_inputs = [item for item in passing_defaults if item.get("tunable")][:2]
        tuning_results: list[dict[str, Any]] = []
        for candidate in tuning_inputs:
            tuning_results.append(
                _run_tuning_for_candidate(
                    suite_dir=suite_dir,
                    corpus_cfg=corpus_cfg,
                    gold_path=gold_path,
                    base_result=candidate,
                    generated_dir=generated_dir,
                )
            )

        winner_pool = list(passing_defaults)
        for tuning_result in tuning_results:
            best = tuning_result.get("best")
            if isinstance(best, dict) and best.get("quality_gate_pass"):
                winner_pool.append(best)
        winner_pool.sort(key=_result_rank_key)
        winner = winner_pool[0] if winner_pool else None
        final_confirm = (
            _run_final_confirm(
                suite_dir=suite_dir,
                corpus_cfg=corpus_cfg,
                gold_path=gold_path,
                winner_result=winner,
            )
            if winner
            else {}
        )
        promotion_recommended = bool(winner and final_confirm.get("all_quality_gate_pass"))
        failed_candidates = [item for item in default_results if not item.get("quality_gate_pass")]
        failed_candidates.sort(key=_result_rank_key)
        fastest_failed = failed_candidates[0] if failed_candidates else None
        corpus_results.append(
            {
                "corpus": corpus_cfg["name"],
                "sample_dir": f"./Sample/{corpus_cfg['sample_subdir']}",
                "sample_count": corpus_cfg["sample_count"],
                "source_lang": corpus_cfg["source_lang"],
                "target_lang": corpus_cfg["target_lang"],
                "gold_path": repo_relative_str(gold_path),
                "gold_review_status": gold_payload.get("review_status", ""),
                "gold_generated_from_run_dir": gold_payload.get("generated_from_run_dir", ""),
                "default_candidates": default_results,
                "tuning_results": tuning_results,
                "winner": winner or {},
                "fastest_failed_candidate": fastest_failed or {},
                "final_confirm": final_confirm,
                "promotion_recommended": promotion_recommended,
                "visual_examples": _build_visual_examples(
                    gold_payload=gold_payload,
                    winner=winner,
                    fastest_failed=fastest_failed,
                ),
            }
        )

    manifest = {
        "status": "benchmark_complete",
        "benchmark": {
            "name": "OCR Combo Runtime Benchmark",
            "kind": "managed family suite",
            "scope": "language-aware OCR+Gemma comparison using benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime",
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
            "entrypoint": r"scripts\benchmark_suite_cuda13.bat --suite-profile ocr-combo-runtime",
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
            "markdown_output": "docs/banchmark_report/ocr-combo-report-ko.md",
            "assets_dir": "docs/assets/benchmarking/ocr-combo/latest",
        },
        "smoke_results": smoke_results,
        "corpora": corpus_results,
        "routing_policy": _build_routing_policy(corpus_results),
    }
    manifest_path = _write_manifest(suite_dir, manifest)
    _generate_report(manifest_path)
    _log(f"suite manifest written: {manifest_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the OCR combo language-aware benchmark suite.")
    parser.add_argument("--sample-root", default=str(ROOT / "Sample"))
    args = parser.parse_args()
    return run_suite(sample_root=Path(args.sample_root))


if __name__ == "__main__":
    raise SystemExit(main())
