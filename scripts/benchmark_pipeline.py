#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("CT_DISABLE_UPDATE_CHECK", "1")
os.environ.setdefault("CT_ENABLE_MEMLOG", "1")
os.environ.setdefault("CT_ENABLE_GPU_BENCH", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from benchmark_common import (
    DEFAULT_CONTAINER_NAMES,
    create_run_dir,
    load_preset,
    render_summary_markdown,
    resolve_corpus,
    run_command,
    stage_runtime_files,
    summarize_metrics,
    write_json,
)
from modules.utils.gpu_metrics import collect_runtime_snapshot, write_snapshot_json


def _settings_snapshot() -> dict[str, object]:
    settings = QSettings("ComicLabs", "ComicTranslate")
    return {key: settings.value(key) for key in settings.allKeys()}


def _restore_settings(snapshot: dict[str, object]) -> None:
    settings = QSettings("ComicLabs", "ComicTranslate")
    settings.clear()
    for key, value in snapshot.items():
        settings.setValue(key, value)
    settings.sync()


def _wait_for_url(url: str, timeout_sec: int = 180) -> None:
    started = time.time()
    while time.time() - started < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=5):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}")


def _ensure_managed_runtime(run_dir: Path, preset: dict[str, object]) -> None:
    runtime_dir = run_dir / "runtime"
    staged = stage_runtime_files(preset, runtime_dir)
    run_command(
        ["docker", "compose", "-f", staged["gemma"]["compose_path"], "up", "-d", "--force-recreate"],
        cwd=runtime_dir / "gemma",
    )
    run_command(
        ["docker", "compose", "-f", staged["ocr"]["compose_path"], "up", "-d", "--force-recreate"],
        cwd=runtime_dir / "ocr",
    )
    _wait_for_url("http://127.0.0.1:18080/health")
    _wait_for_url("http://127.0.0.1:18000/v1/models")
    _wait_for_url("http://127.0.0.1:28118/docs")


def _configure_window(window, preset: dict[str, object], source_lang: str, target_lang: str) -> None:
    ui = window.settings_page.ui
    app_config = preset.get("app", {})
    gemma = preset.get("gemma", {})
    ocr_client = preset.get("ocr_client", {})

    ui.use_gpu_checkbox.setChecked(bool(app_config.get("use_gpu", True)))
    ui.translator_combo.setCurrentText(str(app_config.get("translator", "Custom Local Server(Gemma)")))
    ui.ocr_combo.setCurrentText(str(app_config.get("ocr", "PaddleOCR VL")))
    ui.detector_combo.setCurrentText(str(app_config.get("detector", "RT-DETR-v2")))
    ui.inpainter_combo.setCurrentText(str(app_config.get("inpainter", "AOT")))

    ui.save_keys_checkbox.setChecked(True)
    ui.extra_context.setPlainText(str(app_config.get("extra_context", "")))
    ui.credential_widgets["Custom Local Server(Gemma)_api_url"].setText(
        str(gemma.get("endpoint_url", "http://127.0.0.1:18080/v1"))
    )
    ui.credential_widgets["Custom Local Server(Gemma)_model"].setText(
        str(gemma.get("model", "gemma-4-26b-a4b-it-heretic.q3_k_m.gguf"))
    )

    ui.paddleocr_vl_server_url_input.setText(
        str(ocr_client.get("server_url", "http://127.0.0.1:28118/layout-parsing"))
    )
    ui.paddleocr_vl_max_new_tokens_spinbox.setValue(int(ocr_client.get("max_new_tokens", 256)))
    ui.paddleocr_vl_parallel_workers_spinbox.setValue(int(ocr_client.get("parallel_workers", 2)))

    ui.gemma_chunk_size_spinbox.setValue(int(gemma.get("chunk_size", 4)))
    ui.gemma_max_completion_tokens_spinbox.setValue(int(gemma.get("max_completion_tokens", 512)))
    ui.gemma_request_timeout_spinbox.setValue(int(gemma.get("request_timeout_sec", 180)))
    ui.gemma_raw_response_logging_checkbox.setChecked(bool(gemma.get("raw_response_logging", False)))

    window.s_combo.setCurrentText(source_lang)
    window.t_combo.setCurrentText(target_lang)


def _load_images(window, image_paths: list[Path], source_lang: str, target_lang: str) -> list[str]:
    loaded = window.image_ctrl.load_initial_image([str(path) for path in image_paths])
    window.image_ctrl.on_initial_image_loaded(loaded)
    for image_path in window.image_files:
        state = window.image_ctrl.ensure_page_state(image_path)
        state["source_lang"] = source_lang
        state["target_lang"] = target_lang
    return list(window.image_files)


def _run_single_mode(
    *,
    app: QApplication,
    preset: dict[str, object],
    mode: str,
    run_dir: Path,
    source_lang: str,
    target_lang: str,
    image_paths: list[Path],
) -> dict[str, object]:
    os.environ["CT_BENCH_OUTPUT_DIR"] = str(run_dir)
    try:
        from controller import ComicTranslate
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown-module"
        raise RuntimeError(
            "Benchmark runtime is missing a required dependency: "
            f"{missing}. Install the full app runtime before running pipeline benchmarks."
        ) from exc

    settings_backup = _settings_snapshot()
    window = ComicTranslate()
    try:
        _configure_window(window, preset, source_lang, target_lang)
        loaded_paths = _load_images(window, image_paths, source_lang, target_lang)
        window.curr_img_idx = 0
        window._current_batch_run_type = "one_page_auto" if mode == "one-page" else "batch"
        window.emit_memlog(
            "benchmark_run_start",
            benchmark_mode=mode,
            total_images=len(loaded_paths),
        )

        started = time.perf_counter()
        if mode == "one-page":
            window.pipeline.batch_process([loaded_paths[0]])
        elif mode == "batch":
            window.pipeline.batch_process(loaded_paths)
        elif mode == "webtoon":
            window.pipeline.webtoon_batch_process(loaded_paths)
        else:
            raise ValueError(f"Unsupported benchmark mode: {mode}")
        elapsed = time.perf_counter() - started

        window.pipeline.release_model_caches()
        window.emit_memlog(
            "benchmark_run_finished",
            benchmark_mode=mode,
            elapsed_sec=round(elapsed, 3),
        )
        app.processEvents()
    finally:
        try:
            window._skip_close_prompt = True
            window.close()
            app.processEvents()
        finally:
            _restore_settings(settings_backup)

    metrics_path = run_dir / "metrics.jsonl"
    summary = summarize_metrics(metrics_path)
    summary.update(
        {
            "mode": mode,
            "image_count": len(image_paths) if mode != "one-page" else 1,
            "image_paths": [str(path) for path in image_paths],
        }
    )
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offscreen pipeline benchmarks.")
    parser.add_argument("--preset", required=True, help="Preset name or preset file path")
    parser.add_argument(
        "--mode",
        default="all",
        choices=("one-page", "batch", "webtoon", "all"),
        help="Pipeline mode to benchmark",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats")
    parser.add_argument("--sample-dir", default=str(ROOT / "Sample"), help="Local Sample directory")
    parser.add_argument("--sample-count", type=int, default=30, help="Representative corpus size")
    parser.add_argument("--source-lang", default="Chinese")
    parser.add_argument("--target-lang", default="Korean")
    parser.add_argument(
        "--runtime-mode",
        default="attach-running",
        choices=("attach-running", "managed"),
        help="Use already running services or recreate staged runtime files",
    )
    parser.add_argument("--label", default="", help="Optional label suffix for run directory names")
    args = parser.parse_args()

    preset, preset_path = load_preset(args.preset)
    corpus = resolve_corpus(args.sample_dir, sample_count=args.sample_count)
    modes = ["one-page", "batch", "webtoon"] if args.mode == "all" else [args.mode]

    app = QApplication.instance() or QApplication([])
    aggregated = []
    for repeat_index in range(1, max(1, args.repeat) + 1):
        for mode in modes:
            label = args.label or preset.get("name", args.preset)
            run_dir = create_run_dir(f"{label}_{mode}_r{repeat_index}")
            run_dir.mkdir(parents=True, exist_ok=True)

            selected_paths = corpus["smoke"] if mode == "one-page" else corpus["representative"]
            if mode == "one-page":
                selected_paths = selected_paths[:1]

            write_json(
                run_dir / "benchmark_request.json",
                {
                    "preset_name": preset.get("name", args.preset),
                    "preset_path": str(preset_path),
                    "mode": mode,
                    "repeat_index": repeat_index,
                    "runtime_mode": args.runtime_mode,
                    "source_lang": args.source_lang,
                    "target_lang": args.target_lang,
                    "selected_paths": [str(path) for path in selected_paths],
                },
            )
            write_json(run_dir / "preset_resolved.json", preset)
            write_snapshot_json(
                run_dir / "runtime_snapshot.json",
                collect_runtime_snapshot(DEFAULT_CONTAINER_NAMES),
            )

            if args.runtime_mode == "managed":
                _ensure_managed_runtime(run_dir, preset)
                write_snapshot_json(
                    run_dir / "docker_snapshot.json",
                    collect_runtime_snapshot(DEFAULT_CONTAINER_NAMES),
                )
            else:
                write_snapshot_json(
                    run_dir / "docker_snapshot.json",
                    collect_runtime_snapshot(DEFAULT_CONTAINER_NAMES),
                )

            try:
                summary = _run_single_mode(
                    app=app,
                    preset=preset,
                    mode=mode,
                    run_dir=run_dir,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    image_paths=selected_paths,
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            except Exception:
                traceback.print_exc()
                return 1
            aggregated.append(
                {
                    "run_dir": str(run_dir),
                    "mode": mode,
                    "repeat_index": repeat_index,
                    "summary": summary,
                }
            )
            print(f"[done] {mode} repeat={repeat_index} -> {run_dir}")

    if aggregated:
        write_json(Path(aggregated[-1]["run_dir"]).parent / "last_benchmark_runs.json", {"runs": aggregated})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
