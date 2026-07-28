#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LAUNCHERS = (
    {
        "label": "cuda12",
        "venv_dir": ".venv-win",
        "bat": "run_comic.bat",
        "expected_versions": {
            "torch": "2.11.0+cu128",
            "torchvision": "0.26.0+cu128",
            "setuptools": "80.9.0",
            "einops": "0.8.2",
        },
        "required_any": ("onnxruntime-gpu", "PySide6"),
        "expected_cuda": "12.8",
    },
    {
        "label": "cuda13",
        "venv_dir": ".venv-win-cuda13",
        "bat": "run_comic_cuda13.bat",
        "expected_versions": {
            "torch": "2.11.0+cu130",
            "torchvision": "0.26.0+cu130",
            "onnxruntime-gpu": "1.26.0.dev20260408003",
            "setuptools": "80.9.0",
            "einops": "0.8.2",
        },
        "required_any": ("PySide6",),
        "expected_cuda": "13.0",
    },
)

def run_checked(
    cmd: list[str],
    *,
    timeout: int = 900,
    shell: bool = False,
    label: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"\n=== {label} ===")
    print(" ".join(cmd) if not shell else cmd[0])
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        shell=shell,
        env=env,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return result


def verify_runtime(cfg: dict) -> None:
    python_exe = ROOT / cfg["venv_dir"] / "Scripts" / "python.exe"
    if not python_exe.is_file():
        raise RuntimeError(f"Missing interpreter: {python_exe}")

    probe = f"""
import importlib.metadata as md
import json
import torch

payload = {{}}
for name in {tuple(cfg["expected_versions"].keys()) + tuple(cfg["required_any"])}:
    payload[name] = md.version(name)
payload["cuda"] = getattr(torch.version, "cuda", None)
print(json.dumps(payload, ensure_ascii=False))
"""
    result = run_checked(
        [str(python_exe), "-c", probe],
        label=f"{cfg['label']} runtime probe",
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])

    for package_name, expected in cfg["expected_versions"].items():
        actual = payload.get(package_name)
        if actual != expected:
            raise RuntimeError(
                f"{cfg['label']}: expected {package_name}={expected}, got {actual}"
            )

    for package_name in cfg["required_any"]:
        if not payload.get(package_name):
            raise RuntimeError(f"{cfg['label']}: missing required package {package_name}")

    if payload.get("cuda") != cfg["expected_cuda"]:
        raise RuntimeError(
            f"{cfg['label']}: expected torch CUDA {cfg['expected_cuda']}, got {payload.get('cuda')}"
        )


def verify_bootstrap(cfg: dict) -> None:
    command = f'set COMIC_BOOTSTRAP_ONLY=1 && call {cfg["bat"]}'
    run_checked(
        ["cmd.exe", "/c", command],
        label=f"{cfg['label']} bootstrap launcher",
    )


def verify_source_contract(cfg: dict) -> None:
    env = os.environ.copy()
    env["COMIC_VERIFY_ONLY"] = "1"
    run_checked(
        ["cmd.exe", "/d", "/c", f"call {cfg['bat']}"],
        label=f"{cfg['label']} launcher-source contract",
        env=env,
    )


def verify_smoke(cfg: dict) -> None:
    command = (
        "set CT_DISABLE_UPDATE_CHECK=1 && "
        "set COMIC_SKIP_STARTUP_MODELS=1 && "
        "set COMIC_SMOKE_EXIT_MS=1500 && "
        f"call {cfg['bat']}"
    )
    run_checked(
        ["cmd.exe", "/c", command],
        timeout=900,
        label=f"{cfg['label']} startup smoke",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Windows launchers and pinned runtimes.")
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Verify venvs and bootstrap launchers only.",
    )
    parser.add_argument(
        "--source-contract-only",
        action="store_true",
        help="Verify release source files without probing or installing a venv.",
    )
    args = parser.parse_args()

    for cfg in LAUNCHERS:
        verify_source_contract(cfg)
        if args.source_contract_only:
            continue
        verify_runtime(cfg)
        verify_bootstrap(cfg)
        if not args.skip_smoke:
            verify_smoke(cfg)

    print("\nWindows launcher verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
