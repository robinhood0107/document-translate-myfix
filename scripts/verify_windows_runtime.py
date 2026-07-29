#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import re
import sys
from pathlib import Path


PINNED_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$"
)


def normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_pinned_requirements(
    requirement_files: list[Path],
) -> dict[str, tuple[str, str]]:
    pinned: dict[str, tuple[str, str]] = {}
    visited: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        if not resolved.is_file():
            raise ValueError(f"Requirements file does not exist: {path}")

        for line_number, raw_line in enumerate(
            resolved.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r ", "--requirement ")):
                included = line.split(maxsplit=1)[1].strip()
                visit(resolved.parent / included)
                continue
            if line.startswith("-"):
                continue

            match = PINNED_REQUIREMENT_RE.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"{resolved}:{line_number} must use an exact NAME==VERSION pin"
                )
            display_name = match.group("name")
            expected_version = match.group("version")
            normalized = normalize_distribution_name(display_name)
            existing = pinned.get(normalized)
            if existing is not None and existing[1] != expected_version:
                raise ValueError(
                    f"Conflicting requirement pins for {display_name}: "
                    f"{existing[1]} and {expected_version}"
                )
            pinned[normalized] = (display_name, expected_version)

    for requirement_file in requirement_files:
        visit(requirement_file)
    return pinned


def verify_installed_requirements(
    pinned: dict[str, tuple[str, str]],
) -> list[str]:
    errors: list[str] = []
    for normalized in sorted(pinned):
        display_name, expected_version = pinned[normalized]
        try:
            actual_version = importlib.metadata.version(display_name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{display_name} is not installed")
            continue
        if actual_version != expected_version:
            errors.append(
                f"{display_name} expected {expected_version}, found {actual_version}"
            )
    return errors


def verify_cuda_version(expected_cuda: str) -> list[str]:
    try:
        import torch
    except Exception as exc:
        return [f"torch import failed: {exc}"]
    actual_cuda = str(getattr(torch.version, "cuda", "") or "")
    if actual_cuda != expected_cuda:
        return [f"CUDA runtime expected {expected_cuda}, found {actual_cuda or 'none'}"]
    return []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify every pinned direct Windows runtime requirement."
    )
    parser.add_argument(
        "--requirements",
        type=Path,
        action="append",
        required=True,
        help="Requirements file to verify. Nested -r files are followed.",
    )
    parser.add_argument("--expected-cuda", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        pinned = load_pinned_requirements(args.requirements)
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"[runtime] {exc}", file=sys.stderr)
        return 2

    errors = verify_installed_requirements(pinned)
    errors.extend(verify_cuda_version(args.expected_cuda))
    if errors:
        for error in errors:
            print(f"[runtime] {error}", file=sys.stderr)
        return 1

    print(
        f"[runtime] Verified {len(pinned)} pinned packages "
        f"with CUDA {args.expected_cuda}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
