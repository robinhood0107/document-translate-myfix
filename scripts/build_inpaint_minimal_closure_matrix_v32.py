#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-minimal-closure-matrix-v32"
CATEGORY = "40-inpaint-mask-render"
ROLE_AXES = ("detector", "ownership", "silhouette", "router", "expansion")


def _canonical(row: Mapping[str, object]) -> str:
    return json.dumps(dict(row), sort_keys=True, separators=(",", ":"))


def build_minimal_closure_matrix(
    matrix: dict[str, object],
    *,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    if matrix.get("schema_version") != "inpaint-factorized-matrix-v3":
        raise ValueError("minimal closure requires factorized matrix v3")
    axes = matrix.get("axes")
    controls = matrix.get("controls")
    rows = matrix.get("explicit_combinations")
    oracle = matrix.get("oracle_only", [])
    if not isinstance(axes, dict) or not isinstance(controls, dict):
        raise ValueError("minimal closure matrix lacks axes or controls")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("minimal closure matrix lacks explicit combinations")
    if not isinstance(oracle, list) or any(not isinstance(value, str) for value in oracle):
        raise ValueError("minimal closure oracle list is invalid")
    oracle_ids = frozenset(oracle)
    target: set[tuple[str, str]] = set()
    for role in ROLE_AXES:
        values = axes.get(role)
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError(f"minimal closure axis is invalid: {role}")
        target.update((role, value) for value in values if value not in oracle_ids)

    candidates = sorted((dict(row) for row in rows), key=_canonical)
    selected: list[dict[str, object]] = []
    remaining = set(target)
    control_row = {
        role: str(controls[role])
        for role in ("detector", "expansion", "fill", "ownership", "router", "silhouette")
    }
    selected.append(control_row)
    remaining -= {(role, str(control_row[role])) for role in ROLE_AXES}
    while remaining:
        ranked = sorted(
            (
                (
                    -len(
                        {
                            (role, str(row[role]))
                            for role in ROLE_AXES
                        }
                        & remaining
                    ),
                    _canonical(row),
                    row,
                )
                for row in candidates
            ),
            key=lambda item: (item[0], item[1]),
        )
        if not ranked or ranked[0][0] == 0:
            missing = ", ".join(f"{role}={value}" for role, value in sorted(remaining))
            raise ValueError(f"explicit combinations cannot cover registered axes: {missing}")
        row = dict(ranked[0][2])
        if row not in selected:
            selected.append(row)
        remaining -= {(role, str(row[role])) for role in ROLE_AXES}
        candidates = [candidate for candidate in candidates if candidate != row]

    output = dict(matrix)
    output["factorized"] = False
    output["explicit_combinations"] = selected
    output["closure_reduction"] = {
        "schema_version": "inpaint-minimal-role-coverage-v1",
        "covered_roles": list(ROLE_AXES),
        "source_combination_count": len(rows),
        "selected_combination_count": len(selected),
        "oracle_variants_excluded": sorted(oracle_ids),
        "coverage_complete": True,
    }
    if manifest_path is not None:
        output["manifest"] = str(manifest_path.resolve())
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reduce a factorized matrix to deterministic role-coverage runs."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-name", default="minimal-closure-matrix-v32.json")
    args = parser.parse_args(argv)
    source = json.loads(args.matrix.read_text(encoding="utf-8"))
    if not isinstance(source, dict):
        raise ValueError("matrix root must be an object")
    payload = build_minimal_closure_matrix(
        source,
        manifest_path=args.manifest.resolve() if args.manifest is not None else None,
    )
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / str(args.output_name)
    if output.exists():
        raise FileExistsError("minimal closure matrix output must be fresh")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if managed is not None:
        managed.complete(metadata=payload["closure_reduction"])
        mismatches = managed.verify()
        if mismatches:
            raise RuntimeError(
                "managed artifact verification failed: " + "; ".join(mismatches)
            )
        print(managed.run_root)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
