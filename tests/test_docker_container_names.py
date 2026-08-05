"""제품이 만드는 컨테이너는 모두 이름이 정해져 있어야 한다.

`docker run`에 `--name`을 주지 않으면 Docker가 `strange_brown` 같은 임의 이름을
붙인다. 사용자에게는 정체를 알 수 없는 컨테이너가 떴다 사라지는 것으로 보이고,
`docker ps`로 제품 소유를 확인할 수도 없다. 소스에서 이 계약을 고정한다.
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("modules", "pipeline", "app", "controller.py")


def _string_items(node: ast.AST) -> list[str] | None:
    """리스트 리터럴에서 문자열 원소만 순서대로 뽑는다."""

    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    items: list[str] = []
    for element in node.elts:
        if isinstance(element, ast.Constant) and isinstance(element.value, str):
            items.append(element.value)
        else:
            items.append("")
    return items


def _python_files() -> list[Path]:
    files: list[Path] = []
    for entry in SEARCH_ROOTS:
        target = REPO_ROOT / entry
        if target.is_file():
            files.append(target)
        elif target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
    return files


class DockerContainerNameTests(unittest.TestCase):
    def test_every_docker_run_passes_an_explicit_name(self) -> None:
        offenders: list[str] = []
        for path in _python_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                items = _string_items(node)
                if not items or len(items) < 2:
                    continue
                if items[0] != "docker" or items[1] != "run":
                    continue
                if "--name" not in items:
                    offenders.append(
                        f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}"
                    )
        self.assertEqual(
            offenders,
            [],
            "이름 없이 docker run 을 호출하는 곳이 있습니다: " + ", ".join(offenders),
        )

    def test_named_runs_use_the_product_prefix(self) -> None:
        """이름은 제품 소유를 알 수 있어야 한다."""

        names: list[str] = []
        for path in _python_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                items = _string_items(node)
                if not items or len(items) < 2:
                    continue
                if items[0] != "docker" or items[1] != "run":
                    continue
                if "--name" not in items:
                    continue
                name = items[items.index("--name") + 1]
                if name:
                    names.append(name)
        self.assertTrue(names, "이름을 지정한 docker run 을 찾지 못했습니다.")
        for name in names:
            with self.subTest(name=name):
                self.assertTrue(
                    name.startswith("comic-translate-"),
                    f"제품 접두어가 없습니다: {name}",
                )

    def test_probe_names_are_unique_per_purpose(self) -> None:
        """같은 이름을 두 용도가 쓰면 동시 실행에서 서로를 지운다."""

        names: list[str] = []
        for path in _python_files():
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
                continue
            for node in ast.walk(tree):
                items = _string_items(node)
                if not items or len(items) < 2:
                    continue
                if items[0] != "docker" or items[1] != "run":
                    continue
                if "--name" in items:
                    names.append(items[items.index("--name") + 1])
        self.assertEqual(sorted(names), sorted(set(names)))


if __name__ == "__main__":
    unittest.main()
