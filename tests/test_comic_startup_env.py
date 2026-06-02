from __future__ import annotations

import os
import unittest
from unittest import mock

from comic import (
    _read_bool_env,
    _read_positive_int_env,
    _should_skip_startup_runtime_models,
)


class ComicStartupEnvTests(unittest.TestCase):
    def test_read_positive_int_env_returns_default_for_missing_or_invalid_values(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_read_positive_int_env("COMIC_SMOKE_EXIT_MS", 0), 0)

        with mock.patch.dict(os.environ, {"COMIC_SMOKE_EXIT_MS": "abc"}, clear=True):
            self.assertEqual(_read_positive_int_env("COMIC_SMOKE_EXIT_MS", 25), 25)

        with mock.patch.dict(os.environ, {"COMIC_SMOKE_EXIT_MS": "-5"}, clear=True):
            self.assertEqual(_read_positive_int_env("COMIC_SMOKE_EXIT_MS", 25), 25)

    def test_read_positive_int_env_accepts_positive_integer(self) -> None:
        with mock.patch.dict(os.environ, {"COMIC_SMOKE_EXIT_MS": "1500"}, clear=True):
            self.assertEqual(_read_positive_int_env("COMIC_SMOKE_EXIT_MS", 0), 1500)

    def test_read_bool_env_accepts_common_truthy_values(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with mock.patch.dict(os.environ, {"COMIC_SKIP_STARTUP_MODELS": value}, clear=True):
                self.assertTrue(_read_bool_env("COMIC_SKIP_STARTUP_MODELS"))

    def test_startup_model_skip_is_enabled_for_smoke_or_env(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_should_skip_startup_runtime_models(0))
            self.assertTrue(_should_skip_startup_runtime_models(1500))

        with mock.patch.dict(os.environ, {"COMIC_SKIP_STARTUP_MODELS": "1"}, clear=True):
            self.assertTrue(_should_skip_startup_runtime_models(0))


if __name__ == "__main__":
    unittest.main()
