from __future__ import annotations

import os
import unittest
from unittest import mock

from comic import _read_positive_int_env


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


if __name__ == "__main__":
    unittest.main()
