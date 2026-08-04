from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.settings_page import (  # noqa: E402
    GEMMA_SAMPLER_MIGRATION_VERSION,
    GEMMA_SAMPLER_MIGRATION_VERSION_KEY,
    migrate_gemma_sampler_defaults,
)
from app.ui.settings.gemma_local_server_page import GemmaLocalServerPage  # noqa: E402
from modules.translation.llm.custom_local_gemma import (  # noqa: E402
    DEFAULT_GEMMA_TRANSLATION_MIN_P,
    DEFAULT_GEMMA_TRANSLATION_TEMPERATURE,
    DEFAULT_GEMMA_TRANSLATION_TOP_K,
    DEFAULT_GEMMA_TRANSLATION_TOP_P,
)

TEMPERATURE_KEY = "gemma_local_server/temperature"
TOP_K_KEY = "gemma_local_server/top_k"
TOP_P_KEY = "gemma_local_server/top_p"
MIN_P_KEY = "gemma_local_server/min_p"
RETIRED = {TEMPERATURE_KEY: 0.7, TOP_K_KEY: 64, TOP_P_KEY: 0.95, MIN_P_KEY: 0.0}


class _FakeQSettings:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, object]] = []
        self.sync_count = 0

    def value(self, key: str, default=None, type=None):
        value = self.values.get(key, default)
        if type is None or value is None:
            return value
        return type(value)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value
        self.writes.append((key, value))

    def sync(self) -> None:
        self.sync_count += 1


class GemmaSamplerDefaultsMigrationTests(unittest.TestCase):
    def _assert_promoted(self, settings: _FakeQSettings) -> None:
        self.assertAlmostEqual(
            float(settings.values[TEMPERATURE_KEY]), DEFAULT_GEMMA_TRANSLATION_TEMPERATURE
        )
        self.assertEqual(int(settings.values[TOP_K_KEY]), DEFAULT_GEMMA_TRANSLATION_TOP_K)
        self.assertAlmostEqual(float(settings.values[TOP_P_KEY]), DEFAULT_GEMMA_TRANSLATION_TOP_P)
        self.assertAlmostEqual(float(settings.values[MIN_P_KEY]), DEFAULT_GEMMA_TRANSLATION_MIN_P)
        self.assertEqual(
            settings.values[GEMMA_SAMPLER_MIGRATION_VERSION_KEY],
            GEMMA_SAMPLER_MIGRATION_VERSION,
        )

    def test_promoted_tuple_matches_approved_campaign_winner(self) -> None:
        self.assertAlmostEqual(DEFAULT_GEMMA_TRANSLATION_TEMPERATURE, 0.5)
        self.assertEqual(DEFAULT_GEMMA_TRANSLATION_TOP_K, 32)
        self.assertAlmostEqual(DEFAULT_GEMMA_TRANSLATION_TOP_P, 1.0)
        self.assertAlmostEqual(DEFAULT_GEMMA_TRANSLATION_MIN_P, 0.0)

    def test_settings_page_defaults_track_the_product_constants(self) -> None:
        self.assertAlmostEqual(
            GemmaLocalServerPage.DEFAULT_TEMPERATURE, DEFAULT_GEMMA_TRANSLATION_TEMPERATURE
        )
        self.assertEqual(GemmaLocalServerPage.DEFAULT_TOP_K, DEFAULT_GEMMA_TRANSLATION_TOP_K)
        self.assertAlmostEqual(GemmaLocalServerPage.DEFAULT_TOP_P, DEFAULT_GEMMA_TRANSLATION_TOP_P)
        self.assertAlmostEqual(GemmaLocalServerPage.DEFAULT_MIN_P, DEFAULT_GEMMA_TRANSLATION_MIN_P)

    def test_fresh_install_records_marker_without_reporting_a_change(self) -> None:
        settings = _FakeQSettings({})
        self.assertFalse(migrate_gemma_sampler_defaults(settings))
        self._assert_promoted(settings)
        self.assertEqual(settings.sync_count, 1)

    def test_retired_values_are_overwritten_once(self) -> None:
        settings = _FakeQSettings(dict(RETIRED))
        self.assertTrue(migrate_gemma_sampler_defaults(settings))
        self._assert_promoted(settings)

    def test_version_one_install_is_migrated_again(self) -> None:
        settings = _FakeQSettings({**RETIRED, GEMMA_SAMPLER_MIGRATION_VERSION_KEY: 1})
        self.assertTrue(migrate_gemma_sampler_defaults(settings))
        self._assert_promoted(settings)

    def test_arbitrary_user_values_are_overwritten_once(self) -> None:
        settings = _FakeQSettings(
            {TEMPERATURE_KEY: 1.4, TOP_K_KEY: 500, TOP_P_KEY: 0.62, MIN_P_KEY: 0.09}
        )
        self.assertTrue(migrate_gemma_sampler_defaults(settings))
        self._assert_promoted(settings)

    def test_malformed_marker_is_treated_as_unmigrated(self) -> None:
        settings = _FakeQSettings(
            {**RETIRED, GEMMA_SAMPLER_MIGRATION_VERSION_KEY: "not-a-version"}
        )
        self.assertTrue(migrate_gemma_sampler_defaults(settings))
        self._assert_promoted(settings)

    def test_later_user_change_is_preserved(self) -> None:
        settings = _FakeQSettings(dict(RETIRED))
        self.assertTrue(migrate_gemma_sampler_defaults(settings))
        settings.setValue(TEMPERATURE_KEY, 0.9)
        settings.setValue(TOP_K_KEY, 256)
        self.assertFalse(migrate_gemma_sampler_defaults(settings))
        self.assertAlmostEqual(float(settings.values[TEMPERATURE_KEY]), 0.9)
        self.assertEqual(int(settings.values[TOP_K_KEY]), 256)

    def test_repeated_calls_do_not_rewrite_sampler_keys(self) -> None:
        settings = _FakeQSettings(dict(RETIRED))
        migrate_gemma_sampler_defaults(settings)
        settings.writes.clear()
        self.assertFalse(migrate_gemma_sampler_defaults(settings))
        self.assertEqual(settings.writes, [])

    def test_unrelated_settings_are_untouched(self) -> None:
        settings = _FakeQSettings(
            {**RETIRED, "gemma_local_server/chunk_size": 6, "language": "Korean"}
        )
        migrate_gemma_sampler_defaults(settings)
        self.assertEqual(settings.values["gemma_local_server/chunk_size"], 6)
        self.assertEqual(settings.values["language"], "Korean")


if __name__ == "__main__":
    unittest.main()
