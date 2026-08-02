from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.settings_page import (  # noqa: E402
    GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION,
    GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION_KEY,
    GEMMA_SAMPLER_STABILITY_VALUES,
    migrate_gemma_sampler_stability,
)


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


class GemmaSamplerStabilityMigrationTests(unittest.TestCase):
    def test_first_migration_unconditionally_overwrites_saved_sampler_values(self) -> None:
        settings = _FakeQSettings(
            {
                "gemma_local_server/temperature": 0.7,
                "gemma_local_server/top_p": 0.95,
                "gemma_local_server/top_k": 64,
                "gemma_local_server/min_p": 0.05,
                "gemma_local_server/chunk_size": 6,
                "credentials/Custom Local Server(Gemma)_model": "custom.gguf",
            }
        )

        self.assertTrue(migrate_gemma_sampler_stability(settings))
        for key, expected in GEMMA_SAMPLER_STABILITY_VALUES.items():
            self.assertEqual(settings.values[f"gemma_local_server/{key}"], expected)
        self.assertEqual(
            settings.values[GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION_KEY],
            GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION,
        )
        self.assertEqual(settings.values["gemma_local_server/chunk_size"], 6)
        self.assertEqual(
            settings.values["credentials/Custom Local Server(Gemma)_model"],
            "custom.gguf",
        )
        self.assertEqual(settings.sync_count, 1)

    def test_completed_migration_preserves_later_user_edits(self) -> None:
        settings = _FakeQSettings(
            {
                GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION_KEY: (
                    GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION
                ),
                "gemma_local_server/temperature": 0.35,
                "gemma_local_server/top_p": 0.82,
                "gemma_local_server/top_k": 32,
                "gemma_local_server/min_p": 0.1,
            }
        )

        self.assertFalse(migrate_gemma_sampler_stability(settings))
        self.assertEqual(settings.values["gemma_local_server/temperature"], 0.35)
        self.assertEqual(settings.values["gemma_local_server/top_p"], 0.82)
        self.assertEqual(settings.values["gemma_local_server/top_k"], 32)
        self.assertEqual(settings.values["gemma_local_server/min_p"], 0.1)
        self.assertEqual(settings.writes, [])
        self.assertEqual(settings.sync_count, 0)

    def test_fresh_install_records_version_and_tuple(self) -> None:
        settings = _FakeQSettings({})

        self.assertTrue(migrate_gemma_sampler_stability(settings))
        self.assertEqual(
            settings.values[GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION_KEY],
            GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION,
        )
        for key, expected in GEMMA_SAMPLER_STABILITY_VALUES.items():
            self.assertEqual(settings.values[f"gemma_local_server/{key}"], expected)

    def test_malformed_version_is_treated_as_not_migrated(self) -> None:
        settings = _FakeQSettings(
            {
                GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION_KEY: "invalid",
                "gemma_local_server/temperature": 0.7,
            }
        )

        self.assertTrue(migrate_gemma_sampler_stability(settings))
        self.assertEqual(
            settings.values[GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION_KEY],
            GEMMA_SAMPLER_STABILITY_MIGRATION_VERSION,
        )
        self.assertEqual(
            settings.values["gemma_local_server/temperature"],
            GEMMA_SAMPLER_STABILITY_VALUES["temperature"],
        )


if __name__ == "__main__":
    unittest.main()
