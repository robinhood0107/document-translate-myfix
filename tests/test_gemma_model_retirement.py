from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.gemma_local_server_page import (  # noqa: E402
    GemmaLocalServerPage,
)
from app.ui.settings.settings_page import (  # noqa: E402
    GEMMA_MODEL_RETIREMENT_VERSION,
    GEMMA_MODEL_RETIREMENT_VERSION_KEY,
    GEMMA_RETIRED_IQ4_XS_MODEL,
    migrate_retired_gemma_model,
)


_MODEL_KEY = "credentials/Custom Local Server(Gemma)_model"
_ENDPOINT_KEY = "credentials/Custom Local Server(Gemma)_api_url"


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


class GemmaModelRetirementMigrationTests(unittest.TestCase):
    def test_managed_iq4_xs_is_migrated_without_touching_other_settings(
        self,
    ) -> None:
        untouched = {
            _ENDPOINT_KEY: GemmaLocalServerPage.DEFAULT_ENDPOINT_URL,
            "credentials/Custom Local Server(Gemma)_api_key": "secret-value",
            "gemma_local_server/chunk_size": 9,
            "gemma_local_server/max_completion_tokens": 768,
            "gemma_local_server/temperature": 0.55,
        }
        settings = _FakeQSettings(
            {
                **untouched,
                _MODEL_KEY: GEMMA_RETIRED_IQ4_XS_MODEL,
            }
        )

        changed = migrate_retired_gemma_model(settings)

        self.assertTrue(changed)
        self.assertEqual(
            settings.values[_MODEL_KEY],
            GemmaLocalServerPage.DEFAULT_MODEL,
        )
        self.assertEqual(
            settings.values[GEMMA_MODEL_RETIREMENT_VERSION_KEY],
            GEMMA_MODEL_RETIREMENT_VERSION,
        )
        self.assertEqual(
            {key: settings.values[key] for key in untouched},
            untouched,
        )
        self.assertEqual(settings.sync_count, 1)

    def test_custom_endpoint_preserves_same_model_name(self) -> None:
        settings = _FakeQSettings(
            {
                _ENDPOINT_KEY: "http://example.test/v1",
                _MODEL_KEY: GEMMA_RETIRED_IQ4_XS_MODEL,
            }
        )

        changed = migrate_retired_gemma_model(settings)

        self.assertFalse(changed)
        self.assertEqual(
            settings.values[_MODEL_KEY],
            GEMMA_RETIRED_IQ4_XS_MODEL,
        )
        self.assertEqual(settings.sync_count, 1)

    def test_completed_migration_preserves_later_manual_value(self) -> None:
        settings = _FakeQSettings(
            {
                GEMMA_MODEL_RETIREMENT_VERSION_KEY: (
                    GEMMA_MODEL_RETIREMENT_VERSION
                ),
                _MODEL_KEY: "custom-managed-model.gguf",
            }
        )

        self.assertFalse(migrate_retired_gemma_model(settings))
        self.assertEqual(
            settings.values[_MODEL_KEY],
            "custom-managed-model.gguf",
        )
        self.assertEqual(settings.writes, [])
        self.assertEqual(settings.sync_count, 0)

    def test_new_install_records_version_without_model_override(self) -> None:
        settings = _FakeQSettings({})

        self.assertFalse(migrate_retired_gemma_model(settings))
        self.assertNotIn(_MODEL_KEY, settings.values)
        self.assertEqual(
            settings.values[GEMMA_MODEL_RETIREMENT_VERSION_KEY],
            GEMMA_MODEL_RETIREMENT_VERSION,
        )
        self.assertEqual(settings.sync_count, 1)


if __name__ == "__main__":
    unittest.main()
