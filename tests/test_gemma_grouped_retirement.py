from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.settings_page import (  # noqa: E402
    GEMMA_GROUPED_RETIREMENT_VERSION,
    GEMMA_GROUPED_RETIREMENT_VERSION_KEY,
    GEMMA_REQUEST_MODE_KEY,
    migrate_retired_gemma_request_mode,
)
from modules.translation.llm.custom_local_gemma import (  # noqa: E402
    GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
    RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED,
)


class _FakeQSettings:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, object]] = []
        self.syncs = 0

    def value(self, key: str, default=None, type=None):
        value = self.values.get(key, default)
        if type is None:
            return value
        return type(value)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value
        self.writes.append((key, value))

    def sync(self) -> None:
        # 마커를 쓰고 sync 하지 않으면 크래시 시 마이그레이션이 다시 돈다.
        self.syncs += 1


class GemmaGroupedRetirementMigrationTests(unittest.TestCase):
    def test_grouped_value_is_migrated_once_without_touching_other_settings(
        self,
    ) -> None:
        untouched = {
            "credentials/Custom Local Server(Gemma)_api_url": (
                "http://127.0.0.1:18080/v1"
            ),
            "credentials/Custom Local Server(Gemma)_model": "custom.gguf",
            "credentials/Custom Local Server(Gemma)_api_key": "credential-value",
            "gemma_local_server/chunk_size": 9,
            "gemma_local_server/max_completion_tokens": 768,
            "gemma_local_server/request_timeout_sec": 240,
            "gemma_local_server/temperature": 0.55,
            "gemma_local_server/top_k": 32,
            "gemma_local_server/top_p": 0.9,
            "gemma_local_server/min_p": 0.05,
            "gemma_local_server/raw_response_logging": True,
        }
        settings = _FakeQSettings(
            {
                **untouched,
                GEMMA_REQUEST_MODE_KEY: (
                    RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED
                ),
            }
        )

        changed = migrate_retired_gemma_request_mode(settings)

        self.assertTrue(changed)
        self.assertEqual(
            settings.values[GEMMA_REQUEST_MODE_KEY],
            GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
        )
        self.assertEqual(
            settings.values[GEMMA_GROUPED_RETIREMENT_VERSION_KEY],
            GEMMA_GROUPED_RETIREMENT_VERSION,
        )
        self.assertEqual(
            {
                key: settings.values[key]
                for key in untouched
            },
            untouched,
        )
        self.assertEqual(
            {key for key, _value in settings.writes},
            {
                GEMMA_REQUEST_MODE_KEY,
                GEMMA_GROUPED_RETIREMENT_VERSION_KEY,
            },
        )

    def test_completed_migration_never_overwrites_later_manual_value(
        self,
    ) -> None:
        settings = _FakeQSettings(
            {
                GEMMA_GROUPED_RETIREMENT_VERSION_KEY: (
                    GEMMA_GROUPED_RETIREMENT_VERSION
                ),
                GEMMA_REQUEST_MODE_KEY: (
                    RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED
                ),
            }
        )

        changed = migrate_retired_gemma_request_mode(settings)

        self.assertFalse(changed)
        self.assertEqual(
            settings.values[GEMMA_REQUEST_MODE_KEY],
            RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED,
        )
        self.assertEqual(settings.writes, [])

    def test_new_install_records_migration_without_creating_mode_override(
        self,
    ) -> None:
        settings = _FakeQSettings({})

        changed = migrate_retired_gemma_request_mode(settings)

        self.assertFalse(changed)
        self.assertNotIn(GEMMA_REQUEST_MODE_KEY, settings.values)
        self.assertEqual(
            settings.values[GEMMA_GROUPED_RETIREMENT_VERSION_KEY],
            GEMMA_GROUPED_RETIREMENT_VERSION,
        )

    def test_malformed_migration_version_is_treated_as_not_migrated(
        self,
    ) -> None:
        settings = _FakeQSettings(
            {
                GEMMA_GROUPED_RETIREMENT_VERSION_KEY: "invalid",
                GEMMA_REQUEST_MODE_KEY: (
                    RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED
                ),
            }
        )

        changed = migrate_retired_gemma_request_mode(settings)

        self.assertTrue(changed)
        self.assertEqual(
            settings.values[GEMMA_REQUEST_MODE_KEY],
            GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
        )
        self.assertEqual(
            settings.values[GEMMA_GROUPED_RETIREMENT_VERSION_KEY],
            GEMMA_GROUPED_RETIREMENT_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
