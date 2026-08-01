from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.settings_page import (  # noqa: E402
    MANGALMM_FULL_PAGE_CONTRACT_VERSION,
    MANGALMM_FULL_PAGE_CONTRACT_VERSION_KEY,
    MANGALMM_MAX_COMPLETION_TOKENS_KEY,
    migrate_mangalmm_full_page_contract,
)
from app.ui.settings.mangalmm_ocr_page import MangaLMMOCRPage  # noqa: E402


class _FakeQSettings:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, object]] = []
        self.sync_count = 0

    def value(self, key: str, default=None, type=None):
        value = self.values.get(key, default)
        if type is None:
            return value
        return type(value)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value
        self.writes.append((key, value))

    def sync(self) -> None:
        self.sync_count += 1


class MangaLMMFullPageMigrationTests(unittest.TestCase):
    def test_legacy_default_is_migrated_once(self) -> None:
        settings = _FakeQSettings(
            {
                MANGALMM_MAX_COMPLETION_TOKENS_KEY: 256,
                "mangalmm_ocr/server_url": "http://example.test/v1",
                "mangalmm_ocr/request_timeout_sec": 75,
                "mangalmm_ocr/raw_response_logging": True,
            }
        )

        changed = migrate_mangalmm_full_page_contract(settings)

        self.assertTrue(changed)
        self.assertEqual(
            settings.values[MANGALMM_MAX_COMPLETION_TOKENS_KEY],
            MangaLMMOCRPage.DEFAULT_MAX_COMPLETION_TOKENS,
        )
        self.assertEqual(
            settings.values[MANGALMM_FULL_PAGE_CONTRACT_VERSION_KEY],
            MANGALMM_FULL_PAGE_CONTRACT_VERSION,
        )
        self.assertEqual(
            settings.values["mangalmm_ocr/server_url"],
            "http://example.test/v1",
        )
        self.assertEqual(
            settings.values["mangalmm_ocr/request_timeout_sec"],
            75,
        )
        self.assertTrue(settings.values["mangalmm_ocr/raw_response_logging"])
        self.assertEqual(settings.sync_count, 1)

    def test_custom_token_value_is_preserved(self) -> None:
        settings = _FakeQSettings(
            {MANGALMM_MAX_COMPLETION_TOKENS_KEY: 320}
        )

        changed = migrate_mangalmm_full_page_contract(settings)

        self.assertFalse(changed)
        self.assertEqual(
            settings.values[MANGALMM_MAX_COMPLETION_TOKENS_KEY],
            320,
        )
        self.assertEqual(
            {key for key, _value in settings.writes},
            {MANGALMM_FULL_PAGE_CONTRACT_VERSION_KEY},
        )

    def test_completed_migration_preserves_later_manual_value(self) -> None:
        settings = _FakeQSettings(
            {
                MANGALMM_FULL_PAGE_CONTRACT_VERSION_KEY: (
                    MANGALMM_FULL_PAGE_CONTRACT_VERSION
                ),
                MANGALMM_MAX_COMPLETION_TOKENS_KEY: 512,
            }
        )

        changed = migrate_mangalmm_full_page_contract(settings)

        self.assertFalse(changed)
        self.assertEqual(
            settings.values[MANGALMM_MAX_COMPLETION_TOKENS_KEY],
            512,
        )
        self.assertEqual(settings.writes, [])
        self.assertEqual(settings.sync_count, 0)

    def test_new_install_uses_new_default_without_persisting_override(
        self,
    ) -> None:
        settings = _FakeQSettings({})

        changed = migrate_mangalmm_full_page_contract(settings)

        self.assertFalse(changed)
        self.assertNotIn(
            MANGALMM_MAX_COMPLETION_TOKENS_KEY,
            settings.values,
        )
        self.assertEqual(
            MangaLMMOCRPage.DEFAULT_MAX_COMPLETION_TOKENS,
            4096,
        )
        self.assertEqual(
            settings.values[MANGALMM_FULL_PAGE_CONTRACT_VERSION_KEY],
            MANGALMM_FULL_PAGE_CONTRACT_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
