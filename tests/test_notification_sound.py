from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.utils import notification_sound


class NotificationSoundTests(unittest.TestCase):
    def test_list_music_wav_files_filters_and_sorts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "b.wav").write_bytes(b"")
            (root / "a.wav").write_bytes(b"")
            (root / "ignore.txt").write_text("x", encoding="utf-8")
            with mock.patch("modules.utils.notification_sound.get_music_dir", return_value=root):
                self.assertEqual(notification_sound.list_music_wav_files(), ["a.wav", "b.wav"])

    def test_play_completion_sound_uses_custom_file_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wav_path = root / "notify.wav"
            wav_path.write_bytes(b"RIFF")
            with (
                mock.patch("modules.utils.notification_sound.resolve_music_wav_path", return_value=wav_path),
                mock.patch("modules.utils.notification_sound._play_wav_file", return_value=True) as play_wav,
                mock.patch("modules.utils.notification_sound._play_system_sound", return_value=False),
            ):
                self.assertTrue(notification_sound.play_completion_sound("file", "notify.wav"))
                play_wav.assert_called_once_with(wav_path)

    def test_play_completion_sound_falls_back_to_system(self) -> None:
        with (
            mock.patch("modules.utils.notification_sound.resolve_music_wav_path", return_value=None),
            mock.patch("modules.utils.notification_sound._play_system_sound", return_value=True) as play_system,
        ):
            self.assertTrue(notification_sound.play_completion_sound("file", "missing.wav"))
            play_system.assert_called_once()

    def test_normalize_ntfy_settings_applies_defaults(self) -> None:
        normalized = notification_sound.normalize_ntfy_settings(
            {
                "enable_ntfy_notifications": True,
                "ntfy_server_url": "https://ntfy.example.com////",
                "ntfy_timeout_sec": 999,
            }
        )

        self.assertTrue(normalized["enable_ntfy_notifications"])
        self.assertEqual(normalized["ntfy_server_url"], "https://ntfy.example.com")
        self.assertEqual(normalized["ntfy_timeout_sec"], 60)
        self.assertEqual(normalized["ntfy_topic"], "")
        self.assertTrue(normalized["ntfy_send_success"])

    def test_build_ntfy_message_stays_within_safe_limit(self) -> None:
        long_text = "가" * 5000
        with mock.patch(
            "modules.utils.notification_sound._current_tool_summary",
            return_value={
                "workflow": "Stage-Batched Pipeline (Recommended)",
                "ocr": "Optimal (HunyuanOCR / PaddleOCR VL)",
                "translator": "Custom Local Server(Gemma)",
            },
        ):
            message = notification_sound.build_ntfy_message(
                {
                    "event_type": "pipeline_failed",
                    "run_type": "batch",
                    "image_count": 25,
                    "message": long_text,
                    "detail": long_text,
                    "source_language": "日本語",
                    "target_language": "한국어",
                    "output_root": "C:/temp/output",
                }
            )

        self.assertLessEqual(
            len(message["body"].encode("utf-8")),
            notification_sound.NTFY_SAFE_MESSAGE_LIMIT_BYTES,
        )
        self.assertEqual(message["priority"], "high")

