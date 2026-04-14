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

