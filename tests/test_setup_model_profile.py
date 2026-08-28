from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules.utils import download


class SetupModelProfileTests(unittest.TestCase):
    def test_core_profile_covers_the_complete_default_pipeline(self) -> None:
        self.assertEqual(
            set(download.application_model_profile("core")),
            {
                download.ModelID.RTDETR_V2_ONNX,
                download.ModelID.FONT_DETECTOR_ONNX,
                download.ModelID.CTD_TORCH,
                download.ModelID.CTD_ONNX,
                download.ModelID.CTD_POSITIVE_CLAIM_ONNX,
                download.ModelID.LAMA_LARGE_512PX,
                download.ModelID.LAMA_MPE,
            },
        )

    def test_forbid_policy_never_calls_the_downloader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = download.ModelSpec(
                id=download.ModelID.RTDETR_V2_ONNX,
                url="https://example.invalid/",
                files=["model.bin"],
                sha256=[None],
                save_dir=temp_dir,
            )
            with (
                mock.patch.dict(os.environ, {"COMIC_MODEL_DOWNLOAD_POLICY": "forbid"}),
                mock.patch.object(download.ModelDownloader, "registry", {spec.id: spec}),
                mock.patch.object(download, "_download_spec") as downloader,
            ):
                with self.assertRaises(download.ModelNotPreparedError):
                    download.ModelDownloader.get(spec.id)
                Path(temp_dir, "model.bin").write_bytes(b"prepared")
                download.ModelDownloader.get(spec.id)
                downloader.assert_not_called()
