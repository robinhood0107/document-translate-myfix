from __future__ import annotations

import os
import unittest

from modules.utils.download import ModelDownloader, ModelID, models_base_dir
from modules.utils.paths import get_project_models_dir


class RuntimeModelRegistryTests(unittest.TestCase):
    def test_models_base_dir_uses_project_local_models_folder(self) -> None:
        self.assertEqual(models_base_dir, get_project_models_dir())
        self.assertTrue(models_base_dir.endswith(os.path.join("comic-translate", "models")))

    def test_ctd_and_lama_specs_resolve_inside_project_models(self) -> None:
        expected = {
            ModelID.CTD_TORCH: os.path.join(models_base_dir, "detection"),
            ModelID.CTD_ONNX: os.path.join(models_base_dir, "detection"),
            ModelID.LAMA_LARGE_512PX: os.path.join(models_base_dir, "inpainting"),
            ModelID.LAMA_MPE: os.path.join(models_base_dir, "inpainting"),
        }
        for model_id, save_dir in expected.items():
            with self.subTest(model=model_id.value):
                spec = ModelDownloader.registry[model_id]
                self.assertEqual(spec.save_dir, save_dir)
                self.assertNotIn("이식", spec.save_dir)

    def test_lama_mpe_is_saved_under_local_runtime_filename(self) -> None:
        spec = ModelDownloader.registry[ModelID.LAMA_MPE]
        self.assertEqual(spec.files, ["inpainting_lama_mpe.ckpt"])
        self.assertEqual(spec.save_as, {"inpainting_lama_mpe.ckpt": "lama_mpe.ckpt"})


if __name__ == "__main__":
    unittest.main()
