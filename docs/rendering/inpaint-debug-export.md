# Inpaint Debug Export

Automatic Mode debug exports help separate three failure classes:

- Detector issue: bounding boxes are too small or missing in `detector_overlays`.
- Mask issue: boxes look correct, but `raw_masks`, `mask_overlays`, or `cleanup_mask_delta` still miss glyph pixels.
- Inpainter issue: masks look correct, but `cleaned_images` still show text residue or artifacts.

`Translate All` and `One-Page Auto` share the same export settings and write into the same `comic_translate_<timestamp>` tree.

For bulk review, run `scripts/export_inpaint_debug.py` to export the same detector/mask/inpaint/cleanup artifacts for `Sample/japan` and `Sample/China` into `banchmark_result_log/inpaint_debug/...`.
