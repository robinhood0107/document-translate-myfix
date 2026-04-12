# Inpaint Debug Export

The supported verifier is the default automatic runtime only:

- `RT-DETR-v2`
- `Legacy BBox Rescue`
- `Source LaMa`

Run `scripts/export_inpaint_debug.py` to process the full `Sample` tree without OCR, translation, or rendering. The script writes per-image `source`, `legacy_base_mask`, `hard_box_rescue_mask`, `final_mask`, `mask_overlay`, `cleaned`, `metrics.json`, a horizontal compare panel, root `index.md`, root `summary.json`, and ranked `review_samples/` panels under `banchmark_result_log/inpaint_debug/...`.
