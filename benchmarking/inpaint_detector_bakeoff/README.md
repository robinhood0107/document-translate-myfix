# Inpaint detector bake-off

This lab compares detector evidence without producing translated or inpainted
candidate pages. Raw images, masks, checkpoints, manifests, and review sheets
belong in the ignored `banchmark_result_log/` archive.

The bake-off keeps three concepts separate:

- a pixel claim must come from a detector's raw binary output;
- OCR and block geometry may assign ownership or seed detector-component
  grouping, but never become an edit mask by themselves;
- evaluation target, protected structure, and ambiguous structure masks are
  pairwise-disjoint source-only annotations.

The current evidence-first composition selects raw detector components touching
existing content-component provenance, clips them to the owning semantic text
prior, subtracts the sealed baseline ownership and exact protections, and emits
one page-level positive edit mask. Required conservative skip reasons can scope
this handoff without adding a new image threshold, Hough rule, or autonomous
residue search.

Reference adapters preserve original Python behavior before any product port:

- Ballons CTD raw/refined/3 px dilated masks;
- Ballons CTBD preprocessing, boxes, and adaptive content-mask reference;
- Ballons YSG Ultralytics boxes/OBBs and mask dilation;
- SickZil SegNet 0.1.0 frozen-graph segmentation.

Use the scripts in `scripts/` with the managed private-artifact harness. A
candidate is not a finalist merely because target coverage is high: raw claim
structure overlap, no-edit false edits, component coverage, ownership leakage,
runtime, and model provenance are all retained in the Stage 1 record.
