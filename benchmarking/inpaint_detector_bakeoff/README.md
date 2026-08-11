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

The provenance-fusion experiment also keeps RT-DETR's raw text-box origin. A
required skip directly backed by a raw text detection may use that box as an
ownership boundary for CTD raw pixels; a block synthesized from unmatched
bubble content remains restricted to its content components. The box never
becomes an edit mask, and the final claim remains a subset of CTD raw output.

For a fixed-size finalist, `scripts/export_inpaint_ctd_onnx.py` reproduces the
CTD graph at the selected input size. The provenance runner accepts
`--ctd-runtime onnxruntime`; parity is judged on the final `positive_edit`
binary mask, while the separately retained raw network mask remains diagnostic
telemetry.

The generic positive-mask Stage 2 runner loads the sealed rewritten-PR3 image,
generates each page candidate from the immutable source with at most one LaMa
call, and copies back only the exact positive edit mask. Its final mask is the
union of the sealed baseline mask and positive edit, so outside-final changes
remain measurable as a hard invariant.

The replacement-mask ablation separately evaluates whether detector evidence
can narrow existing source-owned edits. It retains non-source rewritten-PR3
pixels, regenerates the replacement mask once from the immutable original, and
restores rejected source-owned pixels from that original. Required additions
remain raw detector pixels; RT-DETR boxes can validate ownership of an existing
source component but cannot create edit pixels.

Reference adapters preserve original Python behavior before any product port:

- Ballons CTD raw/refined/3 px dilated masks;
- Ballons CTBD preprocessing, boxes, and adaptive content-mask reference;
- Ballons YSG Ultralytics boxes/OBBs and mask dilation;
- Manga109 YOLO26 text-instance ownership from the pinned Python runtime;
- SickZil SegNet 0.1.0 frozen-graph segmentation.

Use the scripts in `scripts/` with the managed private-artifact harness. A
candidate is not a finalist merely because target coverage is high: raw claim
structure overlap, no-edit false edits, component coverage, ownership leakage,
runtime, and model provenance are all retained in the Stage 1 record.

Stage 2 includes a pinned Ballons end-to-end reference. It runs Ballons' native
CTD refined mask, block order, sequential crops, and whole-bubble flat fill
with the source-parity LaMa Large core, writes the actual changed mask, and
scores target components, exact protected and
ambiguous annotations, detector-mask outside changes, residue, CUDA runtime,
and VRAM. `check_inpaint_ballons_lama_reference_parity.py` proves the maintained
LaMa path is pixel-exact against Ballons' original Python `_inpaint` method on a
golden CUDA fixture before the reference result is used.
