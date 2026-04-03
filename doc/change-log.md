# Change Log

## Checkpoint 1

- Created analysis docs for header button flow
- Documented current render-setting behavior
- Recorded implementation strategy before code changes

## Checkpoint 2

- Added shared render-policy helper module for smart color forcing and vertical alignment math
- Extended `TextRenderingSettings` with:
  - `force_font_color`
  - `smart_global_apply_all`
  - `vertical_alignment_id`
- Extended serialized text state with:
  - `vertical_alignment`
  - `source_rect`
  - `block_anchor`

## Checkpoint 3

- Added render-panel UI for:
  - `Apply All SMART Globally`
  - `Force Color`
  - `Top / Center / Bottom`
  - explicit outline `OFF | ON`
- Kept legacy header button wiring untouched
- Hid the old outline checkbox and kept it as the underlying boolean source for compatibility

## Checkpoint 4

- Reworked manual render and manual translate to use the shared render-policy layer
- Reworked regular batch and webtoon batch render-state creation to use the same policy
- Switched text/block matching to prefer `block_anchor` over visible `position`
- Patched webtoon and export coordinate conversion so `position`, `source_rect`, and `block_anchor` are transformed independently

## Checkpoint 5

- Verified changed files with `./.venv/bin/python -m py_compile`
- Verified offscreen window construction of the new render-panel controls
- Verified vertical alignment move/re-align behavior with a `TextBlockItem` smoke test
