# Render Setting Implementation Plan

## Goals

- Keep the current header button routing intact
- Preserve manual vs automatic mode behavior from the screenshots
- Add clearer outline UI without changing existing outline semantics
- Add top/center/bottom vertical alignment as a new global render setting
- Add force controls only for smart color behavior

## Planned extensions

### Data model

- Extend `TextRenderingSettings` with:
  - `force_font_color`
  - `smart_global_apply_all`
  - `vertical_alignment_id`
- Extend `TextItemProperties` with:
  - `vertical_alignment`
  - `source_rect`
- Extend `TextBlockItem` with:
  - `vertical_alignment`
  - `source_rect`

### Style interpretation

Add a shared helper layer for:

- current smart color resolution
- future force-color override
- vertical placement based on source box

Rule set:

- `GLOBAL`: always use current panel state
- `SMART`: use automatic behavior unless force toggle or master smart toggle is on
- `ITEM`: keep current selected-item-only behavior

### Manual integration

- `Render` keeps the same entrypoint but generates text items with:
  - `source_rect`
  - `vertical_alignment`
  - shared color policy
- `Translate` keeps the same entrypoint but post-translation re-wrap will:
  - reuse the shared render policy
  - reapply vertical placement
  - optionally force color when toggles are active

### Automatic integration

- `Translate All` regular batch uses the shared render policy when creating `text_items_state`
- `Translate All` webtoon batch uses the same policy
- final export path remains unchanged and continues to use serialized `text_items_state`

## Compatibility

- old project files must load without new fields
- missing `vertical_alignment` defaults to `top`
- missing `source_rect` falls back to current item position and current size or matching block geometry

## Validation

- manual mode screenshot behavior unchanged
- automatic mode screenshot behavior unchanged
- outline still acts globally
- color can remain smart or be forced globally
- vertical alignment persists across save/load
