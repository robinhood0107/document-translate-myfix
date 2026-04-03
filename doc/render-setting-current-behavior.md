# Render Setting Current Behavior

## Current classification

| Control | Current behavior | Classification |
|---|---|---|
| Font family | Used for new render and batch render | `GLOBAL` |
| Font size dropdown | Edits selected text item only | `ITEM` |
| Line spacing | Used for new render and batch render | `GLOBAL` |
| Font color | Smart fallback: detected block color first, panel color second | `SMART` |
| Horizontal alignment | Used for new render and batch render | `GLOBAL` |
| Bold | Used for new render and batch render | `GLOBAL` |
| Italic | Used for new render and batch render | `GLOBAL` |
| Underline | Used for new render and batch render | `GLOBAL` |
| Outline ON/OFF | Used for new render and batch render | `GLOBAL` |
| Outline color | Used when outline is enabled | `GLOBAL` |
| Outline width | Used when outline is enabled | `GLOBAL` |
| Vertical alignment | Not implemented | `MISSING` |

## What is already global

The following settings already flow through `render_settings()` and are consumed by manual render, regular batch, and webtoon batch:

- `font_family`
- `min_font_size`
- `max_font_size`
- `color`
- `upper_case`
- `outline`
- `outline_color`
- `outline_width`
- `bold`
- `italic`
- `underline`
- `line_spacing`
- `direction`

This means outline is already a true global render setting.

## What is currently smart

Font color is currently resolved by `get_smart_text_color(detected_rgb, setting_color)`.

Policy today:

- if OCR/detection produced a block color, use that
- otherwise use the panel color

So the font color button does not mean "always force this color" today. It means "default color unless detected color exists".

## What is item-only today

The font size dropdown in the right panel only affects the currently selected text item.

For new renders:

- manual render uses `settings_page.get_min_font_size()` and `get_max_font_size()`
- batch render also uses min/max size from settings
- text is auto-fit by `pyside_word_wrap()`

So the visible font size dropdown is not a global render default for new text items.

## Existing gap

There is no top/center/bottom vertical alignment setting.

The current alignment controls only map to:

- `AlignLeft`
- `AlignCenter`
- `AlignRight`

Vertical placement inside the source text box is not modeled separately.

## Implementation rule to preserve

The safest extension path is:

- do not change header button routing
- do not reinterpret existing `GLOBAL` settings as optional toggles
- only add force toggles for current `SMART` controls
- add vertical alignment as a new `GLOBAL` control
