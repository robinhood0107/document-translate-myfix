# Header Button Flow Analysis

## Screenshot mapping

- Screenshot 1 corresponds to `Manual` mode.
  - Header buttons `Detect`, `Recognize`, `Translate`, `Segment`, `Clean`, `Render` are enabled.
  - `Translate All` is disabled.
- Screenshot 2 corresponds to `Automatic` mode.
  - Header buttons `Detect`, `Recognize`, `Translate`, `Segment`, `Clean`, `Render` are disabled.
  - `Translate All` is enabled.

## Header UI

- Header controls are created in `app/ui/main_window/builders/workspace.py`.
- `self.hbutton_group` contains 6 buttons in this exact order:
  - `Detect`
  - `Recognize`
  - `Translate`
  - `Segment`
  - `Clean`
  - `Render`
- `self.manual_radio`, `self.automatic_radio`, and `self.translate_button` live in the same header layout.

## Mode switching

- `Manual` radio connects to `controller.manual_mode_selected()`.
  - Calls `enable_hbutton_group()`
  - Disables `translate_button`
  - Disables `cancel_button`
- `Automatic` radio connects to `controller.batch_mode_selected()`.
  - Calls `disable_hbutton_group()`
  - Enables `translate_button`
  - Enables `cancel_button`

This means the existing header behavior in the screenshots is already implemented correctly and should be preserved.

## Button wiring

All 6 header buttons are wired in `controller.py`:

- `Detect` -> `controller.block_detect()` -> `ManualWorkflowController.block_detect()`
- `Recognize` -> `controller.ocr()` -> `ManualWorkflowController.ocr()`
- `Translate` -> `controller.translate_image()` -> `ManualWorkflowController.translate_image()`
- `Segment` -> `controller.load_segmentation_points()` -> `ManualWorkflowController.load_segmentation_points()`
- `Clean` -> `controller.inpaint_and_set()` -> `ManualWorkflowController.inpaint_and_set()`
- `Render` -> `TextController.render_text()`

`Translate All` is separate:

- `Translate All` -> `controller.start_batch_process()`
- Regular pages -> `pipeline.batch_process()`
- Webtoon mode -> `pipeline.webtoon_batch_process()`

## Manual flow summary

### Detect

- Single/current page:
  - `pipeline.detect_blocks()`
  - `pipeline.on_blk_detect_complete()`
- Multi-page:
  - `ManualWorkflowController.block_detect()` performs detection per selected path and stores rectangles into `image_states[file]["viewer_state"]["rectangles"]`

### Recognize

- Single/current page:
  - `pipeline.OCR_image()` or `pipeline.OCR_webtoon_visible_area()`
- Multi-page:
  - `ManualWorkflowController.ocr()` processes OCR per selected path and updates `blk_list`

### Translate

- Single/current page:
  - `pipeline.translate_image()` or `pipeline.translate_webtoon_visible_area()`
  - then `ManualWorkflowController.update_translated_text_items()`
- Multi-page:
  - `ManualWorkflowController.translate_image()` translates selected paths
  - then `update_translated_text_items()` re-wraps visible text items

### Segment

- `ManualWorkflowController.load_segmentation_points()`
- Clears live rectangles and text items, computes `inpaint_bboxes`, and restores segmentation strokes into state

### Clean

- `ManualWorkflowController.inpaint_and_set()`
- Single page uses `pipeline.inpaint()`
- Multi-page uses `inpainting.inpaint_page_from_saved_strokes()`

### Render

- `TextController.render_text()`
- Single page:
  - formats translations
  - calls `manual_wrap()`
  - emits `blk_rendered`
  - creates `TextBlockItem` instances
- Multi-page:
  - wraps blocks with `pyside_word_wrap()`
  - stores new `text_items_state` into each selected page

## Automatic flow summary

### Translate All

- `controller.start_batch_process()` validates settings and dispatches:
  - `pipeline.batch_process()` for normal pages
  - `pipeline.webtoon_batch_process()` for webtoon mode

### Regular batch

- `pipeline.batch_process()` performs:
  - detection
  - OCR
  - inpainting
  - translation
  - text rendering state creation
- Render output is serialized into `image_states[image_path]["viewer_state"]["text_items_state"]`

### Webtoon batch

- `pipeline.webtoon_batch_process()` streams virtual pages
- Render output is serialized into the same `text_items_state` concept
- Final export uses `ImageSaveRenderer`

## Safe implementation boundaries

To avoid breaking the current header/button model:

- Keep all existing button wiring in `controller.py`
- Keep `manual_mode_selected()` and `batch_mode_selected()` behavior
- Keep existing manual entrypoints in `ManualWorkflowController`
- Keep `start_batch_process()` as the only automatic entrypoint
- Add new behavior inside render-setting interpretation and text state generation, not by changing header routing
