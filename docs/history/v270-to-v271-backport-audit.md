# `comic-translate 2.7.0 -> 2.7.1` Manual Backport Audit

## Summary

This fork upgrades from the locally adapted `2.7.0` baseline to a selectively backported `2.7.1` baseline.

Upstream `2.7.1` touched only a small set of files. This fork applies the relevant fixes manually and adapts them to the current fork structure instead of merging the release wholesale.

## Upstream file delta

Upstream `v2.7.0...v2.7.1` changed these files:

- `.github/workflows/build-macos-dmg.yml`
- `app/controllers/image.py`
- `app/controllers/projects.py`
- `app/controllers/psd_importer.py`
- `app/controllers/task_runner.py`
- `app/ui/list_view_image_loader.py`
- `app/ui/main_window/builders/nav.py`
- `app/version.py`

## Applied in this fork

### `app/controllers/image.py`

- import `prepare_psd_font_catalog()` from the PSD importer
- prepare the PSD font catalog before threaded PSD import begins
- route async navigation callbacks through `QTimer.singleShot(..., self.main, ...)`

### `app/controllers/projects.py`

- route autosave worker error/finished callbacks through the main-thread receiver

### `app/controllers/psd_importer.py`

- add public helper `prepare_psd_font_catalog()`
- add logging around `get_image_data()`, `get_channel_by_id()`, and `get_channel_by_index()` decode failures
- log fully empty RGB channel decode cases
- add `_can_build_font_catalog_in_current_thread()` and avoid eager font-catalog building from the wrong thread

### `app/controllers/task_runner.py`

- route queued operation continuation through `QTimer.singleShot(..., self.main, ...)`
- route `result`, `error`, and `finished` callbacks back onto the main thread consistently

### `app/ui/list_view_image_loader.py`

- move worker output from `QPixmap` to `QImage`
- perform `QPixmap.fromImage(...)` only on the main thread
- use deep-copied `QImage` data so thumbnail delivery does not depend on temporary numpy buffers
- replace direct cross-thread method calls with queued signal delivery
- keep the fork's card-based avatar update path while applying the upstream threading fix

### `app/ui/main_window/builders/nav.py`

- keep `PSD` next to `Project File`
- simplify the visible label from `PSD File` to `PSD`

### `app/version.py`

- bump the application version to `2.7.1`

## Explicitly excluded

### `.github/workflows/build-macos-dmg.yml`

This fork does not ship the upstream macOS DMG workflow in this round.

Reason:

- the current packaging/release workflow for this fork is Windows-first
- the user explicitly excluded the macOS workflow from the `2.7.1` selective backport scope

## Adaptation notes

This backport is not a verbatim file copy.

- the fork already carries local OCR/runtime changes that upstream does not have
- the fork's list view still uses card/avatar widgets instead of the exact upstream item-decoration path
- PSD import/export has already diverged to support this fork's round-trip flow

So the `2.7.1` changes are applied as **behavioral fixes**, not as wholesale upstream file replacement.

## Conclusion

The `2.7.1` backport in this fork is intentionally narrow: it focuses on PSD import safety, UI-thread correctness, thumbnail loader stability, navigation cleanup, and the version bump, while preserving the fork's local runtime and OCR architecture.
