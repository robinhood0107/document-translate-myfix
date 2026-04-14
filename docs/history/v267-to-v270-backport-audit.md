# `comic-translate 2.6.7 -> 2.7.0` Manual Backport Audit

## Summary

This repository started from upstream `comic-translate` `v2.6.7` and then accumulated additional fork-specific runtime and workflow changes.

The `2.7.0` upgrade in this fork was a **manual selective backport** based on upstream compare and release notes. It was not a wholesale merge of upstream `v2.7.0`.

This audit answers two questions:

1. what changed in upstream `2.7.0`
2. which parts were brought into this fork, which were already present, and which were intentionally left out

## Upstream `2.7.0` change groups

### Backported into this fork

| upstream item | fork status | note |
| --- | --- | --- |
| Configurable keyboard shortcuts | applied | adapted to the current shortcuts/settings controller structure |
| PSD export | applied | PhotoshopAPI-based export path introduced |
| PSD import | applied | supported around this fork's exporter/importer round-trip |
| Chapter-aware export | applied | wired into the export dialog and project export flow |
| Rename/Move project files | applied | implemented through the title-bar popup and project controller |
| Startup Home Copy Path / Delete File | applied | added to the recent-project row actions |
| Multi-select text block formatting | applied | added ctrl-click selection and batch formatting support |
| Undo Text Render as one undo step | applied | added as a render macro-style undo boundary |
| Unlimited extra context for custom translator | applied | enabled only for the custom translator path |
| Hebrew / Croatian target languages | applied | added to the target-language list |
| Persian / Hebrew / Arabic RTL handling | applied | adapted into this fork's render direction logic |
| Title bar Snap Multitasking on Windows | applied | adapted to the frameless window path |
| Improved webtoon reader / duplication fixes | applied selectively | only the parts that fit the current fork structure were brought over |
| Claude 4.6 Sonnet label refresh | applied | user-visible label refresh only |

### Already satisfied before the backport

| upstream item | current status | note |
| --- | --- | --- |
| Save batch reports to the project file | already present | this fork already had an equivalent capability through a different implementation |

### Intentionally excluded

| item | reason |
| --- | --- |
| wholesale restoration of upstream file structure and UI | this fork had already diverged heavily around OCR/runtime/product flow |
| wholesale replacement of OCR/runtime internals | the fork already had more advanced local PaddleOCR VL / Hunyuan / Gemma integration |
| `torch_autocast` productization | experiment results did not justify product adoption |
| full arbitrary external PSD fidelity | this round prioritized round-trip support for PSDs produced by the app |

## Integration approach

The backport followed these rules:

- do not overwrite fork files wholesale from upstream
- port features individually
- adapt them to the fork's current controller/state/runtime structure
- do not regress local OCR/runtime/device/factory behavior

In practice, that meant the same user-facing capability was often re-implemented through adapters instead of being copied verbatim from upstream.

## Fork review focus

### baseline features

- shortcuts configuration and restore
- PSD export/import round-trip
- chapter-aware export

### lower-risk additions

- startup home actions (`Copy Path`, `Delete File`, missing-file cleanup)
- unlimited custom translator context
- Hebrew/Croatian target language support
- RTL render handling
- undo render macro
- Claude 4.6 label refresh

### medium-risk additions

- multi-select text formatting
- title-bar rename/move

### higher-risk additions

- webtoon reader improvements
- list-view behavior improvements
- duplication fixes
- Windows snap multitasking

## `torch_autocast` note

`torch_autocast` was evaluated on an experimental track only and was not included in the tracked product code.

Reasons:

- no tracked product diff in the final shipping path
- unstable benefit across engines and runs
- output and runtime changes did not clear the bar for product adoption

## Conclusion

The `2.7.0` upgrade in this fork is a **selective manual backport** that preserves the fork's own OCR/runtime/product direction while importing user-facing value from upstream where it fits safely.
