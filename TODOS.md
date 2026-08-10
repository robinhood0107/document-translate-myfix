# TODOS

## Geometry

### Separate immutable source geometry from mutable render geometry

**What:** Replace the overloaded `TextBlock.xyxy` contract with typed source,
editable UI, and render geometry.

**Why:** Masking, rendering, checkpoints, and UI edits currently infer ownership
from one mutable box, which can silently move detector/OCR evidence.

**Context:** The inpaint mask now resolves a preserved OCR anchor when compatible
render metadata is coherent, but this remains a compatibility layer.

`TextBlock.xyxy` still serves several incompatible roles: detector/OCR evidence,
editable UI geometry, cache identity, checkpoint signatures, and legacy render
layout.

Before replacing it with immutable source geometry, define and migrate all of
the following together:

- Specify whether a UI box drag moves the mask anchor, render area, or both, and
  preserve undo/redo behavior.
- Define source and render geometry for semantically merged and split blocks,
  including cross-page webtoon blocks.
- Decide whether the existing `text_free` OCR crop contraction remains part of
  source evidence or becomes a separate crop policy.
- Version and migrate checkpoint structure signatures, persistent OCR cache
  keys, and search/replace block keys that currently include `xyxy`.
- Keep old projects readable and add round-trip tests across page-local,
  scene, virtual-page, physical-page, and crop-local coordinate systems.

Completion requires a typed geometry model with one immutable detector/OCR
anchor, explicit editable UI geometry, explicit render geometry, and no mask or
cache consumer reading a field owned by another stage.

**Effort:** XL
**Priority:** P2
**Depends on:** UI drag semantics, merge/split ownership, `text_free` crop policy,
and checkpoint/cache migration rules

## Detection and OCR coverage

### Connect undetected page text and SFX to the translation pipeline

**What:** Add a page-level recovery path for Japanese text and sound effects
that never become detected text blocks.

**Why:** Inpaint quality work can remove residue only for blocks that reach the
mask and translation stages. Undetected text must be measured separately and
recovered before it can be erased or translated.

**Context:** The private inpaint evaluation contract records
`undetected_text` review outcomes, but this signal does not fail the P2 mask and
composite gate. A future implementation should connect page-level detection,
OCR, block classification, and translation without weakening protected-line or
outside-mask guarantees.

Completion requires a neutral-ID corpus gate, detected-versus-annotated recall
metrics, SFX and ordinary-text fixtures, and end-to-end proof that recovered
blocks are translated and safely inpainted.

**Effort:** XL
**Priority:** P3
**Depends on:** stable private target annotations and page-level detection/OCR
ownership

## Completed
