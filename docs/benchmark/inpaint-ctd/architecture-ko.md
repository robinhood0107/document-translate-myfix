# Inpaint CTD Architecture

- detector: `RT-DETR-v2`
- CUDA13 benchmark family에서는 RT-DETR-v2를 CPU로 고정한다.
  이유: 현재 ORT CUDA13 + RT-DETR-v2 경로에서 CuDNN internal error가 재현되어, 이번 family는 mask/inpaint 비교만 안정적으로 수행하기 위해 detector를 고정 변수로 둔다.
- precise mask: `CTD refined_mask`
- protect mask: bubble border + strong line protect
- inpainter: `AOT`, `lama_large_512px`, `lama_mpe`
- OCR 고정값
  - China: `HunyuanOCR + Gemma`
  - japan: `PaddleOCR VL + Gemma`
- benchmark gate
  - OCR invariance
  - elapsed / VRAM / cleanup count
