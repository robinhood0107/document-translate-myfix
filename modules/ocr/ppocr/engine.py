from __future__ import annotations

from typing import Any, List, Tuple, Optional
import logging
import numpy as np
from PySide6.QtCore import QCoreApplication

from ..base import OCREngine
from modules.utils.language_utils import is_no_space_lang
from modules.utils.textblock import TextBlock
from modules.utils.textblock import lists_to_blk_list, sort_textblock_rectangles
from modules.utils.ocr_debug import (
	OCR_STATUS_EMPTY_AFTER_RETRY,
	OCR_STATUS_EMPTY_INITIAL,
	OCR_STATUS_OK,
	OCR_STATUS_OK_AFTER_RETRY,
	build_retry_crop_bbox,
	build_retry_crop_from_bbox,
	crop_bbox_image,
	resolve_block_crop_bbox,
	set_block_ocr_crop_diagnostics,
	set_block_ocr_diagnostics,
)
from modules.utils.device import get_providers
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.onnx import make_session, make_session_options
from .preprocessing import det_preprocess, crop_quad, rec_resize_norm
from .postprocessing import DBPostProcessor, CTCLabelDecoder

logger = logging.getLogger(__name__)


LANG_TO_REC_MODEL: dict[str, ModelID] = {
	'ch': ModelID.PPOCR_V5_REC_MOBILE,
	'en': ModelID.PPOCR_V5_REC_EN_MOBILE,
	'ko': ModelID.PPOCR_V5_REC_KOREAN_MOBILE,
	'latin': ModelID.PPOCR_V5_REC_LATIN_MOBILE,
	'ru': ModelID.PPOCR_V5_REC_ESLAV_MOBILE,
	'eslav': ModelID.PPOCR_V5_REC_ESLAV_MOBILE,
}


class PPOCRv5Engine(OCREngine):
	"""Lightweight PP-OCRv5 ONNX pipeline (det + rec).
	"""

	def __init__(self):
		# Sessions are created lazily in initialize(); keep startup light.
		self.det_sess: Optional[Any] = None
		self.rec_sess: Optional[Any] = None
		self.det_post = DBPostProcessor(
			thresh=0.3, 
			box_thresh=0.5, 
			unclip_ratio=2.0, 
			use_dilation=False
		)
		self.decoder: Optional[CTCLabelDecoder] = None
		self.rec_img_shape = (3, 48, 320)

	def initialize(
		self, 
		lang: str = 'ch', 
		device: str = 'cpu', 
		det_model: str = 'mobile'
	) -> None:
		# Ensure models present
		det_id = ModelID.PPOCR_V5_DET_MOBILE if det_model == 'mobile' else ModelID.PPOCR_V5_DET_SERVER
		rec_id = LANG_TO_REC_MODEL.get(lang, ModelID.PPOCR_V5_REC_LATIN_MOBILE)
		ModelDownloader.ensure([det_id, rec_id])

		# Load ONNX sessions
		det_path = ModelDownloader.primary_path(det_id)
		rec_paths = ModelDownloader.file_path_map(rec_id)
		rec_model = [p for n, p in rec_paths.items() if n.endswith('.onnx')][0]
		# dict file name can vary per lang
		dict_file = [p for n, p in rec_paths.items() if n.endswith('.txt')]
		dict_path = dict_file[0] if dict_file else None

		providers = get_providers(device)
		sess_opt = make_session_options(log_severity_level=3)
		self.det_sess = make_session(det_path, sess_options=sess_opt, providers=providers)
		self.rec_sess = make_session(rec_model, sess_options=sess_opt, providers=providers)

		# Prepare CTC decoder
		if dict_path:
			self.decoder = CTCLabelDecoder(dict_path=dict_path)
		else:
			# try pull embedded vocab from model metadata
			meta = self.rec_sess.get_modelmeta().custom_metadata_map
			if 'character' in meta:
				chars = meta['character'].splitlines()
				self.decoder = CTCLabelDecoder(charset=chars)
			else:
				raise RuntimeError('Recognition dictionary not found')

	def _det_infer(self, img: np.ndarray) -> Tuple[np.ndarray, List[float]]:
		assert self.det_sess is not None
		inp = det_preprocess(img, limit_side_len=960, limit_type='min')
		input_name = self.det_sess.get_inputs()[0].name
		output_name = self.det_sess.get_outputs()[0].name
		pred = self.det_sess.run([output_name], {input_name: inp})[0]
		boxes, scores = self.det_post(pred, (img.shape[0], img.shape[1]))
		return boxes, scores

	def _rec_infer(self, crops: List[np.ndarray]) -> Tuple[List[str], List[float]]:
		assert self.rec_sess is not None and self.decoder is not None
		if not crops:
			return [], []
		# batch by padded width heuristic
		ratios = [c.shape[1] / float(max(1, c.shape[0])) for c in crops]
		order = np.argsort(ratios)
		texts = [""] * len(crops)
		confs = [0.0] * len(crops)
		bs = 8
		c, H, W = self.rec_img_shape
		for b in range(0, len(crops), bs):
			idxs = order[b:b+bs]
			max_ratio = max(ratios[i] for i in idxs) if idxs.size > 0 else (W/float(H))
			batch = [rec_resize_norm(crops[i], self.rec_img_shape, max_ratio)[None, ...] for i in idxs]
			x = np.concatenate(batch, axis=0).astype(np.float32)
			inp_name = self.rec_sess.get_inputs()[0].name
			out_name = self.rec_sess.get_outputs()[0].name
			logits = self.rec_sess.run([out_name], {inp_name: x})[0]  # (N, T, C) or (N, C, T)
			if logits.ndim == 3 and logits.shape[1] > logits.shape[2]:
				# If output is (N, C, T), transpose to (N, T, C)
				logits = np.transpose(logits, (0, 2, 1))
			# Match PaddleOCR behavior: do not drop characters by per-step prob threshold
			dec_texts, dec_confs = self.decoder(logits, prob_threshold=0.0)
			for oi, t, s in zip(idxs, dec_texts, dec_confs):
				texts[oi] = t
				confs[oi] = float(s)
		return texts, confs

	@staticmethod
	def _quad_to_bbox(quad: np.ndarray) -> tuple[int, int, int, int]:
		xs = quad[:, 0]
		ys = quad[:, 1]
		return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

	@staticmethod
	def _compact_text(text: str) -> str:
		return "".join((text or "").split())

	def _block_geometry(self, blk: TextBlock) -> tuple[float, float, float]:
		x1, y1, x2, y2 = [float(v) for v in blk.xyxy]
		width = max(1.0, x2 - x1)
		height = max(1.0, y2 - y1)
		return width, height, width * height

	def _is_suspicious_result(self, blk: TextBlock, text: str, conf: float) -> bool:
		compact = self._compact_text(text)
		if not compact:
			return True

		width, height, area = self._block_geometry(blk)
		text_len = len(compact)
		conf = float(conf or 0.0)

		if text_len <= 1 and area >= 25000:
			return True
		if text_len <= 2 and area >= 90000:
			return True
		if text_len <= 3 and conf < 0.45 and area >= 50000:
			return True
		if text_len <= 4 and conf < 0.60 and min(width, height) >= 220:
			return True
		if conf < 0.18 and area >= 30000:
			return True
		return False

	def _candidate_score(self, blk: TextBlock, text: str, conf: float, source: str) -> float:
		compact = self._compact_text(text)
		if not compact:
			return float("-inf")

		width, height, area = self._block_geometry(blk)
		text_len = len(compact)
		score = float(conf or 0.0) * 10.0 + min(text_len, 12)

		if source.startswith("local_det"):
			score += 0.75
		if source.startswith("retry_local_det"):
			score += 0.25

		if text_len <= 1 and area >= 25000:
			score -= 8.0
		elif text_len <= 2 and area >= 90000:
			score -= 6.0
		elif text_len <= 3 and float(conf or 0.0) < 0.45 and area >= 50000:
			score -= 4.0
		elif text_len <= 4 and float(conf or 0.0) < 0.60 and min(width, height) >= 220:
			score -= 3.0

		return score

	def _should_probe_local_det(self, blk: TextBlock, text: str, conf: float) -> bool:
		compact = self._compact_text(text)
		width, height, area = self._block_geometry(blk)
		conf = float(conf or 0.0)

		if area >= 160000:
			return True
		if height >= 220 and width >= 360:
			return True
		if getattr(blk, "text_class", "") == "text_bubble" and width >= 420 and height >= 150:
			return True
		if conf < 0.88 and area >= 70000:
			return True
		if len(compact) <= 8 and area >= 180000:
			return True
		return False

	def _select_best_candidate(self, blk: TextBlock, candidates: list[tuple[str, str, float]]) -> tuple[str, float, str]:
		best_text = ""
		best_conf = 0.0
		best_source = ""
		best_score = float("-inf")

		for source, text, conf in candidates:
			score = self._candidate_score(blk, text, conf, source)
			if score > best_score:
				best_text = (text or "").strip()
				best_conf = float(conf or 0.0)
				best_source = source
				best_score = score

		return best_text, best_conf, best_source

	def _det_rec_in_crop(self, crop: np.ndarray, blk: TextBlock) -> tuple[str, float]:
		boxes, _ = self._det_infer(crop)
		if boxes is None or len(boxes) == 0:
			return "", 0.0

		line_crops = [crop_quad(crop, quad.astype(np.float32)) for quad in boxes]
		line_texts, line_confs = self._rec_infer(line_crops)

		line_candidates: list[tuple[tuple[int, int, int, int], str, float, int, int]] = []
		for quad, text, conf in zip(boxes, line_texts, line_confs):
			compact_text = (text or "").strip()
			if not compact_text:
				continue
			bbox = self._quad_to_bbox(quad)
			width = max(1, bbox[2] - bbox[0])
			height = max(1, bbox[3] - bbox[1])
			line_candidates.append((bbox, compact_text, float(conf or 0.0), width, height))

		if not line_candidates:
			return "", 0.0

		if len(line_candidates) > 1:
			heights = [height for _, _, _, _, height in line_candidates]
			widths = [width for _, _, _, width, _ in line_candidates]
			median_h = float(np.median(heights))
			median_w = float(np.median(widths))
			filtered_candidates = []
			for bbox, compact_text, conf, width, height in line_candidates:
				tiny_line = (
					height < max(10.0, median_h * 0.45)
					or width < max(20.0, median_w * 0.22)
				)
				if tiny_line and conf < 0.45:
					continue
				filtered_candidates.append((bbox, compact_text, conf, width, height))
			if filtered_candidates:
				line_candidates = filtered_candidates

		line_entries: list[tuple[tuple[int, int, int, int], str]] = [
			(bbox, compact_text) for bbox, compact_text, _, _, _ in line_candidates
		]
		valid_confs: list[float] = [conf for _, _, conf, _, _ in line_candidates]

		sorted_entries = sort_textblock_rectangles(line_entries, blk.source_lang_direction)
		if is_no_space_lang(getattr(blk, "source_lang", "")):
			joined_text = "".join(text for _, text in sorted_entries if text)
		else:
			joined_text = " ".join(text for _, text in sorted_entries if text)

		if not joined_text.strip():
			return "", 0.0

		avg_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
		return joined_text.strip(), avg_conf

	def _rec_only_blocks(self, img: np.ndarray, blk_list: List[TextBlock]) -> List[TextBlock]:
		initial_payloads: list[tuple[int, TextBlock, np.ndarray | None, tuple[int, int, int, int] | None]] = []
		initial_crops: list[np.ndarray] = []
		initial_indices: list[int] = []

		for idx, blk in enumerate(blk_list):
			bbox, crop_source = resolve_block_crop_bbox(
				blk,
				img.shape,
				bubble_as_clamp=True,
				fallback_to_bubble=getattr(blk, "xyxy", None) is None,
			)
			set_block_ocr_crop_diagnostics(blk, effective_crop_xyxy=bbox, crop_source=crop_source)
			crop = crop_bbox_image(img, bbox, auto_rotate_tall=True) if bbox is not None else None
			initial_payloads.append((idx, blk, crop, bbox))
			if crop is not None:
				initial_crops.append(crop)
				initial_indices.append(idx)

		initial_texts, initial_confs = self._rec_infer(initial_crops)
		initial_results = {
			idx: (text, conf)
			for idx, text, conf in zip(initial_indices, initial_texts, initial_confs)
		}

		initial_empty_indices: list[int] = []
		initial_suspicious_indices: list[int] = []
		retry_candidates: list[tuple[int, TextBlock, str, float]] = []

		for idx, blk, crop, _ in initial_payloads:
			if crop is None:
				initial_empty_indices.append(idx)
				retry_candidates.append((idx, blk, "", 0.0))
				set_block_ocr_diagnostics(
					blk,
					text="",
					confidence=0.0,
					status=OCR_STATUS_EMPTY_INITIAL,
					empty_reason=QCoreApplication.translate("Messages", "Initial crop is empty."),
					attempt_count=1,
				)
				continue

			text, conf = initial_results.get(idx, ("", 0.0))
			text = (text or "").strip()
			if text:
				candidates: list[tuple[str, str, float]] = [("initial_rec_only", text, conf)]
				if self._should_probe_local_det(blk, text, conf):
					local_text, local_conf = self._det_rec_in_crop(crop, blk)
					if local_text:
						candidates.append(("local_det", local_text, local_conf))
					best_text, best_conf, _ = self._select_best_candidate(blk, candidates)
					if best_text:
						text = best_text
						conf = best_conf
				if self._is_suspicious_result(blk, text, conf):
					initial_suspicious_indices.append(idx)
					retry_candidates.append((idx, blk, text, conf))
					continue
				set_block_ocr_diagnostics(
					blk,
					text=text,
					confidence=conf,
					status=OCR_STATUS_OK,
					empty_reason="",
					attempt_count=1,
				)
				continue

			initial_empty_indices.append(idx)
			retry_candidates.append((idx, blk, "", 0.0))
			set_block_ocr_diagnostics(
				blk,
				text="",
				confidence=0.0,
				status=OCR_STATUS_EMPTY_INITIAL,
				empty_reason=QCoreApplication.translate(
					"Messages",
					"OCR returned empty text on the initial crop.",
				),
				attempt_count=1,
			)

		if initial_empty_indices:
			logger.info("ppocr empty blocks after initial pass: %s", initial_empty_indices)
		if initial_suspicious_indices:
			logger.info("ppocr suspicious non-empty blocks after initial pass: %s", initial_suspicious_indices)

		retry_payloads: list[tuple[int, TextBlock, np.ndarray | None, tuple[int, int, int, int] | None]] = []
		retry_crops: list[np.ndarray] = []
		retry_indices: list[int] = []

		for idx, blk, _, _ in retry_candidates:
			retry_source_bbox = getattr(blk, "xyxy", None)
			if retry_source_bbox is None:
				retry_source_bbox = getattr(blk, "ocr_effective_crop_xyxy", None)
			retry_clamp = getattr(blk, "bubble_xyxy", None) if getattr(blk, "xyxy", None) is not None else None
			retry_bbox = build_retry_crop_bbox(img.shape, retry_source_bbox, clamp_xyxy=retry_clamp)
			set_block_ocr_crop_diagnostics(blk, retry_crop_xyxy=retry_bbox)
			retry_crop = build_retry_crop_from_bbox(img, retry_bbox)
			retry_payloads.append((idx, blk, retry_crop, retry_bbox))
			if retry_crop is not None:
				retry_crops.append(retry_crop)
				retry_indices.append(idx)

		retry_texts, retry_confs = self._rec_infer(retry_crops)
		retry_results = {
			idx: (text, conf)
			for idx, text, conf in zip(retry_indices, retry_texts, retry_confs)
		}

		retry_empty_indices: list[int] = []
		for idx, blk, retry_crop, _ in retry_payloads:
			initial_text, initial_conf = next(
				(candidate_text, candidate_conf)
				for candidate_idx, candidate_blk, candidate_text, candidate_conf in retry_candidates
				if candidate_idx == idx and candidate_blk is blk
			)
			candidates: list[tuple[str, str, float]] = []
			if initial_text:
				candidates.append(("initial_rec_only", initial_text, initial_conf))

			initial_crop = initial_payloads[idx][2]
			if initial_crop is not None:
				local_text, local_conf = self._det_rec_in_crop(initial_crop, blk)
				if local_text:
					candidates.append(("local_det", local_text, local_conf))

			if retry_crop is None:
				text, conf, _ = self._select_best_candidate(blk, candidates)
				text = (text or "").strip()
				if text and not self._is_suspicious_result(blk, text, conf):
					set_block_ocr_diagnostics(
						blk,
						text=text,
						confidence=conf,
						status=OCR_STATUS_OK_AFTER_RETRY,
						empty_reason="",
						attempt_count=2,
					)
					continue
				retry_empty_indices.append(idx)
				set_block_ocr_diagnostics(
					blk,
					text="",
					confidence=0.0,
					status=OCR_STATUS_EMPTY_AFTER_RETRY,
					empty_reason=QCoreApplication.translate(
						"Messages",
						"Retry crop is empty after expansion.",
					),
					attempt_count=2,
				)
				continue

			local_retry_text, local_retry_conf = self._det_rec_in_crop(retry_crop, blk)
			if local_retry_text:
				candidates.append(("retry_local_det", local_retry_text, local_retry_conf))
			rec_retry_text, rec_retry_conf = retry_results.get(idx, ("", 0.0))
			if rec_retry_text:
				candidates.append(("retry_rec_only", rec_retry_text, rec_retry_conf))

			text, conf, _ = self._select_best_candidate(blk, candidates)
			text = (text or "").strip()
			if text:
				if self._is_suspicious_result(blk, text, conf):
					retry_empty_indices.append(idx)
					set_block_ocr_diagnostics(
						blk,
						text="",
						confidence=0.0,
						status=OCR_STATUS_EMPTY_AFTER_RETRY,
						empty_reason=QCoreApplication.translate(
							"Messages",
							"Retry also failed after contrast preprocessing.",
						),
						attempt_count=2,
					)
					continue
				set_block_ocr_diagnostics(
					blk,
					text=text,
					confidence=conf,
					status=OCR_STATUS_OK_AFTER_RETRY,
					empty_reason="",
					attempt_count=2,
				)
				continue

			retry_empty_indices.append(idx)
			set_block_ocr_diagnostics(
				blk,
				text="",
				confidence=0.0,
				status=OCR_STATUS_EMPTY_AFTER_RETRY,
				empty_reason=QCoreApplication.translate(
					"Messages",
					"Retry also failed after contrast preprocessing.",
				),
				attempt_count=2,
			)

		if retry_empty_indices:
			logger.info("ppocr empty blocks after retry: %s", retry_empty_indices)

		return blk_list

	def process_image(self, img: np.ndarray, blk_list: List[TextBlock]) -> List[TextBlock]:
		if self.det_sess is None or self.rec_sess is None or self.decoder is None:
			return blk_list
		if blk_list:
			return self._rec_only_blocks(img, blk_list)
		boxes, _ = self._det_infer(img)
		if boxes is None or len(boxes) == 0:
			return blk_list
		crops = [crop_quad(img, quad.astype(np.float32)) for quad in boxes]
		texts, _ = self._rec_infer(crops)
		# map quads -> axis-aligned boxes
		bboxes = []
		for quad in boxes:
			xs = quad[:, 0]
			ys = quad[:, 1]
			x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
			bboxes.append((x1, y1, x2, y2))
		return lists_to_blk_list(blk_list, bboxes, texts)

