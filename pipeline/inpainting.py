import numpy as np
import logging
import imkit as imk
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush

from modules.utils.gpu_handoff import (
    DEFAULT_VRAM_RELEASE_MIN_DROP_MB,
    DEFAULT_VRAM_RELEASE_POLL_SEC,
    DEFAULT_VRAM_RELEASE_TIMEOUT_SEC,
    cleanup_python_cuda_memory,
    estimate_torch_cuda_storage_mb,
    wait_for_vram_release,
)
from modules.utils.gpu_metrics import query_cuda_handoff_metrics
from modules.utils.device import resolve_device
from modules.utils.inpaint_strokes import (
    PATCH_KIND_INPAINT,
    normalize_stroke_role,
    normalize_patch_kind,
    STROKE_ROLE_ADD,
    STROKE_ROLE_EXCLUDE,
    STROKE_ROLE_GENERATED,
)
from modules.inpainting.runtime_contract import (
    INPAINT_RETRY_POLICY_VERSION,
    InpaintingCudaOOMError,
    InpaintingRuntimeContractError,
    bounded_retry_roi,
    inpaint_cuda_oom_message,
    inpaint_release_unconfirmed_message,
    inspect_learned_inpainter_runtime,
    is_cuda_oom_error,
    runtime_mask_diagnostics,
    validate_learned_inpaint_runtime,
)
from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
    normalize_edit_mask,
)
from modules.utils.pipeline_config import inpaint_map, get_config, get_inpainter_runtime
from modules.inpainting.source_lama_blockwise import (
    release_source_lama_cache,
    source_lama_blockwise_inpaint,
)

logger = logging.getLogger(__name__)


class InpaintingHandler:
    """Handles image inpainting functionality."""
    
    def __init__(self, main_page):
        self.main_page = main_page
        self.inpainter_cache = None
        self.cached_inpainter_key = None
        self.cached_inpainter_runtime_signature = None
        self.inpainter_driver_baseline = None
        self.last_inpaint_edit_mask = None
        self.last_inpaint_diagnostics: dict[str, Any] = {}
        self.inpainter_runtime_contract: dict[str, Any] = {}
        self.last_profile_change_release: dict[str, Any] = {}

    def _ensure_inpainter(self):
        settings_page = self.main_page.settings_page
        runtime = get_inpainter_runtime(settings_page)
        inpainter_key = runtime['key']
        runtime_signature = (
            str(inpainter_key),
            str(runtime.get('backend', '') or ''),
            str(runtime.get('device', '') or ''),
            int(runtime.get('inpaint_size', 0) or 0),
            str(runtime.get('precision', '') or ''),
        )
        validate_learned_inpaint_runtime(
            inpainter_key=str(inpainter_key),
            device=str(runtime.get('device', '') or ''),
            precision=str(runtime.get('precision', '') or ''),
        )
        if (
            self.inpainter_cache is None
            or self.cached_inpainter_runtime_signature != runtime_signature
        ):
            if self.inpainter_cache is not None:
                release_report = self.release_inpainter_resources()
                self.last_profile_change_release = release_report
                release_gate = dict(
                    release_report.get("vram_release_gate") or {}
                )
                if (
                    release_gate.get("required")
                    and not release_gate.get("observed")
                ):
                    raise InpaintingRuntimeContractError(
                        inpaint_release_unconfirmed_message()
                    )
            self.inpainter_driver_baseline = query_cuda_handoff_metrics()
            backend = runtime['backend']
            device = resolve_device(settings_page.is_gpu_enabled(), backend)
            validate_learned_inpaint_runtime(
                inpainter_key=str(inpainter_key),
                device=str(device),
                precision=str(runtime.get('precision', '') or ''),
            )
            InpainterClass = inpaint_map[inpainter_key]
            candidate_inpainter = InpainterClass(
                device,
                backend=backend,
                runtime_device=runtime.get('device', device),
                inpaint_size=runtime.get('inpaint_size'),
                precision=runtime.get('precision'),
            )
            try:
                runtime_contract = inspect_learned_inpainter_runtime(
                    candidate_inpainter,
                    inpainter_key=str(inpainter_key),
                    requested_device=str(device),
                    requested_precision=str(
                        runtime.get('precision', '') or ''
                    ),
                )
            except Exception:
                release_report = self._detach_inpainter_native_resources(
                    candidate_inpainter
                )
                cleanup_python_cuda_memory(
                    release_cuda_allocator=bool(
                        release_report.get("gpu_release_expected")
                    ),
                )
                raise
            self.inpainter_cache = candidate_inpainter
            self.inpainter_runtime_contract = runtime_contract
            self.cached_inpainter_key = inpainter_key
            self.cached_inpainter_runtime_signature = runtime_signature
        return self.inpainter_cache

    @staticmethod
    def _detach_inpainter_native_resources(inpainter: Any) -> dict[str, Any]:
        if inpainter is None:
            return {
                "cached": False,
                "loaded_native_resource_count": 0,
                "expected_process_reclaim_mb": 0.0,
                "untracked_gpu_resource_count": 0,
                "gpu_release_expected": False,
            }
        device = str(
            getattr(
                inpainter,
                "runtime_device",
                getattr(inpainter, "device", ""),
            )
            or ""
        )
        loaded_native_resource_count = 0
        expected_process_reclaim_mb = 0.0
        untracked_gpu_resource_count = 0
        for attribute in ("model", "session"):
            if not hasattr(inpainter, attribute):
                continue
            resource = getattr(inpainter, attribute, None)
            if resource is not None:
                loaded_native_resource_count += 1
                if device.lower().startswith("cuda"):
                    estimate = estimate_torch_cuda_storage_mb(resource)
                    tracked_mb = float(estimate.get("total_mb", 0.0) or 0.0)
                    if tracked_mb > 0.0:
                        expected_process_reclaim_mb += tracked_mb
                    else:
                        untracked_gpu_resource_count += 1
            setattr(inpainter, attribute, None)
        return {
            "cached": True,
            "device": device,
            "loaded_native_resource_count": loaded_native_resource_count,
            "expected_process_reclaim_mb": expected_process_reclaim_mb,
            "untracked_gpu_resource_count": untracked_gpu_resource_count,
            "gpu_release_expected": (
                loaded_native_resource_count > 0
                and device.lower().startswith("cuda")
            ),
        }

    def release_inpainter_resources(
        self,
        *,
        vram_timeout_sec: float = DEFAULT_VRAM_RELEASE_TIMEOUT_SEC,
        vram_poll_interval_sec: float = DEFAULT_VRAM_RELEASE_POLL_SEC,
        vram_min_drop_mb: float = DEFAULT_VRAM_RELEASE_MIN_DROP_MB,
    ) -> dict[str, Any]:
        """Drop only inpainter model/session caches and verify CUDA handoff."""

        before = query_cuda_handoff_metrics()
        cached_key = self.cached_inpainter_key
        cached_inpainter = self.inpainter_cache
        driver_baseline = self.inpainter_driver_baseline
        self.inpainter_cache = None
        self.cached_inpainter_key = None
        self.cached_inpainter_runtime_signature = None
        self.inpainter_driver_baseline = None
        self.inpainter_runtime_contract = {}

        handler_release = self._detach_inpainter_native_resources(cached_inpainter)
        source_release = release_source_lama_cache()
        gpu_release_expected = bool(
            handler_release["gpu_release_expected"]
            or source_release["gpu_release_expected"]
        )
        expected_process_reclaim_mb = float(
            handler_release.get("expected_process_reclaim_mb", 0.0) or 0.0
        ) + float(
            source_release.get("expected_process_reclaim_mb", 0.0) or 0.0
        )
        untracked_gpu_resource_count = int(
            handler_release.get("untracked_gpu_resource_count", 0) or 0
        ) + int(
            source_release.get("untracked_gpu_resource_count", 0) or 0
        )
        del cached_inpainter

        cleanup = cleanup_python_cuda_memory(
            release_cuda_allocator=gpu_release_expected,
        )
        vram_release_gate = wait_for_vram_release(
            before,
            gpu_release_expected=gpu_release_expected,
            expected_process_drop_mb=expected_process_reclaim_mb,
            untracked_gpu_resource_count=untracked_gpu_resource_count,
            driver_baseline=driver_baseline,
            timeout_sec=vram_timeout_sec,
            poll_interval_sec=vram_poll_interval_sec,
            min_drop_mb=vram_min_drop_mb,
        )
        return {
            "cached_inpainter_key": str(cached_key or ""),
            "handler_release": handler_release,
            "source_lama_release": source_release,
            "python_native_cleanup": cleanup,
            "gpu_release_expected": gpu_release_expected,
            "expected_process_reclaim_mb": expected_process_reclaim_mb,
            "untracked_gpu_resource_count": untracked_gpu_resource_count,
            "vram_release_gate": vram_release_gate,
        }

    def manual_inpaint(self):
        image_viewer = self.main_page.image_viewer
        settings_page = self.main_page.settings_page
        mask = image_viewer.get_mask_for_inpainting()
        
        # Handle webtoon mode vs regular mode differently
        if self.main_page.webtoon_mode:
            # In webtoon mode, use visible area image for inpainting
            image, mappings = image_viewer.get_visible_area_image()
        else:
            # Regular mode - get the full image
            image = image_viewer.get_image_array()

        if image is None or mask is None:
            return None

        self._ensure_inpainter()
        config = get_config(settings_page)
        inpaint_input_img = self.inpainter_cache(image, mask, config)
        inpaint_input_img = imk.convert_scale_abs(inpaint_input_img) 

        return inpaint_input_img

    def inpaint_with_blocks(self, image: np.ndarray, mask: np.ndarray, blk_list, config=None):
        self.last_inpaint_edit_mask = None
        self.last_inpaint_diagnostics = {}
        if image is None or mask is None:
            return None
        self._ensure_inpainter()
        if config is None:
            config = get_config(self.main_page.settings_page)
        blocks = list(blk_list or [])
        device = str(
            getattr(
                self.inpainter_cache,
                "runtime_device",
                getattr(self.inpainter_cache, "device", ""),
            )
            or ""
        )
        precision = str(
            getattr(self.inpainter_cache, "precision", "fp32") or "fp32"
        )
        diagnostics = {
            **dict(self.inpainter_runtime_contract or {}),
            **validate_learned_inpaint_runtime(
                inpainter_key=str(self.cached_inpainter_key or ""),
                device=device,
                precision=precision,
            ),
            **runtime_mask_diagnostics(mask, image.shape),
            "inpaint_size": int(
                getattr(self.inpainter_cache, "inpaint_size", 0) or 0
            ),
            "retry_policy": INPAINT_RETRY_POLICY_VERSION,
            "oom_retry_count": 0,
            "oom_retry_roi": None,
            "model_call_diagnostics": [],
            "status": "running",
        }
        try:
            result, edit_mask, model_diagnostics = (
                source_lama_blockwise_inpaint(
                    image,
                    mask,
                    blocks,
                    self.inpainter_cache,
                    config,
                    check_need_inpaint=True,
                    return_edit_mask=True,
                    return_diagnostics=True,
                )
            )
            diagnostics["model_call_diagnostics"] = model_diagnostics
            diagnostics["oom_retry_count"] = sum(
                int(item.get("oom_retry_count", 0) or 0)
                for item in model_diagnostics
            )
        except InpaintingCudaOOMError as exc:
            diagnostics["status"] = "failed_after_model_retry"
            model_failure = dict(
                getattr(exc, "diagnostics", {}) or {}
            )
            if model_failure:
                diagnostics["model_call_diagnostics"] = [model_failure]
                diagnostics["oom_retry_count"] = int(
                    model_failure.get("oom_retry_count", 0) or 0
                )
            self.last_inpaint_diagnostics = diagnostics
            raise
        except Exception as exc:
            if not is_cuda_oom_error(exc):
                diagnostics["status"] = "failed"
                diagnostics["error_type"] = type(exc).__name__
                self.last_inpaint_diagnostics = diagnostics
                raise
            retry_roi = bounded_retry_roi(mask, image.shape)
            diagnostics["oom_retry_count"] = 1
            diagnostics["oom_retry_roi"] = (
                retry_roi.as_list() if retry_roi is not None else None
            )
            diagnostics["error_type"] = type(exc).__name__
            if retry_roi is None:
                diagnostics["status"] = "failed_no_smaller_roi"
                self.last_inpaint_diagnostics = diagnostics
                raise InpaintingCudaOOMError(
                    inpaint_cuda_oom_message(),
                    diagnostics=diagnostics,
                ) from exc
            try:
                import torch

                torch.cuda.empty_cache()
                x1, y1, x2, y2 = retry_roi.as_list()
                retry_image = np.ascontiguousarray(
                    image[y1:y2, x1:x2]
                )
                retry_mask = np.ascontiguousarray(mask[y1:y2, x1:x2])
                retry_result = self.inpainter_cache(
                    retry_image,
                    retry_mask,
                    config,
                )
            except Exception as retry_exc:
                diagnostics["status"] = "failed_after_roi_retry"
                diagnostics["retry_error_type"] = type(retry_exc).__name__
                self.last_inpaint_diagnostics = diagnostics
                raise InpaintingCudaOOMError(
                    inpaint_cuda_oom_message(),
                    diagnostics=diagnostics,
                ) from retry_exc
            result = np.asarray(image).copy()
            result[y1:y2, x1:x2] = composite_with_edit_mask(
                retry_image,
                retry_result,
                retry_mask,
            )
            edit_mask = normalize_edit_mask(mask, image.shape)
            diagnostics["status"] = "completed_after_roi_retry"
        else:
            diagnostics["status"] = "completed"
        result = composite_with_edit_mask(image, result, edit_mask)
        diagnostics["outside_mask_changed_pixel_count"] = (
            count_changed_outside_edit_mask(image, result, edit_mask)
        )
        self.last_inpaint_edit_mask = edit_mask
        self.last_inpaint_diagnostics = diagnostics
        return result

    def _qimage_to_np(self, qimg: QImage):
        if qimg.width() <= 0 or qimg.height() <= 0:
            return np.zeros((max(1, qimg.height()), max(1, qimg.width())), dtype=np.uint8)
        ptr = qimg.constBits()
        arr = np.array(ptr).reshape(qimg.height(), qimg.bytesPerLine())
        return arr[:, :qimg.width()]

    def _generate_mask_from_saved_strokes(
        self,
        strokes: list[dict],
        image: np.ndarray,
        *,
        base_mask: np.ndarray | None = None,
    ):
        if image is None or (not strokes and base_mask is None):
            return None
        height, width = image.shape[:2]
        if width <= 0 or height <= 0:
            return None

        add_qimg = QImage(width, height, QImage.Format_Grayscale8)
        gen_qimg = QImage(width, height, QImage.Format_Grayscale8)
        exclude_qimg = QImage(width, height, QImage.Format_Grayscale8)
        add_qimg.fill(0)
        gen_qimg.fill(0)
        exclude_qimg.fill(0)

        add_painter = QPainter(add_qimg)
        gen_painter = QPainter(gen_qimg)
        exclude_painter = QPainter(exclude_qimg)

        add_painter.setBrush(QBrush(QColor(255, 255, 255)))
        gen_painter.setBrush(QBrush(QColor(255, 255, 255)))
        exclude_painter.setBrush(QBrush(QColor(255, 255, 255)))
        gen_painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

        has_any = False
        for stroke in strokes:
            path = stroke.get('path')
            if path is None:
                continue

            role = normalize_stroke_role(stroke.get('role'), brush=stroke.get('brush'))
            if role == STROKE_ROLE_GENERATED:
                gen_painter.drawPath(path)
                has_any = True
                continue

            if role not in {STROKE_ROLE_ADD, STROKE_ROLE_EXCLUDE}:
                continue
            width_px = max(1, int(stroke.get('width', 25)))
            painter = exclude_painter if role == STROKE_ROLE_EXCLUDE else add_painter
            painter.setPen(QPen(QColor(255, 255, 255), width_px, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawPath(path)
            has_any = True

        add_painter.end()
        gen_painter.end()
        exclude_painter.end()

        if not has_any:
            return None

        add_mask = self._qimage_to_np(add_qimg)
        gen_mask = self._qimage_to_np(gen_qimg)
        exclude_mask = self._qimage_to_np(exclude_qimg)
        kernel = np.ones((5, 5), np.uint8)
        add_mask = imk.dilate(add_mask, kernel, iterations=2)
        gen_mask = imk.dilate(gen_mask, kernel, iterations=3)
        exclude_mask = imk.dilate(exclude_mask, kernel, iterations=2)
        if base_mask is None:
            base = np.zeros((height, width), dtype=np.uint8)
        else:
            base = np.asarray(base_mask)
            if base.ndim == 3:
                base = base[:, :, 0]
            if base.shape != (height, width):
                raise ValueError(
                    "Saved-stroke base mask does not match the image."
                )
        include_mask = np.where(
            (base > 0) | (add_mask > 0) | (gen_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        if np.count_nonzero(include_mask) == 0:
            return None
        final_mask = np.where((include_mask > 0) & ~(exclude_mask > 0), 255, 0).astype(np.uint8)
        if np.count_nonzero(final_mask) == 0:
            return None
        return final_mask

    def _get_regular_patches(self, mask: np.ndarray, inpainted_image: np.ndarray):
        contours, _ = imk.find_contours(mask)
        patches = []
        for c in contours:
            x, y, w, h = imk.bounding_rect(c)
            patch = inpainted_image[y:y + h, x:x + w]
            patches.append({
                'bbox': [x, y, w, h],
                'image': patch.copy(),
                'kind': PATCH_KIND_INPAINT,
            })
        return patches

    def extract_patches_from_image(
        self,
        mask: np.ndarray | None,
        image: np.ndarray | None,
        *,
        kind: str = PATCH_KIND_INPAINT,
    ) -> list[dict]:
        if mask is None or image is None or np.count_nonzero(mask) == 0:
            return []

        patches = self._get_regular_patches(mask, image)
        normalized_kind = normalize_patch_kind(kind)
        for patch in patches:
            patch['kind'] = normalized_kind
        return patches

    def inpaint_page_from_saved_strokes(self, image: np.ndarray, strokes: list[dict]):
        mask = self._generate_mask_from_saved_strokes(strokes, image)
        if mask is None:
            return []
        self._ensure_inpainter()
        config = get_config(self.main_page.settings_page)
        inpainted = self.inpainter_cache(image, mask, config)
        inpainted = imk.convert_scale_abs(inpainted)
        return self._get_regular_patches(mask, inpainted)

    def inpaint_complete(self, patch_list):
        patch_list = list(patch_list or [])
        # Handle webtoon mode vs regular mode
        if self.main_page.webtoon_mode:
            # In webtoon mode, group patches by page and apply them
            patches_by_page = {}
            for patch in patch_list:
                if 'page_index' in patch and 'file_path' in patch:
                    file_path = patch['file_path']
                    
                    if file_path not in patches_by_page:
                        patches_by_page[file_path] = []
                    
                    # Remove page-specific keys for the patch command but keep scene_pos for webtoon mode
                    clean_patch = {
                        'bbox': patch['bbox'],
                        'image': patch['image'],
                        'kind': normalize_patch_kind(patch.get('kind', PATCH_KIND_INPAINT)),
                    }
                    # Add scene position info for webtoon mode positioning
                    if 'scene_pos' in patch:
                        clean_patch['scene_pos'] = patch['scene_pos']
                        clean_patch['page_index'] = patch['page_index']
                    patches_by_page[file_path].append(clean_patch)
            
            # Apply patches to each page
            for file_path, patches in patches_by_page.items():
                self.main_page.image_ctrl.on_inpaint_patches_processed(patches, file_path)
        else:
            # Regular mode - original behavior
            self.main_page.apply_inpaint_patches(patch_list)
        
        self.main_page.image_viewer.clear_brush_strokes() 
        self.main_page.undo_group.activeStack().endMacro()  
        # get_best_render_area(self.main_page.blk_list, original_image, inpainted)    

    def get_inpainted_patches(self, mask: np.ndarray, inpainted_image: np.ndarray):
        # slice mask into bounding boxes
        contours, _ = imk.find_contours(mask)
        patches = []
        # Handle webtoon mode vs regular mode
        if self.main_page.webtoon_mode:
            # In webtoon mode, we need to map patches back to their respective pages
            visible_image, mappings = self.main_page.image_viewer.get_visible_area_image()
            if visible_image is None or not mappings:
                return patches
                
            for i, c in enumerate(contours):
                x, y, w, h = imk.bounding_rect(c)
                patch_bottom = y + h

                # Find all pages that this patch overlaps with
                overlapping_mappings = []
                for mapping in mappings:
                    if (y < mapping['combined_y_end'] and patch_bottom > mapping['combined_y_start']):
                        overlapping_mappings.append(mapping)
                
                if not overlapping_mappings:
                    continue
                    
                # If patch spans multiple pages, clip and redistribute
                for mapping in overlapping_mappings:
                    # Calculate the intersection with this page
                    clip_top = max(y, mapping['combined_y_start'])
                    clip_bottom = min(patch_bottom, mapping['combined_y_end'])
                    
                    if clip_bottom <= clip_top:
                        continue
                        
                    # Extract the portion of the patch for this page
                    clipped_patch = inpainted_image[clip_top:clip_bottom, x:x+w]
                    
                    # Convert coordinates back to page-local coordinates
                    page_local_y = clip_top - mapping['combined_y_start'] + mapping['page_crop_top']
                    clipped_height = clip_bottom - clip_top
                    
                    # Calculate the correct scene position by converting from visible area coordinates to scene coordinates
                    scene_y = mapping['scene_y_start'] + (clip_top - mapping['combined_y_start'])
                    
                    patches.append({
                        'bbox': [x, int(page_local_y), w, clipped_height],
                        'image': clipped_patch.copy(),
                        'kind': PATCH_KIND_INPAINT,
                        'page_index': mapping['page_index'],
                        'file_path': self.main_page.image_files[mapping['page_index']],
                        'scene_pos': [x, scene_y]  # Store correct scene position for webtoon mode
                    })
        else:
            # Regular mode - original behavior
            for c in contours:
                x, y, w, h = imk.bounding_rect(c)
                patch = inpainted_image[y:y+h, x:x+w]
                patches.append({
                    'bbox': [x, y, w, h],
                    'image': patch.copy(),
                })
                
        return patches
    
    def inpaint(self):
        mask = self.main_page.image_viewer.get_mask_for_inpainting()
        if mask is None or np.count_nonzero(mask) == 0:
            return []
        painted = self.manual_inpaint()
        if painted is None:
            return []
        patches = self.get_inpainted_patches(mask, painted)
        return patches         
