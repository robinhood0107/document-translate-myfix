import numpy as np
import logging
import imkit as imk

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QBrush

from modules.utils.device import resolve_device
from modules.utils.inpaint_strokes import (
    PATCH_KIND_INPAINT,
    normalize_stroke_role,
    normalize_patch_kind,
    STROKE_ROLE_ADD,
    STROKE_ROLE_EXCLUDE,
    STROKE_ROLE_GENERATED,
)
from modules.utils.pipeline_config import inpaint_map, get_config, get_inpainter_runtime
from modules.inpainting.source_lama_blockwise import source_lama_blockwise_inpaint

logger = logging.getLogger(__name__)


class InpaintingHandler:
    """Handles image inpainting functionality."""
    
    def __init__(self, main_page):
        self.main_page = main_page
        self.inpainter_cache = None
        self.cached_inpainter_key = None

    def _ensure_inpainter(self):
        settings_page = self.main_page.settings_page
        runtime = get_inpainter_runtime(settings_page)
        inpainter_key = runtime['key']
        if self.inpainter_cache is None or self.cached_inpainter_key != inpainter_key:
            backend = runtime['backend']
            device = resolve_device(settings_page.is_gpu_enabled(), backend)
            InpainterClass = inpaint_map[inpainter_key]
            self.inpainter_cache = InpainterClass(
                device,
                backend=backend,
                runtime_device=runtime.get('device', device),
                inpaint_size=runtime.get('inpaint_size'),
                precision=runtime.get('precision'),
            )
            self.cached_inpainter_key = inpainter_key
        return self.inpainter_cache

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
        if image is None or mask is None:
            return None
        self._ensure_inpainter()
        if config is None:
            config = get_config(self.main_page.settings_page)
        blocks = list(blk_list or [])
        return source_lama_blockwise_inpaint(
            image,
            mask,
            blocks,
            self.inpainter_cache,
            config,
            check_need_inpaint=True,
        )

    def _qimage_to_np(self, qimg: QImage):
        if qimg.width() <= 0 or qimg.height() <= 0:
            return np.zeros((max(1, qimg.height()), max(1, qimg.width())), dtype=np.uint8)
        ptr = qimg.constBits()
        arr = np.array(ptr).reshape(qimg.height(), qimg.bytesPerLine())
        return arr[:, :qimg.width()]

    def _generate_mask_from_saved_strokes(self, strokes: list[dict], image: np.ndarray):
        if image is None or not strokes:
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
        include_mask = np.where((add_mask > 0) | (gen_mask > 0), 255, 0).astype(np.uint8)
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
