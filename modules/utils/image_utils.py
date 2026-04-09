import numpy as np
import base64
import imkit as imk
from PySide6.QtGui import QColor

from modules.utils.textblock import TextBlock
from modules.utils.inpaint_quality import build_inpaint_mask_bundle

def rgba2hex(rgba_list):
    r,g,b,a = [int(num) for num in rgba_list]
    return "#{:02x}{:02x}{:02x}{:02x}".format(r, g, b, a)

def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def get_smart_text_color(detected_rgb: tuple, setting_color: QColor) -> QColor:
    """
    Determines the best text color to use based on the detected color from the image
    and the user's preferred setting color.

    Policy:
      - If detection succeeded, use the detected colour (it came from
        actual pixel analysis and is most likely correct).
      - If detection is empty / invalid, fall back to the user setting.
    """
    if not detected_rgb:
        return setting_color

    try:
        detected_color = QColor(*detected_rgb)
        if not detected_color.isValid():
            return setting_color

        return detected_color

    except Exception:
        pass

    return setting_color

def generate_mask(img: np.ndarray, blk_list: list[TextBlock], default_padding: int = 5) -> np.ndarray:
    return build_inpaint_mask_bundle(img, blk_list, default_padding=default_padding).mask
