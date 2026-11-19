import hashlib

import numpy as np


def decoded_image_sha256(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii", errors="strict"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(int(item)) for item in contiguous.shape).encode(
            "ascii",
            errors="strict",
        )
    )
    digest.update(b"\0")
    digest.update(contiguous)
    return digest.hexdigest()
