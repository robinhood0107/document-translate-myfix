from __future__ import annotations

MIT48_RELEASE_TAG = "beta-0.3"
MIT48_UPSTREAM_COMMIT = "a9ef04da0adda877413afe40b7656aebe22dc4be"
MIT48_UPSTREAM_LICENSE_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
MIT48_XPOS_LICENSE_COMMIT = "47a4c0adc0f9328ff72f241e7067b4d1a95401cd"
MIT48_XPOS_LICENSE_SHA256 = "c2cfccb812fe482101a8f04597dfc5a9991a6b2748266c47ac91b6a5aae15383"

MIT48_AR_CHECKPOINT = "ocr_ar_48px.ckpt"
MIT48_AR_CHECKPOINT_SHA256 = "29daa46d080818bb4ab239a518a88338cbccff8f901bef8c9db191a7cb97671d"
MIT48_AR_DICTIONARY = "alphabet-all-v7.txt"
MIT48_AR_DICTIONARY_SHA256 = "f5722368146aa0fbcc9f4726866e4efc3203318ebb66c811d8cbbe915576538a"

MIT48_CTC_ARCHIVE = "ocr-ctc.zip"
MIT48_CTC_ARCHIVE_SHA256 = "fc61c52f7a811bc72c54f6be85df814c6b60f63585175db27cb94a08e0c30101"
MIT48_CTC_CHECKPOINT = "ocr-ctc.ckpt"
MIT48_CTC_CHECKPOINT_SHA256 = "8b0837a24da5fde96c23ca47bb7abd590cd5b185c307e348c6e0b7238178ed89"
MIT48_CTC_DICTIONARY = "alphabet-all-v5.txt"
MIT48_CTC_DICTIONARY_SHA256 = "c1295ae1962e69e35b5b225a0405d1f3432e368c9941d23bfd3acda12654da33"
MIT48_CTC_EXTRACT_DIRECTORY = MIT48_CTC_ARCHIVE_SHA256[:16]

MIT48_MODEL_SOURCE_SHA256 = {
    "model_48px.py": "6f5a8aee4af67223f2b11a9421dd873a18ce78d2deebd6a75be81b5db18ae453",
    "model_48px_ctc.py": "dcb3d7e830f9fb188e7f0b10b6a7bd2713ad36df205706dbb86dfff29cdb05d6",
    "xpos_relative_position.py": "27420e6068ab687a084878a4be6c59a245290927fd51057dc9f65838e24e694f",
    "LICENSE-GPL-3.0.txt": MIT48_UPSTREAM_LICENSE_SHA256,
    "LICENSE-MIT-XPOS.txt": MIT48_XPOS_LICENSE_SHA256,
}

MIT48_MODEL_SOURCE_URLS = {
    "model_48px.py": (
        "https://raw.githubusercontent.com/zyddnys/manga-image-translator/"
        f"{MIT48_UPSTREAM_COMMIT}/manga_translator/ocr/model_48px.py"
    ),
    "model_48px_ctc.py": (
        "https://raw.githubusercontent.com/zyddnys/manga-image-translator/"
        f"{MIT48_UPSTREAM_COMMIT}/manga_translator/ocr/model_48px_ctc.py"
    ),
    "xpos_relative_position.py": (
        "https://raw.githubusercontent.com/zyddnys/manga-image-translator/"
        f"{MIT48_UPSTREAM_COMMIT}/manga_translator/ocr/xpos_relative_position.py"
    ),
    "LICENSE-GPL-3.0.txt": (
        "https://raw.githubusercontent.com/zyddnys/manga-image-translator/"
        f"{MIT48_UPSTREAM_COMMIT}/LICENSE"
    ),
    "LICENSE-MIT-XPOS.txt": (
        "https://raw.githubusercontent.com/microsoft/torchscale/"
        f"{MIT48_XPOS_LICENSE_COMMIT}/LICENSE"
    ),
}

MIT48_AR_MODEL_IDENTITY = (
    f"mit48-ar:{MIT48_UPSTREAM_COMMIT}:{MIT48_AR_CHECKPOINT_SHA256}:"
    f"{MIT48_AR_DICTIONARY_SHA256}"
)
MIT48_CTC_MODEL_IDENTITY = (
    f"mit48-ctc:{MIT48_UPSTREAM_COMMIT}:{MIT48_CTC_CHECKPOINT_SHA256}:"
    f"{MIT48_CTC_DICTIONARY_SHA256}"
)
MIT48_RUNTIME_IDENTITY = (
    f"mit48-worker-v2:{MIT48_UPSTREAM_COMMIT}:"
    f"{MIT48_MODEL_SOURCE_SHA256['model_48px.py']}:"
    f"{MIT48_MODEL_SOURCE_SHA256['model_48px_ctc.py']}:"
    f"{MIT48_MODEL_SOURCE_SHA256['xpos_relative_position.py']}"
)
