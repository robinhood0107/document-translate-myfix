from __future__ import annotations

import unittest

from modules.utils.llama_cpp_runtime import (
    DEFAULT_LLAMA_CPP_IMAGE,
    normalize_llama_cpp_image,
)


class LlamaCppRuntimePolicyTests(unittest.TestCase):
    def test_digest_pinned_image_is_preserved(self) -> None:
        pinned = (
            "ghcr.io/ggml-org/llama.cpp@sha256:"
            "22e0e3bfe967af4fd1df6a918022abbfd4e72e4d40a4769e616a4176790acbcb"
        )
        self.assertEqual(normalize_llama_cpp_image(pinned), pinned)

    def test_mutable_llama_cpp_tags_still_normalize_to_repository_default(self) -> None:
        self.assertEqual(
            normalize_llama_cpp_image("ghcr.io/ggml-org/llama.cpp:server-cuda"),
            DEFAULT_LLAMA_CPP_IMAGE,
        )


if __name__ == "__main__":
    unittest.main()
