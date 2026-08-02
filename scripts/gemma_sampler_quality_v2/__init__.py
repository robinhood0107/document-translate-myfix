"""Private-artifact Gemma sampler quality v2 benchmark primitives.

The package intentionally keeps corpus text, model replies, and judgments out
of tracked files.  Only protocol definitions and sanitized summaries belong in
the repository.
"""

from .protocol import (
    CORPUS_CASE_COUNT,
    HOLDOUT_CASE_COUNT,
    PINNED_LLAMA_CPP_COMMIT,
    PINNED_LLAMA_CPP_IMAGE,
    TUNING_CASE_COUNT,
    SamplerTuple,
)

__all__ = [
    "CORPUS_CASE_COUNT",
    "HOLDOUT_CASE_COUNT",
    "PINNED_LLAMA_CPP_COMMIT",
    "PINNED_LLAMA_CPP_IMAGE",
    "TUNING_CASE_COUNT",
    "SamplerTuple",
]
