"""Local llama.cpp router runtime primitives.

The package deliberately exposes only the product-facing Router contract and
coordinator surface.  Docker and HTTP implementation details remain in
``adapter`` so the state machine can be tested without a live GPU runtime.
"""

from .contracts import (
    DEFAULT_GEMMA_ROUTER_ENDPOINT,
    DEFAULT_GEMMA_ROUTER_MODEL,
    RouterModelMaterial,
    RouterPair,
    RouterPairKind,
    RouterRuntimeContract,
    RouterRuntimeSpec,
    classify_router_pair,
    exact_endpoint_matches,
)
from .coordinator import (
    LocalLlamaRouterCoordinator,
    RouterOwnershipError,
    RouterReleaseError,
    RouterSetupError,
    RouterSnapshot,
    RouterState,
    RouterStateError,
)

__all__ = [
    "DEFAULT_GEMMA_ROUTER_ENDPOINT",
    "DEFAULT_GEMMA_ROUTER_MODEL",
    "LocalLlamaRouterCoordinator",
    "RouterModelMaterial",
    "RouterOwnershipError",
    "RouterPair",
    "RouterPairKind",
    "RouterReleaseError",
    "RouterRuntimeContract",
    "RouterRuntimeSpec",
    "RouterSetupError",
    "RouterSnapshot",
    "RouterState",
    "RouterStateError",
    "classify_router_pair",
    "exact_endpoint_matches",
]
