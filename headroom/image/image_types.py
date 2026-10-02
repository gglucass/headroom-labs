"""Lightweight image-routing types shared across the image stack.

Kept dependency-free (pure enum + dataclasses, no torch / transformers / onnx)
so importing the image compressor or the ONNX router does not eagerly import
the heavy ML stack via ``trained_router``. On Python 3.13+ that eager import
crashed with ``AttributeError: module 'torch' has no attribute 'compiler'``
because ``transformers`` touched ``torch.compiler`` before torch finished
initializing inside the proxy process (#2513).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

_T = TypeVar("_T")


class ImageMemo(Generic[_T]):
    """Bounded LRU of per-image results, keyed on the image bytes' sha256.

    Every turn resends the whole conversation, so the same screenshots reach
    the image stack on every request; OCR and the SigLIP encoder are pure
    functions of the bytes, so each image pays for them once. A ``None`` (no
    confident text, or a failed call) is cached like any result.

    A transcode turn scans the history oldest-first, and a history larger than
    the memo makes LRU thrash: every miss evicts the image the scan reaches
    next, so nothing is reused. Entries are a 32-byte key and an OCR string
    (~1 KB) or four floats, so the bound sits far above any one history, a
    few MB at most, and only guards a long-lived worker across conversations.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        self._max_entries = max_entries
        self._entries: OrderedDict[bytes, _T] = OrderedDict()

    def __contains__(self, image_data: bytes) -> bool:
        return hashlib.sha256(image_data).digest() in self._entries

    def get(self, image_data: bytes, compute: Callable[[], _T]) -> _T:
        key = hashlib.sha256(image_data).digest()
        try:
            value = self._entries.pop(key)
        except KeyError:
            value = compute()
        self._entries[key] = value  # (re)inserted last: most recently used
        if len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)
        return value


class Technique(Enum):
    """Image optimization techniques."""

    TRANSCODE = "transcode"  # Convert to text description (99% savings)
    CROP = "crop"  # Extract relevant region (50-90% savings)
    PRESERVE = "preserve"  # Keep full quality (0% savings)
    FULL_LOW = "full_low"  # Full image, lower quality (87% savings)


@dataclass
class ImageSignals:
    """Signals extracted from image analysis."""

    has_text: float
    is_document: float
    is_complex: float
    has_small_details: float


@dataclass
class RouteDecision:
    """Result of routing decision."""

    technique: Technique
    confidence: float
    reason: str
    image_signals: ImageSignals | None = None
    query_prediction: str | None = None
    query_confidence: float | None = None
