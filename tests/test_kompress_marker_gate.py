"""The CCR retrieval marker follows the saving, not a flat ratio.

A lossy Kompress result that shipped without a marker was discarded by the
router's lossy-unrecoverable guard (#1307), so every pass that shrank a block
by less than 20 percent ran the model and saved nothing. The gate now asks
whether the saving pays for the marker.
"""

from __future__ import annotations

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import (
    CCR_MARKER_WORDS,
    KompressCompressor,
    KompressConfig,
    ccr_retrieval_marker,
)


class _Enc(dict):
    def word_ids(self, batch_index=0):
        return self["_word_ids"][batch_index]


class _Tok:
    def __call__(self, chunk_words, **kw):
        batch_words = (
            chunk_words if chunk_words and isinstance(chunk_words[0], list) else [chunk_words]
        )
        return _Enc(
            input_ids=[[0] * len(words) for words in batch_words],
            attention_mask=[[1] * len(words) for words in batch_words],
            _word_ids=[list(range(len(words))) for words in batch_words],
        )


class _DropFirst:
    """Keep every word except the first ``drop`` of each row."""

    def __init__(self, drop: int) -> None:
        self.drop = drop

    def get_keep_mask(self, input_ids, attention_mask):
        return [[idx >= self.drop for idx, _ in enumerate(row)] for row in input_ids]

    def get_scores(self, input_ids, attention_mask):
        return [[0.0 if idx < self.drop else 1.0 for idx, _ in enumerate(row)] for row in input_ids]


def _install(monkeypatch, drop: int) -> None:
    monkeypatch.setattr(kc, "_load_kompress", lambda *a, **k: (_DropFirst(drop), _Tok(), "onnx"))
    monkeypatch.setattr(kc, "_model_device_type", lambda *a, **k: "cpu")


def _prose(n: int) -> str:
    # Plain lowercase words: nothing the must-keep override would pin.
    return " ".join(f"w{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}" for i in range(n))


def _compressor(monkeypatch) -> KompressCompressor:
    compressor = KompressCompressor(KompressConfig(min_input_words=10))
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    monkeypatch.setattr(compressor, "_store_in_ccr", lambda *a, **k: "abc123")
    return compressor


def test_marker_word_cost_constant_matches_the_marker():
    marker = ccr_retrieval_marker(1000, 900, "x\n" * 5, "c00eb437e5e5c00eb437e5e5")
    assert len(marker.split()) <= CCR_MARKER_WORDS


def test_weak_shrink_still_gets_a_marker(monkeypatch):
    # 100 -> 80 words is ratio 0.80: the old `ratio < 0.8` gate shipped this
    # unmarked and the router threw it away.
    _install(monkeypatch, drop=20)
    result = _compressor(monkeypatch).compress(_prose(100))
    assert result.compressed_tokens == 80
    assert result.cache_key == "abc123"
    assert "Retrieve more: hash=abc123" in result.compressed


def test_saving_smaller_than_the_marker_stays_unmarked(monkeypatch):
    _install(monkeypatch, drop=5)
    result = _compressor(monkeypatch).compress(_prose(100))
    assert result.compressed_tokens == 95
    assert result.cache_key is None
    assert "Retrieve more" not in result.compressed


def test_batch_path_uses_the_same_gate(monkeypatch):
    _install(monkeypatch, drop=20)
    [result] = _compressor(monkeypatch).compress_batch([_prose(100)], batch_size=8)
    assert result.compressed_tokens == 80
    assert "Retrieve more: hash=abc123" in result.compressed
