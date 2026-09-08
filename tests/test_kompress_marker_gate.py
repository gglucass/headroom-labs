"""The CCR retrieval marker follows the saving, and the saving must cover it.

A lossy Kompress result that shipped without a marker was discarded by the
router's lossy-unrecoverable guard (#1307), so every pass that shrank a block
by less than 20 percent ran the model and saved nothing. The gate now asks
whether the saving pays for the marker in tokens, not words: the marker is 12
words but 36-38 cl100k tokens, and the words Kompress drops can be single
tokens. A result that cannot pay passes through; a marked result reports the
marker in its own accounting.
"""

from __future__ import annotations

import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import (
    CCR_MARKER_COST_WORDS,
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


HASH = "c00eb437e5e5c00eb437e5e5"  # real CCR hash length


def _compressor(monkeypatch, **config) -> KompressCompressor:
    compressor = KompressCompressor(KompressConfig(min_input_words=10, **config))
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    monkeypatch.setattr(compressor, "_store_in_ccr", lambda *a, **k: HASH)
    return compressor


def test_marker_cost_bound_covers_a_real_marker_in_tokens():
    enc = pytest.importorskip("tiktoken").get_encoding("cl100k_base")
    marker = ccr_retrieval_marker(12345, 9876, "x\n" * 400, HASH)
    # Every dropped word is at least one token, so a bound in words covers the
    # marker whenever it exceeds the marker's token count.
    assert len(enc.encode(marker)) < CCR_MARKER_COST_WORDS


def test_weak_shrink_gets_a_marker_and_reports_it(monkeypatch):
    # 300 -> 240 words is ratio 0.80: the old `ratio < 0.8` gate shipped this
    # unmarked and the router threw it away.
    _install(monkeypatch, drop=60)
    result = _compressor(monkeypatch).compress(_prose(300))
    assert result.cache_key == HASH
    assert f"Retrieve more: hash={HASH}" in result.compressed
    assert result.compressed_tokens == 240 + CCR_MARKER_COST_WORDS
    assert result.compression_ratio == (240 + CCR_MARKER_COST_WORDS) / 300


def test_saving_that_cannot_pay_for_the_marker_passes_through(monkeypatch):
    # The reviewer's case: drop 16 of 100 words and append a real-length
    # marker, and cl100k goes 196 -> 200. Under the 13-word gate this shipped
    # as ratio 0.84.
    _install(monkeypatch, drop=16)
    prose = _prose(100)
    result = _compressor(monkeypatch).compress(prose)
    assert result.compressed == prose
    assert result.cache_key is None
    assert result.compressed_tokens == 100
    assert result.compression_ratio == 1.0
    enc = pytest.importorskip("tiktoken").get_encoding("cl100k_base")
    marked = " ".join(prose.split()[16:]) + ccr_retrieval_marker(100, 84, prose, HASH)
    assert len(enc.encode(marked)) > len(enc.encode(prose))


def test_batch_path_uses_the_same_gate_and_accounting(monkeypatch):
    _install(monkeypatch, drop=60)
    [marked] = _compressor(monkeypatch).compress_batch([_prose(300)], batch_size=8)
    assert f"Retrieve more: hash={HASH}" in marked.compressed
    assert marked.compressed_tokens == 240 + CCR_MARKER_COST_WORDS
    _install(monkeypatch, drop=16)
    prose = _prose(100)
    [through] = _compressor(monkeypatch).compress_batch([prose], batch_size=8)
    assert through.compressed == prose
    assert through.compression_ratio == 1.0


def test_no_ccr_mode_ships_unmarked_lossy_unchanged(monkeypatch):
    # Without CCR there is no marker to pay for: the deliberate output is the
    # bare lossy result, on both paths, exactly as before.
    _install(monkeypatch, drop=16)
    compressor = _compressor(monkeypatch, enable_ccr=False)
    result = compressor.compress(_prose(100))
    assert result.compressed_tokens == 84
    assert "Retrieve more" not in result.compressed
    [batched] = compressor.compress_batch([_prose(100)], batch_size=8)
    assert batched.compressed_tokens == 84
