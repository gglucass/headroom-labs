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

import hashlib

import pytest

from headroom.transforms import kompress_compressor as kc
from headroom.transforms.kompress_compressor import (
    KompressCompressor,
    KompressConfig,
    ccr_marker_cost,
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


def _real_key(source: str) -> str:
    """The CCR store's own key: the SHA-256 prefix of the stored source."""
    return hashlib.sha256(source.encode()).hexdigest()[:24]


def _compressor(monkeypatch, **config) -> KompressCompressor:
    compressor = KompressCompressor(KompressConfig(min_input_words=10, **config))
    monkeypatch.setattr(compressor, "_should_batch_single_content", lambda *a, **k: False)
    monkeypatch.setattr(compressor, "_should_use_sequential_fallback", lambda: False)
    monkeypatch.setattr(compressor, "_store_in_ccr", lambda source, *a, **k: _real_key(source))
    return compressor


# 100 single-token words: the tightest source there is. Dropping 41 saves 41
# tokens, and the marker for it (with its real source-derived hash) costs 43.
SINGLE_TOKEN_SOURCE = " ".join(["alpha"] * 99 + ["nfs"])


def test_marker_cost_is_the_real_marker_priced_in_tokens():
    enc = pytest.importorskip("tiktoken").get_encoding("cl100k_base")
    marker = ccr_retrieval_marker(100, 59, SINGLE_TOKEN_SOURCE, _real_key(SINGLE_TOKEN_SOURCE))
    assert _real_key(SINGLE_TOKEN_SOURCE) == "7dbb8f8de9f3e1d7c6f3a6e1"
    assert ccr_marker_cost(marker) == len(enc.encode(marker)) == 43
    # Digits and a different hash change the price; it is measured, not fixed.
    other = ccr_retrieval_marker(123456, 98765, "x\n" * 400, "c00eb437e5e5c00eb437e5e5")
    assert ccr_marker_cost(other) == len(enc.encode(other))


def test_marker_cost_without_an_encoder_is_one_token_per_character(monkeypatch):
    monkeypatch.setattr(kc, "_marker_encoder", False)
    marker = ccr_retrieval_marker(100, 59, SINGLE_TOKEN_SOURCE, _real_key(SINGLE_TOKEN_SOURCE))
    assert ccr_marker_cost(marker) == len(marker)


def test_single_token_words_that_cannot_pay_for_the_marker_pass_through(monkeypatch):
    # 100 -> 59 words saves 41 tokens; the marked payload is 102 tokens. Under
    # the earlier fixed 40-word allowance this shipped at reported ratio 0.99.
    _install(monkeypatch, drop=41)
    result = _compressor(monkeypatch).compress(SINGLE_TOKEN_SOURCE)
    assert result.compressed == SINGLE_TOKEN_SOURCE
    assert result.cache_key is None
    assert (result.original_tokens, result.compressed_tokens, result.compression_ratio) == (
        100,
        100,
        1.0,
    )
    [batched] = _compressor(monkeypatch).compress_batch([SINGLE_TOKEN_SOURCE], batch_size=8)
    assert batched.compressed == SINGLE_TOKEN_SOURCE and batched.compression_ratio == 1.0


def test_marked_result_reports_kept_words_plus_the_measured_marker(monkeypatch):
    # 300 single-token words, drop 60: 60 saved tokens cover a ~43-token marker.
    source = " ".join(["alpha"] * 299 + ["nfs"])
    _install(monkeypatch, drop=60)
    result = _compressor(monkeypatch).compress(source)
    marker = ccr_retrieval_marker(300, 240, source, _real_key(source))
    assert result.compressed == " ".join(source.split()[60:]) + marker
    assert result.cache_key == _real_key(source)
    assert result.compressed_tokens == 240 + ccr_marker_cost(marker)
    assert result.compression_ratio == (240 + ccr_marker_cost(marker)) / 300
    [batched] = _compressor(monkeypatch).compress_batch([source], batch_size=8)
    assert batched.compressed == result.compressed
    assert batched.compressed_tokens == result.compressed_tokens


def test_no_ccr_mode_ships_unmarked_lossy_unchanged(monkeypatch):
    # Without CCR there is no marker to pay for: the deliberate output is the
    # bare lossy result, on both paths, exactly as before.
    _install(monkeypatch, drop=41)
    compressor = _compressor(monkeypatch, enable_ccr=False)
    result = compressor.compress(SINGLE_TOKEN_SOURCE)
    assert result.compressed_tokens == 59
    assert "Retrieve more" not in result.compressed
    [batched] = compressor.compress_batch([SINGLE_TOKEN_SOURCE], batch_size=8)
    assert batched.compressed_tokens == 59
