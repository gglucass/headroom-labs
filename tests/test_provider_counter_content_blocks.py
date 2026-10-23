"""Provider token counters must price every content block, not just ``text``.

Each counter in ``headroom/providers/`` had grown its own shortened content-block
walker handling only the shapes its provider was expected to send. Everything else
fell through and contributed nothing. Measured on one 6,800-char block
(``count_messages`` of a single-block message, so 7-8 is message overhead alone):

    block type          OpenAI ctr   Anthropic ctr
    text (control)            3409            3748
    tool_result                  8            3748
    thinking                     8               7
    document                     8               7
    mcp_tool_result              8               7
    output_text                  8               7
    refusal                      8               7

Note each counter zeroed blocks from its OWN provider — ``output_text`` and
``refusal`` are OpenAI Responses shapes, ``thinking`` and ``document`` are
Anthropic's. And these counters are what the LIVE proxy pipelines use
(``proxy/server.py`` builds them with ``AnthropicProvider`` / ``OpenAIProvider``),
so this was the main request path, not an edge case.

The counters now delegate to ``count_content_blocks``, which reuses
``BaseTokenizer._count_content_parts``. That matters beyond coverage: a naive
``count_text(str(block))`` catch-all would serialize a base64 image and price it
as text — a 1MB screenshot reads as ~330K phantom tokens. The shared walker gives
media a pixel/byte-based estimate instead.
"""

from __future__ import annotations

import base64
import zlib

import pytest

from headroom.providers.anthropic import AnthropicProvider
from headroom.providers.openai import OpenAIProvider
from headroom.tokenizers.base import count_content_blocks

_BIG = "x " * 3400  # ~6,800 chars


def _counters():
    return {
        "openai": OpenAIProvider().get_token_counter("gpt-4o"),
        "anthropic": AnthropicProvider(warn=False).get_token_counter("claude-sonnet-4-6"),
    }


@pytest.mark.parametrize(
    "block",
    [
        {"type": "tool_result", "tool_use_id": "t", "content": _BIG},
        {"type": "tool_use", "id": "t", "name": "grep", "input": {"pattern": _BIG}},
        {"type": "thinking", "thinking": _BIG},
        {"type": "document", "source": {"data": _BIG}},
        {"type": "mcp_tool_result", "content": _BIG},
        {"type": "output_text", "text": _BIG},
        {"type": "refusal", "refusal": _BIG},
        {"type": "search_result", "content": _BIG},
    ],
    ids=lambda b: str(b.get("type")),
)
def test_no_provider_counter_prices_a_large_block_at_zero(block: dict) -> None:
    """Every one of these returned 7-8 tokens — message overhead only."""
    message = {"role": "user", "content": [block]}
    for name, counter in _counters().items():
        got = counter.count_messages([message])
        assert got > 1_000, f"{name} priced a ~6,800-char {block['type']} block at {got}"


def test_base64_media_is_not_priced_as_text() -> None:
    """The reason a str(block) catch-all would have been the wrong fix."""
    image = {"type": "image", "source": {"type": "base64", "data": "A" * 200_000}}
    message = {"role": "user", "content": [image]}

    for name, counter in _counters().items():
        got = counter.count_messages([message])
        # 200KB of base64 as text would be ~50,000 tokens; the pixel estimate is 1600.
        assert got < 5_000, f"{name} priced a 200KB base64 image as text: {got}"
        assert got > 1_000, f"{name} ignored a declared image entirely: {got}"


def test_plain_text_blocks_are_unchanged() -> None:
    """The control: the shape both counters already handled must not move."""
    message = {"role": "user", "content": [{"type": "text", "text": _BIG}]}
    for name, counter in _counters().items():
        text_form = {"role": "user", "content": _BIG}
        block_form = counter.count_messages([message])
        # A text block and the equivalent string should agree closely.
        assert abs(block_form - counter.count_messages([text_form])) <= 5, name


def test_shared_walker_ignores_non_block_parts() -> None:
    """A bare int is not a block and must contribute nothing."""
    assert count_content_blocks([123], len) == 0
    assert count_content_blocks([], len) == 0


def test_shared_walker_counts_nested_tool_result_blocks() -> None:
    """A tool that returns blocks nests them; they must be walked, not serialized."""
    nested = {
        "type": "tool_result",
        "tool_use_id": "t",
        "content": [{"type": "text", "text": _BIG}],
    }
    flat = {"type": "tool_result", "tool_use_id": "t", "content": _BIG}
    assert abs(count_content_blocks([nested], len) - count_content_blocks([flat], len)) < 100


def _pdf(page_count: int, *, tree: bool = True, pad: int = 1_000_000) -> bytes:
    """A PDF-shaped blob: its page tree (or only page objects), then opaque bytes."""
    pages = b"".join(
        b"%d 0 obj << /Type /Page /Parent 2 0 R >> endobj\n" % (3 + i) for i in range(page_count)
    )
    root = b"2 0 obj << /Type /Pages /Count %d >> endobj\n" % page_count if tree else b""
    return (
        b"%PDF-1.7\n1 0 obj << /Type /Catalog >> endobj\n"
        + root
        + pages
        + bytes(range(256)) * (pad // 256)
    )


def _base64_document(raw: bytes) -> dict:
    data = base64.b64encode(raw).decode()
    return {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": data},
    }


def test_base64_pdf_document_is_priced_by_pages_not_as_text() -> None:
    """A 1 MB base64 PDF read as prose was ~907K tokens; 12 pages are ~18K."""
    document = _base64_document(_pdf(12))
    for placement in (
        [document],
        [{"type": "tool_result", "tool_use_id": "t", "content": [document]}],
    ):
        assert count_content_blocks(placement, len) == 12 * 1500
        for name, counter in _counters().items():
            got = counter.count_messages([{"role": "user", "content": placement}])
            assert 12 * 1500 <= got < 12 * 1500 + 100, f"{name} priced a 12-page PDF at {got}"


def test_pdf_page_count_falls_back_from_tree_to_page_objects_to_size() -> None:
    assert count_content_blocks([_base64_document(_pdf(5, tree=False))], len) == 5 * 1500
    # PDF 1.5+ keeps the page tree inside a compressed object stream.
    packed = zlib.compress(b"<< /Type /Pages /Kids [3 0 R] /Count 7 >>")
    hidden = (
        b"%PDF-1.7\n4 0 obj << /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode >>\nstream\n"
        + packed
        + b"\nendstream\nendobj\n"
        + bytes(range(256)) * 4096
    )
    assert count_content_blocks([_base64_document(hidden)], len) == 7 * 1500
    # Page tree hidden (compressed object streams): ~3KB of file per page.
    opaque = bytes(range(256)) * 4096
    assert count_content_blocks([_base64_document(opaque)], len) == len(opaque) // 3000 * 1500
    # Undecodable base64 is still a document: one page, no exception.
    broken = {"type": "document", "source": {"type": "base64", "data": "not base64!"}}
    assert count_content_blocks([broken], len) == 1500


def test_text_and_content_document_sources_count_their_text() -> None:
    text = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": _BIG},
    }
    assert count_content_blocks([text], len) == len(_BIG)
    content = {
        "type": "document",
        "source": {"type": "content", "content": [{"type": "text", "text": _BIG}]},
    }
    assert count_content_blocks([content], len) == len(_BIG)
    titled = {**text, "title": "Q3 report", "context": "board deck"}
    assert count_content_blocks([titled], len) == len(_BIG) + len("Q3 report") + len("board deck")


def test_url_and_file_document_sources_are_one_nominal_page() -> None:
    for source in (
        {"type": "url", "url": "https://example.com/a.pdf"},
        {"type": "file", "file_id": "file_1"},
    ):
        assert count_content_blocks([{"type": "document", "source": source}], len) == 1500
