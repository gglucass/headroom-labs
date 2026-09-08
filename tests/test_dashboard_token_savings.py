"""Regression tests for dashboard token savings copy."""

from __future__ import annotations

import re

from headroom.dashboard import get_dashboard_html


def _getter_body(html: str, name: str) -> str:
    match = re.search(rf"get {name}\(\) \{{(?P<body>.*?)\n\s*\}},", html, re.S)
    assert match is not None
    return match.group("body")


def test_token_savings_headline_leads_with_new_input_then_total_wire() -> None:
    html = get_dashboard_html()

    # New input first (the /stats field), whole-wire as the fallback when no
    # cache breakdown exists; the old attempted-denominator must stay gone.
    gate_body = _getter_body(html, "headlineUsesNewInput")
    assert "stats.tokens?.new_input_tokens" in gate_body
    headline_body = _getter_body(html, "headlineSavingsPercent")
    assert "stats.tokens?.new_input_savings_percent" in headline_body
    assert "stats.tokens?.savings_percent" in headline_body
    assert "stats.tokens?.proxy_savings_percent" in headline_body
    assert "stats.tokens?.active_savings_percent" not in headline_body
    assert "stats.tokens?.proxy_attempted_tokens" not in headline_body

    title_body = _getter_body(html, "headlineSavingsTitle")
    assert "Of new input tokens" in title_body
    assert "Of total wire input tokens" in title_body
    assert "Of compressible tokens attempted" not in title_body

    # The whole-wire figure is still shown beside it, labelled as such.
    assert "'Of total wire: '" in html
