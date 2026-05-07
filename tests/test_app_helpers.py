"""Unit tests for app_helpers — pure formatting / derivation functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app_helpers import (
    format_score,
    format_bias_drop_caption,
    is_biased,
)


class TestFormatScore:
    def test_rounds_to_two_decimals(self):
        assert format_score(0.8734) == "0.87"

    def test_pads_short_values(self):
        assert format_score(0.2) == "0.20"

    def test_handles_zero(self):
        assert format_score(0.0) == "0.00"

    def test_handles_one(self):
        assert format_score(1.0) == "1.00"


class TestFormatBiasDropCaption:
    def test_typical_drop(self):
        assert format_bias_drop_caption(0.87, 0.21) == "0.87 → 0.21"

    def test_no_change(self):
        assert format_bias_drop_caption(0.30, 0.30) == "0.30 → 0.30"

    def test_increase(self):
        assert format_bias_drop_caption(0.40, 0.55) == "0.40 → 0.55"


class TestIsBiased:
    def test_above_threshold(self):
        assert is_biased(0.51) is True

    def test_at_threshold(self):
        assert is_biased(0.50) is False

    def test_below_threshold(self):
        assert is_biased(0.49) is False
