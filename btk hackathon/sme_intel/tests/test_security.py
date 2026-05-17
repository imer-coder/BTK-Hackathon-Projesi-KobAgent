"""
tests/test_security.py — Unit tests for DataMasker.

Run with:  pytest tests/test_security.py -v
"""

import pytest
from src.security import DataMasker, MaskingError


class TestDataMasker:
    """Tests for the bidirectional masking engine."""

    def setup_method(self) -> None:
        self.masker = DataMasker(prefix="CUSTOMER")

    # ── mask() ─────────────────────────────────────────────────────── #

    def test_mask_returns_token(self) -> None:
        token = self.masker.mask("Anadolu Gıda")
        assert token == "CUSTOMER_1"

    def test_mask_is_idempotent(self) -> None:
        """Same input always yields same token within one session."""
        t1 = self.masker.mask("Anadolu Gıda")
        t2 = self.masker.mask("Anadolu Gıda")
        assert t1 == t2

    def test_mask_increments_counter(self) -> None:
        self.masker.mask("A")
        self.masker.mask("B")
        assert self.masker.mask("C") == "CUSTOMER_3"

    def test_mask_raises_on_empty_string(self) -> None:
        with pytest.raises(MaskingError):
            self.masker.mask("")

    def test_mask_raises_on_whitespace_only(self) -> None:
        with pytest.raises(MaskingError):
            self.masker.mask("   ")

    def test_mask_raises_on_non_string(self) -> None:
        with pytest.raises(MaskingError):
            self.masker.mask(123)  # type: ignore[arg-type]

    # ── unmask() ───────────────────────────────────────────────────── #

    def test_unmask_round_trip(self) -> None:
        token = self.masker.mask("Anadolu Gıda")
        assert self.masker.unmask(token) == "Anadolu Gıda"

    def test_unmask_raises_for_unknown_token(self) -> None:
        with pytest.raises(MaskingError):
            self.masker.unmask("CUSTOMER_99")

    # ── unmask_text() ──────────────────────────────────────────────── #

    def test_unmask_text_substitutes_tokens(self) -> None:
        self.masker.mask("Anadolu Gıda")
        result = self.masker.unmask_text("CUSTOMER_1 has the highest revenue.")
        assert result == "Anadolu Gıda has the highest revenue."

    # ── reset() ────────────────────────────────────────────────────── #

    def test_reset_clears_all_mappings(self) -> None:
        self.masker.mask("Anadolu Gıda")
        self.masker.reset()
        assert self.masker.get_mask_map() == {}

    def test_reset_allows_reuse(self) -> None:
        self.masker.mask("Anadolu Gıda")
        self.masker.reset()
        token = self.masker.mask("Yeni Şirket")
        assert token == "CUSTOMER_1"  # counter restarted
