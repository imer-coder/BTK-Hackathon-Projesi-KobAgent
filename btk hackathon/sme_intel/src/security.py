"""
security.py — Data masking layer for SME-Intel.

SECURITY CONTRACT:
    Raw commercial identifiers (customer names, etc.) MUST NEVER
    leave this module in plain text form.  All downstream code
    (LLM calls, logs, API payloads) must operate exclusively on
    the anonymous tokens produced by DataMasker.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class MaskingError(Exception):
    """Raised when a masking or unmasking operation fails."""


class DataMasker:
    """
    Bidirectional, in-memory pseudonymisation engine.

    Converts real commercial identifiers (e.g., customer names)
    into deterministic, opaque tokens such as ``CUSTOMER_1``, and
    can reverse that mapping back to the original value on demand.

    Design decisions:
    - Two plain dicts (``_mask_map``, ``_unmask_map``) give O(1)
      look-ups in both directions with zero external dependencies.
    - Tokens are **stable** within a single session: the same name
      always yields the same token throughout the lifetime of this
      object, preventing drift in multi-step LangGraph pipelines.
    - The object is **not** thread-safe by design; create one
      instance per user session / Streamlit session state entry.

    Args:
        prefix: Token prefix string (default ``"CUSTOMER"``).
                Change to ``"VENDOR"`` or ``"ENTITY"`` as needed.
    """

    def __init__(self, prefix: str = "CUSTOMER") -> None:
        self._prefix: str = prefix
        self._mask_map: Dict[str, str] = {}    # real_name  → token
        self._unmask_map: Dict[str, str] = {}  # token      → real_name
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def mask(self, real_value: str) -> str:
        """
        Return the opaque token for *real_value*, creating one if needed.

        Args:
            real_value: The sensitive identifier to pseudonymise.

        Returns:
            A deterministic token string, e.g. ``"CUSTOMER_3"``.

        Raises:
            MaskingError: If *real_value* is empty or not a string.
        """
        self._validate_input(real_value, operation="mask")

        if real_value in self._mask_map:
            return self._mask_map[real_value]

        self._counter += 1
        token = f"{self._prefix}_{self._counter}"
        self._mask_map[real_value] = token
        self._unmask_map[token] = real_value
        logger.debug("Masked '%s' → '%s'", real_value, token)
        return token

    def unmask(self, token: str) -> str:
        """
        Resolve *token* back to its original real value.

        Args:
            token: A token previously produced by :meth:`mask`.

        Returns:
            The original sensitive identifier.

        Raises:
            MaskingError: If *token* is unknown or invalid.
        """
        self._validate_input(token, operation="unmask")

        if token not in self._unmask_map:
            raise MaskingError(
                f"Unknown token '{token}'. It may not have been created by "
                "this DataMasker instance, or the instance was reset."
            )

        real_value = self._unmask_map[token]
        logger.debug("Unmasked '%s' → '%s'", token, real_value)
        return real_value

    def mask_series(self, values: List[str]) -> List[str]:
        """
        Convenience wrapper: mask an entire list of identifiers.

        Args:
            values: List of sensitive identifiers.

        Returns:
            List of corresponding tokens, same order and length.
        """
        return [self.mask(v) for v in values]

    def unmask_series(self, tokens: List[str]) -> List[str]:
        """
        Convenience wrapper: unmask an entire list of tokens.

        Args:
            tokens: List of tokens produced by :meth:`mask_series`.

        Returns:
            List of original sensitive identifiers, same order.
        """
        return [self.unmask(t) for t in tokens]

    def unmask_text(self, text: str) -> str:
        """
        Replace all token occurrences inside a free-form text string
        with their original real values.

        Useful for post-processing LLM responses that contain tokens
        such as ``"CUSTOMER_1"`` before displaying them to the end user.

        Args:
            text: Free-form string potentially containing tokens.

        Returns:
            String with all known tokens substituted back.
        """
        result = text
        for token, real_value in self._unmask_map.items():
            result = result.replace(token, real_value)
        return result

    def get_mask_map(self) -> Dict[str, str]:
        """Return a **copy** of the real_value → token mapping (read-only)."""
        return dict(self._mask_map)

    def reset(self) -> None:
        """
        Clear all mappings and reset the counter.

        Use this between separate analysis sessions if the same
        DataMasker instance is reused.
        """
        self._mask_map.clear()
        self._unmask_map.clear()
        self._counter = 0
        logger.info("DataMasker reset. All mappings cleared.")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_input(value: str, operation: str) -> None:
        """
        Assert that *value* is a non-empty string.

        Args:
            value: The value to validate.
            operation: Label used in the error message ('mask'/'unmask').

        Raises:
            MaskingError: If validation fails.
        """
        if not isinstance(value, str):
            raise MaskingError(
                f"[{operation}] Expected str, got {type(value).__name__!r}."
            )
        if not value.strip():
            raise MaskingError(
                f"[{operation}] Value must not be empty or whitespace-only."
            )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"DataMasker(prefix={self._prefix!r}, "
            f"entries={self._counter})"
        )
