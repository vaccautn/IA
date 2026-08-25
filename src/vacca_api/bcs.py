"""BCS score adaptation at the backend-facing API boundary."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_DOWN
from numbers import Real


class BCSScoreValidationError(ValueError):
    """Raised when a model score cannot be exposed through the BCS API."""


def round_bcs_score_for_backend(score: object) -> int:
    """Validate and round a finite model score for backend evaluations.

    Decimal half-down semantics are intentional: exact ``.5`` values round
    toward the lower integer, while values above the tie round upward.
    """
    if isinstance(score, bool) or not isinstance(score, (Real, Decimal)):
        raise BCSScoreValidationError(
            "BCS model score must be numeric (int, float, or Decimal)."
        )

    try:
        decimal_score = Decimal(str(score))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise BCSScoreValidationError(
            "BCS model score must be representable as a decimal number."
        ) from exc

    if not decimal_score.is_finite():
        raise BCSScoreValidationError("BCS model score must be finite.")
    if not Decimal("1") <= decimal_score <= Decimal("5"):
        raise BCSScoreValidationError(
            f"BCS model score must be within the inclusive range 1..5; got {score!r}."
        )

    return int(decimal_score.quantize(Decimal("1"), rounding=ROUND_HALF_DOWN))
