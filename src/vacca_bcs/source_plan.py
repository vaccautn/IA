from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .source_client import (
    BCSSourceEvidence,
    BCSSourceEvaluationRow,
    BCSSourceExport,
)


class SourcePlanError(Exception):
    pass


class SourcePlanIntegrityError(SourcePlanError):
    pass


class SourcePlanConflictError(SourcePlanError):
    pass


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    evaluation_id: int
    session_id: int
    animal_id: int
    evidence_id: int
    storage_key: str
    bcs_score: int
    evidence_ids: tuple[int, ...]
    evaluation_ids: tuple[int, ...]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(evaluation_id={self.evaluation_id!r}, "
            f"evidence_id={self.evidence_id!r}, bcs_score={self.bcs_score!r}, "
            f"evidence_ids={self.evidence_ids!r}, evaluation_ids={self.evaluation_ids!r})"
        )


@dataclass(frozen=True, slots=True)
class SourceExclusion:
    evaluation_id: int
    evidence_id: int
    bcs_score: int
    reason: str


@dataclass(frozen=True, slots=True)
class SourcePlan:
    candidates: tuple[SourceCandidate, ...]
    exclusions: tuple[SourceExclusion, ...]
    class_counts: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "class_counts", tuple(self.class_counts))


def _rows_from(
    source: BCSSourceExport | Iterable[BCSSourceEvaluationRow],
) -> tuple[BCSSourceEvaluationRow, ...]:
    rows = source.rows if isinstance(source, BCSSourceExport) else tuple(source)
    if any(type(row) is not BCSSourceEvaluationRow for row in rows):
        raise SourcePlanIntegrityError(
            "source plan input contains an invalid evaluation row"
        )
    return rows


def _duplicate_ids(values: Iterable[int]) -> tuple[int, ...]:
    items = tuple(values)
    return tuple(sorted(value for value in set(items) if items.count(value) > 1))


def normalize_source_export(
    source: BCSSourceExport | Iterable[BCSSourceEvaluationRow],
) -> SourcePlan:
    rows = _rows_from(source)
    duplicate_evaluations = _duplicate_ids(row.evaluation_id for row in rows)
    if duplicate_evaluations:
        raise SourcePlanIntegrityError(
            f"duplicate evaluation IDs: {duplicate_evaluations}"
        )
    duplicate_evidence = _duplicate_ids(
        evidence.evidence_id for row in rows for evidence in row.evidence
    )
    if duplicate_evidence:
        raise SourcePlanIntegrityError(f"duplicate evidence IDs: {duplicate_evidence}")

    groups: dict[str, list[tuple[BCSSourceEvaluationRow, BCSSourceEvidence]]] = {}
    exclusions: list[SourceExclusion] = []
    for row in rows:
        for evidence in row.evidence:
            key = evidence.storage_key
            if key == "":
                exclusions.append(
                    SourceExclusion(
                        row.evaluation_id,
                        evidence.evidence_id,
                        row.valor_cc,
                        "empty_storage_key",
                    )
                )
            elif not key.strip():
                exclusions.append(
                    SourceExclusion(
                        row.evaluation_id,
                        evidence.evidence_id,
                        row.valor_cc,
                        "whitespace_storage_key",
                    )
                )
            elif key != key.strip():
                raise SourcePlanIntegrityError(
                    "storage key has surrounding whitespace for "
                    f"evidence_id={evidence.evidence_id}, evaluation_id={row.evaluation_id}"
                )
            else:
                groups.setdefault(key, []).append((row, evidence))

    candidates: list[SourceCandidate] = []
    for records in groups.values():
        labels = {row.valor_cc for row, _ in records}
        if len(labels) > 1:
            evidence_ids = tuple(
                sorted(evidence.evidence_id for _, evidence in records)
            )
            evaluation_ids = tuple(sorted(row.evaluation_id for row, _ in records))
            raise SourcePlanConflictError(
                "conflicting labels for "
                f"evidence_ids={evidence_ids}, evaluation_ids={evaluation_ids}"
            )
        canonical_row, canonical_evidence = min(
            records,
            key=lambda item: (item[1].evidence_id, item[0].evaluation_id),
        )
        candidates.append(
            SourceCandidate(
                canonical_row.evaluation_id,
                canonical_row.session_id,
                canonical_row.animal_id,
                canonical_evidence.evidence_id,
                canonical_evidence.storage_key,
                canonical_row.valor_cc,
                tuple(sorted(evidence.evidence_id for _, evidence in records)),
                tuple(sorted(row.evaluation_id for row, _ in records)),
            )
        )

    ordered_candidates = tuple(
        sorted(candidates, key=lambda item: (item.bcs_score, item.evidence_id))
    )
    counts = [0] * 5
    for candidate in ordered_candidates:
        counts[candidate.bcs_score - 1] += 1
    ordered_exclusions = tuple(
        sorted(
            exclusions,
            key=lambda item: (item.evidence_id, item.evaluation_id, item.reason),
        )
    )
    return SourcePlan(ordered_candidates, ordered_exclusions, tuple(counts))


build_source_plan = normalize_source_export
