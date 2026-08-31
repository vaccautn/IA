from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import re

from .local_source import LOCAL_SOURCE_SCHEMA, LocalSourceScan
from .source_client import (
    BCSSourceEvidence,
    BCSSourceEvaluationRow,
    BCSSourceExport,
)

_RECORD_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class SourcePlanError(Exception):
    pass


class SourcePlanIntegrityError(SourcePlanError):
    pass


class SourcePlanConflictError(SourcePlanError):
    def __init__(self, provenance: tuple[SourceProvenance, ...]) -> None:
        self.provenance = tuple(provenance)
        self.evidence_ids = tuple(item.evidence_id for item in self.provenance)
        self.evaluation_ids = tuple(item.evaluation_id for item in self.provenance)
        super().__init__(
            "conflicting labels for "
            f"evidence_ids={self.evidence_ids}, evaluation_ids={self.evaluation_ids}"
        )


@dataclass(frozen=True, slots=True)
class BackendSourceProvenance:
    evidence_id: int
    evaluation_id: int
    session_id: int
    animal_id: int


@dataclass(frozen=True, slots=True)
class LocalSourceProvenance:
    relative_path: str
    source_label: str


@dataclass(frozen=True, slots=True)
class SourceRecord:
    record_id: str
    source_schema: str
    bcs_score: int
    materializer_key: str | int
    provenance: tuple[BackendSourceProvenance | LocalSourceProvenance, ...]

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not _RECORD_ID_RE.fullmatch(self.record_id):
            raise SourcePlanIntegrityError("source record has an invalid record ID")
        if type(self.source_schema) is not str or not self.source_schema:
            raise SourcePlanIntegrityError("source record has an invalid schema")
        if type(self.bcs_score) is not int or not 1 <= self.bcs_score <= 5:
            raise SourcePlanIntegrityError("source record has an invalid score")
        if type(self.materializer_key) not in (str, int):
            raise SourcePlanIntegrityError("source record has an invalid materializer key")
        object.__setattr__(self, "provenance", tuple(self.provenance))


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    evidence_id: int
    evaluation_id: int


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    evaluation_id: int
    session_id: int
    animal_id: int
    evidence_id: int
    storage_key: str
    bcs_score: int
    provenance: tuple[SourceProvenance, ...]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(evaluation_id={self.evaluation_id!r}, "
            f"evidence_id={self.evidence_id!r}, bcs_score={self.bcs_score!r}, "
            f"provenance={self.provenance!r})"
        )


@dataclass(frozen=True, slots=True)
class SourceExclusion:
    evaluation_id: int
    evidence_id: int
    bcs_score: int
    reason: str


@dataclass(frozen=True, slots=True)
class SourcePlan:
    candidates: tuple[SourceCandidate | SourceRecord, ...]
    exclusions: tuple[SourceExclusion, ...]
    class_counts: tuple[int, int, int, int, int]
    source_schema: str = "bcs-source-v1"
    identity_scheme: str = "backend-evidence-v1"
    mapping_lineage: tuple[tuple[str, int], ...] = ()
    observed_classes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "class_counts", tuple(self.class_counts))
        object.__setattr__(self, "mapping_lineage", tuple(self.mapping_lineage))
        object.__setattr__(self, "observed_classes", tuple(self.observed_classes))


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


def _provenance(
    records: Iterable[tuple[BCSSourceEvaluationRow, BCSSourceEvidence]],
) -> tuple[SourceProvenance, ...]:
    return tuple(
        sorted(
            (
                SourceProvenance(
                    evidence_id=evidence.evidence_id,
                    evaluation_id=row.evaluation_id,
                )
                for row, evidence in records
            ),
            key=lambda item: (item.evidence_id, item.evaluation_id),
        )
    )


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
    conflicts: list[tuple[SourceProvenance, ...]] = []
    for records in groups.values():
        labels = {row.valor_cc for row, _ in records}
        if len(labels) > 1:
            conflicts.append(_provenance(records))
            continue
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
                _provenance(records),
            )
        )

    if conflicts:
        raise SourcePlanConflictError(
            min(
                conflicts,
                key=lambda group: tuple(
                    (item.evidence_id, item.evaluation_id) for item in group
                ),
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


def _backend_record_id(evidence_id: int) -> str:
    return hashlib.sha256(f"bcs-source-v1\0evidence/{evidence_id}".encode()).hexdigest()


def normalize_backend_source_export(
    source: BCSSourceExport | Iterable[BCSSourceEvaluationRow],
) -> SourcePlan:
    """Adapt the existing backend policy to source-neutral records."""
    rows = _rows_from(source)
    legacy = normalize_source_export(rows)
    provenance_by_id = {
        (row.evaluation_id, evidence.evidence_id): BackendSourceProvenance(
            evidence.evidence_id,
            row.evaluation_id,
            row.session_id,
            row.animal_id,
        )
        for row in rows
        for evidence in row.evidence
    }
    candidates = tuple(
        SourceRecord(
            record_id=_backend_record_id(item.evidence_id),
            source_schema="bcs-source-v1",
            bcs_score=item.bcs_score,
            materializer_key=item.evidence_id,
            provenance=tuple(
                provenance_by_id[(provenance.evaluation_id, provenance.evidence_id)]
                for provenance in item.provenance
            ),
        )
        for item in legacy.candidates
    )
    return SourcePlan(
        candidates,
        legacy.exclusions,
        legacy.class_counts,
        source_schema="bcs-source-v1",
        identity_scheme="backend-evidence-v1",
        observed_classes=tuple(i + 1 for i, count in enumerate(legacy.class_counts) if count),
    )


def normalize_local_source_scan(scan: LocalSourceScan) -> SourcePlan:
    """Adapt a validated local scan without manufacturing backend identifiers."""
    if type(scan) is not LocalSourceScan:
        raise SourcePlanIntegrityError("local source input is invalid")
    candidates = tuple(
        SourceRecord(
            record.record_id,
            LOCAL_SOURCE_SCHEMA,
            record.bcs_score,
            record.relative_path,
            (LocalSourceProvenance(record.relative_path, record.source_label),),
        )
        for record in scan.records
    )
    return SourcePlan(
        candidates,
        (),
        scan.counts,
        source_schema=LOCAL_SOURCE_SCHEMA,
        identity_scheme="local-path-sha256-v1",
        mapping_lineage=scan.mapping_lineage,
        observed_classes=scan.observed_classes,
    )


build_source_plan = normalize_source_export
