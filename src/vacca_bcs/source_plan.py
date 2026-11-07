"""Normalized local BCS category source records."""
from __future__ import annotations

from dataclasses import dataclass
import re

from .local_source import LOCAL_SOURCE_SCHEMA, LocalSourceScan
from .constants import BCS_CLASS_SCORES, NUM_CLASSES

_RECORD_ID_RE = re.compile(r"^[0-9a-f]{64}$")


class SourcePlanError(Exception):
    pass


class SourcePlanIntegrityError(SourcePlanError):
    pass


@dataclass(frozen=True, slots=True)
class LocalSourceProvenance:
    relative_path: str
    source_label: str


@dataclass(frozen=True, slots=True)
class SourceExclusion:
    record_id: str
    relative_path: str
    source_label: str
    bcs_category: int
    sha256: str
    reason: str


@dataclass(frozen=True, slots=True)
class SourceRecord:
    record_id: str
    source_schema: str
    bcs_category: int
    materializer_key: str
    provenance: tuple[LocalSourceProvenance, ...]
    capture_group: str
    sha256: str
    member_id: str = ""

    def __post_init__(self) -> None:
        if type(self.record_id) is not str or not _RECORD_ID_RE.fullmatch(self.record_id):
            raise SourcePlanIntegrityError("source record has an invalid record ID")
        if self.source_schema != LOCAL_SOURCE_SCHEMA:
            raise SourcePlanIntegrityError("source record has an invalid schema")
        if type(self.bcs_category) is not int or self.bcs_category not in BCS_CLASS_SCORES:
            raise SourcePlanIntegrityError("source record has an invalid category")
        if type(self.materializer_key) is not str or not self.materializer_key:
            raise SourcePlanIntegrityError("source record has an invalid materializer key")
        if type(self.capture_group) is not str or not self.capture_group:
            raise SourcePlanIntegrityError("source record has an invalid capture group")
        if type(self.sha256) is not str or not _RECORD_ID_RE.fullmatch(self.sha256):
            raise SourcePlanIntegrityError("source record has an invalid digest")
        if type(self.member_id) is not str or not self.member_id:
            raise SourcePlanIntegrityError("source record has an invalid capture member")
        object.__setattr__(self, "provenance", tuple(self.provenance))


@dataclass(frozen=True, slots=True)
class SourcePlan:
    candidates: tuple[SourceRecord, ...]
    exclusions: tuple[SourceExclusion, ...] = ()
    class_counts: tuple[int, ...] = (0,) * NUM_CLASSES
    source_schema: str = LOCAL_SOURCE_SCHEMA
    identity_scheme: str = "local-path-sha256-v1"
    mapping_lineage: tuple[tuple[str, int], ...] = ()
    observed_classes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "exclusions", tuple(self.exclusions))
        object.__setattr__(self, "class_counts", tuple(self.class_counts))
        object.__setattr__(self, "mapping_lineage", tuple(self.mapping_lineage))
        object.__setattr__(self, "observed_classes", tuple(self.observed_classes))


def normalize_local_source_scan(scan: LocalSourceScan) -> SourcePlan:
    """Adapt a validated local scan without manufacturing animal identities."""
    if type(scan) is not LocalSourceScan:
        raise SourcePlanIntegrityError("local source input is invalid")
    candidates = tuple(
        SourceRecord(
            record_id=record.record_id,
            source_schema=LOCAL_SOURCE_SCHEMA,
            bcs_category=record.bcs_category,
            materializer_key=record.relative_path,
            provenance=(LocalSourceProvenance(record.relative_path, record.source_label),),
            capture_group=record.capture_group,
            sha256=record.sha256,
            member_id=record.member_id,
        )
        for record in scan.records
    )
    return SourcePlan(
        candidates,
        tuple(
            SourceExclusion(
                exclusion.record_id,
                exclusion.relative_path,
                exclusion.source_label,
                exclusion.bcs_category,
                exclusion.sha256,
                exclusion.reason,
            )
            for exclusion in scan.exclusions
        ),
        scan.counts,
        LOCAL_SOURCE_SCHEMA,
        "local-path-sha256-v1",
        scan.mapping_lineage,
        scan.observed_classes,
    )
