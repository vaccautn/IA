"""Immutable logical and physical change counts for dataset builds."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class ChangeCounts:
    selected: int
    train: int
    val: int
    staged: int
    added: int
    updated: int
    unchanged: int
    stale: int

    def as_dict(self) -> dict[str, int]:
        return {
            "selected": self.selected,
            "train": self.train,
            "val": self.val,
            "staged": self.staged,
            "added": self.added,
            "updated": self.updated,
            "unchanged": self.unchanged,
            "stale": self.stale,
        }


def _sum_counts(counts: Iterable[ChangeCounts]) -> ChangeCounts:
    values = list(counts)
    return ChangeCounts(
        selected=sum(item.selected for item in values),
        train=sum(item.train for item in values),
        val=sum(item.val for item in values),
        staged=sum(item.staged for item in values),
        added=sum(item.added for item in values),
        updated=sum(item.updated for item in values),
        unchanged=sum(item.unchanged for item in values),
        stale=sum(item.stale for item in values),
    )


@dataclass(frozen=True)
class ChangeSummary:
    per_class: tuple[tuple[str, ChangeCounts], ...]

    @property
    def totals(self) -> ChangeCounts:
        return _sum_counts(counts for _, counts in self.per_class)

    def counts_for(self, class_name: str) -> ChangeCounts:
        for name, counts in self.per_class:
            if name == class_name:
                return counts
        raise KeyError(class_name)

    def with_staged(self, staged_by_class: dict[str, int]) -> ChangeSummary:
        return ChangeSummary(
            tuple(
                (
                    class_name,
                    ChangeCounts(
                        selected=counts.selected,
                        train=counts.train,
                        val=counts.val,
                        staged=staged_by_class[class_name],
                        added=counts.added,
                        updated=counts.updated,
                        unchanged=counts.unchanged,
                        stale=counts.stale,
                    ),
                )
                for class_name, counts in self.per_class
            )
        )

    def as_per_class_dict(self) -> dict[str, dict[str, int]]:
        return {class_name: counts.as_dict() for class_name, counts in self.per_class}

    def as_totals_dict(self) -> dict[str, int]:
        return self.totals.as_dict()
