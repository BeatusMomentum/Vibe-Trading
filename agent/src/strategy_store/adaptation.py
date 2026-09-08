"""Description-driven adaptation of SDM strategy artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

from src.strategy_store.models import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ValidationStatus,
)


class AdaptationError(ValueError):
    """Raised when a parent cannot be adapted under the Phase 3 contract."""


@dataclass(frozen=True)
class AdaptationPatch:
    """Allowed description-driven changes. Signal fields are not on this type."""

    universe: str | None = None
    name: str | None = None
    position_sizing: str | None = None


class _ArtifactStore(Protocol):
    def get_artifact(self, artifact_id: str) -> Artifact | None: ...

    def register_artifact(self, artifact: Artifact) -> str: ...


def adapt_artifact(parent: Artifact, patch: AdaptationPatch) -> Artifact:
    """Return a new strategy artifact derived from *parent*. Does not write."""
    if parent.type is not ArtifactType.STRATEGY:
        raise AdaptationError(
            f"only strategy artifacts can be adapted; got type={parent.type.value!r}"
        )
    if not parent.id:
        raise AdaptationError("parent artifact id is required")

    new_universe = patch.universe if patch.universe is not None else parent.universe
    new_name = patch.name if patch.name is not None else parent.name
    if new_universe == parent.universe and new_name == parent.name:
        raise AdaptationError("adaptation must change universe or name")

    new_sizing = (
        patch.position_sizing if patch.position_sizing is not None else parent.position_sizing
    )
    return replace(
        parent,
        id="",
        name=new_name,
        universe=new_universe,
        position_sizing=new_sizing,
        derived_from=parent.id,
        status=ArtifactStatus.CREATED,
        run_dir=None,
        created_at="",
        updated_at="",
        disabled_at=None,
        disabled_reason=None,
        validation_status=ValidationStatus.UNVALIDATED,
        validation_date=None,
    )


def register_adaptation(
    store: _ArtifactStore, parent_id: str, patch: AdaptationPatch
) -> str:
    """Persist an adapted child through the strategy store. Returns the new id."""
    parent = store.get_artifact(parent_id)
    if parent is None:
        raise AdaptationError(f"parent artifact {parent_id!r} not found")
    child = adapt_artifact(parent, patch)
    return store.register_artifact(child)
