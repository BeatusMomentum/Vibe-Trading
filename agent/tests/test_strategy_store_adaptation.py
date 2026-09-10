"""Adaptation writes a new CREATED strategy through the artifact store."""

from __future__ import annotations

import pytest

from src.strategy_store.adaptation import (
    AdaptationError,
    AdaptationPatch,
    adapt_artifact,
    register_adaptation,
)
from src.strategy_store.models import (
    Artifact,
    ArtifactStatus,
    ArtifactType,
    ValidationStatus,
)
from src.strategy_store.store import InMemoryStrategyStore


@pytest.fixture(autouse=True)
def _reset_store(tmp_path):
    import src.strategy_store._shared as shared
    from src.strategy_store.sqlite_store import SqliteStrategyStore

    shared._store = SqliteStrategyStore(db_path=tmp_path / "test.db")
    yield
    shared._store = None


SIGNAL = "close > sma(20)"
ENTRY = "buy next open"
EXIT = "sell next open"
ENGINE = "engines/momentum.py"


def _strategy(
    *,
    artifact_id: str = "art_parent",
    name: str = "momentum",
    universe: str = "CSI300",
    artifact_type: ArtifactType = ArtifactType.STRATEGY,
    **overrides,
) -> Artifact:
    fields = dict(
        id=artifact_id,
        type=artifact_type,
        name=name,
        universe=universe,
        signal_definition=SIGNAL,
        entry_rules=ENTRY,
        exit_rules=EXIT,
        position_sizing="monthly rebalance",
        signal_engine_path=ENGINE,
        run_dir="/tmp/parent_run",
        status=ArtifactStatus.ACTIVE,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-02T00:00:00+00:00",
        developer="Ada",
        owner="Bob",
        theme=("momentum",),
        columns_required=("close",),
        decay_horizon=20,
        source_paper="Jegadeesh 1993",
        hypothesis_id="hyp_1",
        validation_status=ValidationStatus.UNVALIDATED,
    )
    fields.update(overrides)
    return Artifact(**fields)


class TestAdaptArtifact:
    def test_child_keeps_parent_signal_fields(self) -> None:
        parent = _strategy()
        child = adapt_artifact(parent, AdaptationPatch(universe="SP500"))
        assert child.signal_definition == SIGNAL
        assert child.entry_rules == ENTRY
        assert child.exit_rules == EXIT
        assert child.signal_engine_path == ENGINE
        assert child.theme == parent.theme
        assert child.columns_required == parent.columns_required
        assert child.decay_horizon == parent.decay_horizon
        assert child.source_paper == parent.source_paper
        assert child.hypothesis_id == parent.hypothesis_id
        assert child.developer == parent.developer
        assert child.owner == parent.owner

    def test_child_identity_starts_clean(self) -> None:
        parent = _strategy()
        child = adapt_artifact(parent, AdaptationPatch(universe="SP500"))
        assert child.derived_from == parent.id
        assert child.id == ""
        assert child.run_dir is None
        assert child.status is ArtifactStatus.CREATED
        assert child.created_at == ""
        assert child.updated_at == ""
        assert child.disabled_at is None
        assert child.disabled_reason is None
        assert child.validation_status is ValidationStatus.UNVALIDATED
        assert child.validation_date is None

    def test_factor_parent_refused(self) -> None:
        parent = _strategy(artifact_type=ArtifactType.FACTOR)
        with pytest.raises(AdaptationError, match="strategy"):
            adapt_artifact(parent, AdaptationPatch(universe="SP500"))

    def test_noop_same_name_and_universe_refused(self) -> None:
        parent = _strategy()
        with pytest.raises(AdaptationError, match="adaptation must change universe or name"):
            adapt_artifact(parent, AdaptationPatch(position_sizing="weekly rebalance"))


class TestRegisterAdaptation:
    def test_new_universe_same_name_succeeds(self) -> None:
        store = InMemoryStrategyStore()
        parent_id = store.register_artifact(_strategy(artifact_id=""))
        child_id = register_adaptation(store, parent_id, AdaptationPatch(universe="SP500"))
        child = store.get_artifact(child_id)
        parent = store.get_artifact(parent_id)
        assert child is not None
        assert parent is not None
        assert child.id == child_id
        assert child.id != parent_id
        assert child.derived_from == parent_id
        assert child.run_dir is None
        assert child.status is ArtifactStatus.CREATED
        assert child.name == parent.name
        assert child.universe == "SP500"

    def test_same_universe_same_name_raises_duplicate(self) -> None:
        store = InMemoryStrategyStore()
        parent_id = store.register_artifact(_strategy(artifact_id=""))
        register_adaptation(store, parent_id, AdaptationPatch(name="momentum_weekly"))
        with pytest.raises(ValueError, match="already exists"):
            register_adaptation(store, parent_id, AdaptationPatch(name="momentum_weekly"))


class TestSqliteDerivedFromRoundtrip:
    def test_derived_from_survives_sqlite_roundtrip(self) -> None:
        from src.strategy_store._shared import get_store

        store = get_store()
        parent_id = store.register_artifact(_strategy(artifact_id=""))
        child_id = register_adaptation(store, parent_id, AdaptationPatch(universe="SP500"))
        fetched = store.get_artifact(child_id)
        assert fetched is not None
        assert fetched.derived_from == parent_id
        assert fetched.id == child_id
        assert fetched.id != parent_id
        assert fetched.run_dir is None
        assert fetched.status is ArtifactStatus.CREATED
        assert fetched.signal_definition == SIGNAL
