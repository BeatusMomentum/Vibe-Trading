"""API regressions for the live-trading runtime wiring (R6: C1 + C2 + runner control).

Covers the open-file integration seam the orchestrator and parcel R6 own:
- C1: a ``propose_mandate_profiles`` tool_result is translated into a top-level
  ``mandate.proposal`` SSE frame WITHOUT touching the protected ``src/agent/loop.py``.
- C2: ``GET /live/status`` surfaces the dormant-by-default channel (auth, mandate,
  runner liveness, halt) and ``POST /live/authorize`` is a discover-only on-ramp.
- Runner control: ``POST /live/runner/start|stop`` are privileged surface actions
  gated on a committed, unexpired mandate and a clear kill switch.

All tests run against stubbed runner/liveness state — no real agent or broker.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from fastapi.testclient import TestClient

import pytest

import api_server
from tests import robinhood_mcp_helpers as rh


def _client(tmp_path: Path, monkeypatch) -> TestClient:
    # Redirect the runtime root (``~/.vibe-trading``) at the home boundary so the
    # live tree, HALT sentinel, mandate store, and proposal store all resolve
    # under tmp_path. get_runtime_root() == Path.home() / ".vibe-trading".
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_tasks", {}, raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)
    return TestClient(api_server.app, client=("127.0.0.1", 50000))


def _valid_mandate_state(broker: str = "robinhood") -> api_server.ActiveMandateState:
    """Build a committed, unexpired active-mandate snapshot for stubbing."""
    return api_server.ActiveMandateState(
        broker=broker,
        account_ref="acct_test",
        created_at="2026-05-29T00:00:00+00:00",
        expires_at="2099-01-01T00:00:00+00:00",
        expires_in_seconds=10_000_000,
        expired=False,
        limits=api_server.MandateLimits(
            max_order_notional_usd=750.0,
            max_total_exposure_usd=5000.0,
            max_leverage=1.0,
            max_trades_per_day=5,
            allowed_instruments=["equity"],
            account_funding_usd=5000.0,
        ),
    )


# --------------------------------------------------------------------------- #
# C2 — GET /live/status
# --------------------------------------------------------------------------- #


def test_live_status_dormant_by_default(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/live/status")

    assert response.status_code == 200
    body = response.json()
    assert body["global_halted"] is False
    brokers = {b["auth"]["broker"]: b for b in body["brokers"]}
    assert "robinhood" in brokers
    rh = brokers["robinhood"]
    # Channel is OFF until OAuth + mandate: no token, no mandate, runner dead.
    assert rh["auth"]["oauth_token_present"] is False
    assert rh["auth"]["is_live_broker"] is True
    assert rh["mandate"] is None
    assert rh["runner"]["alive"] is False
    assert rh["halted"] is False


def test_live_status_single_broker_filter(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/live/status", params={"broker": "Robinhood "})

    assert response.status_code == 200
    body = response.json()
    assert [b["auth"]["broker"] for b in body["brokers"]] == ["robinhood"]


def test_live_status_blank_broker_rejected(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/live/status", params={"broker": "   "})

    assert response.status_code == 400


def test_live_status_reflects_active_mandate(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api_server, "_active_mandate_state", lambda broker: _valid_mandate_state(broker)
    )

    response = client.get("/live/status", params={"broker": "robinhood"})

    assert response.status_code == 200
    rh = response.json()["brokers"][0]
    assert rh["mandate"]["expired"] is False
    assert rh["mandate"]["limits"]["max_order_notional_usd"] == 750.0


# --------------------------------------------------------------------------- #
# C2 — POST /live/authorize (discover-only on-ramp; never authorizes server-side)
# --------------------------------------------------------------------------- #


def test_authorize_onramp_describes_cli_flow(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/live/authorize", json={"broker": "robinhood"})

    assert response.status_code == 200
    body = response.json()
    assert body["broker"] == "robinhood"
    assert body["connector_profile"] == "robinhood-live-mcp"
    assert body["oauth_token_present"] is False
    # On-ramp must point at the desktop CLI flow and never return a token.
    assert "vibe-trading connector authorize robinhood-live-mcp" in body["instruction"]
    assert "token" not in body


def test_authorize_unknown_broker_rejected(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/live/authorize", json={"broker": "etrade"})

    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Runner control — POST /live/runner/start|stop
# --------------------------------------------------------------------------- #


def test_runner_start_requires_committed_mandate(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/live/runner/start", json={"broker": "robinhood"})

    assert response.status_code == 409
    assert "mandate" in response.json()["detail"].lower()


def test_runner_start_rejects_readonly_connector_without_runner(
    tmp_path: Path, monkeypatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/live/runner/start", json={"broker": "ibkr"})

    assert response.status_code == 400
    detail = response.json()["detail"].lower()
    assert "ibkr" in detail
    assert "runner" in detail


def test_runner_start_blocked_by_kill_switch(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api_server, "_active_mandate_state", lambda broker: _valid_mandate_state(broker)
    )
    from src.live.halt import trip_halt

    trip_halt(by="test", reason="safety", broker="robinhood")

    response = client.post("/live/runner/start", json={"broker": "robinhood"})

    assert response.status_code == 409
    assert "kill switch" in response.json()["detail"].lower()


def test_runner_start_success_then_idempotent(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api_server, "_active_mandate_state", lambda broker: _valid_mandate_state(broker)
    )
    monkeypatch.setattr(api_server, "_runner_factory", lambda broker: SimpleNamespace(broker=broker))

    async def _noop_drive(runner) -> None:  # never spawns a real agent/broker loop
        return None

    monkeypatch.setattr(api_server, "_drive_runner", _noop_drive)

    first = client.post("/live/runner/start", json={"broker": "robinhood"})
    assert first.status_code == 200
    assert first.json() == {"broker": "robinhood", "started": True, "already_running": False}

    # Pre-seed a still-pending task so the idempotency branch is deterministic.
    class _PendingTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self._cancelled = True

    monkeypatch.setattr(api_server, "_runner_tasks", {"robinhood": _PendingTask()})

    second = client.post("/live/runner/start", json={"broker": "robinhood"})
    assert second.status_code == 200
    assert second.json()["already_running"] is True


def test_runner_stop_idempotent_when_not_running(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/live/runner/stop", json={"broker": "robinhood"})

    assert response.status_code == 200
    assert response.json() == {"broker": "robinhood", "stopped": False, "was_running": False}


def test_runner_stop_cancels_running_task(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)

    cancelled = {"value": False}

    class _PendingTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            cancelled["value"] = True

    monkeypatch.setattr(api_server, "_runner_tasks", {"robinhood": _PendingTask()})

    response = client.post("/live/runner/stop", json={"broker": "robinhood"})

    assert response.status_code == 200
    assert response.json() == {"broker": "robinhood", "stopped": True, "was_running": True}
    assert cancelled["value"] is True


# --------------------------------------------------------------------------- #
# Runner stop must actually stop the scheduler (R-INT lifecycle)
# --------------------------------------------------------------------------- #


def _live_env(tmp_path: Path, monkeypatch) -> None:
    """Sandbox the runtime root and the module-level runner registries."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_tasks", {}, raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)


def _live_client() -> httpx.AsyncClient:
    """An ASGI client bound to the *running* loop (no lifespan, no portal thread).

    ``TestClient`` opens a throwaway event loop per request unless it is used as a
    context manager, and the context-manager form additionally pays the app
    lifespan (~10s here). The driver task started by ``/live/runner/start``
    outlives the request that created it, so these tests drive the app on a
    single long-lived loop — which is what uvicorn does in production.

    The base URL is loopback so the DNS-rebinding Host check and the loopback
    auth trust behave exactly as they do for ``_client`` above.
    """
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_server.app, client=("127.0.0.1", 50000)),
        base_url="http://127.0.0.1:50000",
    )


class _StubScheduler:
    """Scheduler stub mirroring ``Scheduler``'s ``_task`` / async ``stop`` contract."""

    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self.stop_calls = 0
        self.ticks = 0

    def start(self) -> None:
        async def _loop() -> None:
            while True:  # a real scheduler never returns on its own
                self.ticks += 1
                await asyncio.sleep(0)

        self._task = asyncio.get_running_loop().create_task(_loop(), name="live-scheduler")

    async def stop(self) -> None:
        self.stop_calls += 1
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()


def _install_runner(monkeypatch, broker: str = "robinhood", *, start: bool = True):
    """Stub the runner factory; return the ``_StubScheduler`` the runner starts."""
    monkeypatch.setattr(
        api_server, "_active_mandate_state", lambda b: _valid_mandate_state(b)
    )
    scheduler = _StubScheduler()

    class _StubRunner:
        def __init__(self) -> None:
            self.broker = broker
            self._scheduler = scheduler

        def run_loop(self, jobs=None) -> None:  # sync, non-blocking, returns None
            if start:
                self._scheduler.start()

    monkeypatch.setattr(api_server, "_runner_factory", lambda b: _StubRunner())
    return scheduler


async def _await_ticks(scheduler: _StubScheduler, ticks: int = 5) -> None:
    """Let the scheduler loop actually run, so "it is still alive" is meaningful."""
    for _ in range(200):
        if scheduler.ticks >= ticks:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("scheduler loop never ticked")


async def _await_stopped(scheduler: _StubScheduler) -> None:
    """Let the cancelled driver run its teardown."""
    for _ in range(200):
        if not scheduler.running:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("scheduler loop never stopped")


def test_runner_start_keeps_task_registered_while_scheduler_runs(
    tmp_path: Path, monkeypatch
) -> None:
    """The driver task must stay registered for as long as the scheduler ticks.

    Regression: ``_drive_runner`` returned as soon as ``run_loop`` did, so the
    ``add_done_callback`` popped ``_runner_tasks[broker]`` while the scheduler
    was still firing — leaving the runner unstoppable from the API.
    """
    _live_env(tmp_path, monkeypatch)
    scheduler = _install_runner(monkeypatch)

    async def scenario() -> None:
        async with _live_client() as client:
            response = await client.post("/live/runner/start", json={"broker": "robinhood"})
            assert response.status_code == 200, response.text
            assert response.json() == {
                "broker": "robinhood",
                "started": True,
                "already_running": False,
            }

            await _await_ticks(scheduler)
            # Give a wrongly-finished driver every chance to finish and unregister.
            for _ in range(5):
                await asyncio.sleep(0)

            task = api_server._runner_tasks.get("robinhood")
            assert task is not None, (
                "driver task was unregistered while the scheduler was still running"
            )
            assert not task.done(), "driver task completed while the scheduler was running"
            assert scheduler.running is True

            await client.post("/live/runner/stop", json={"broker": "robinhood"})
            await _await_stopped(scheduler)

    asyncio.run(scenario())


def test_runner_stop_actually_stops_the_scheduler(tmp_path: Path, monkeypatch) -> None:
    """``POST /live/runner/stop`` must stop the scheduler, not just a wrapper task.

    Regression: the stop endpoint reported ``was_running: false`` because the
    wrapper task had already been popped, and the scheduler kept ticking.
    """
    _live_env(tmp_path, monkeypatch)
    scheduler = _install_runner(monkeypatch)

    async def scenario() -> None:
        async with _live_client() as client:
            started = await client.post("/live/runner/start", json={"broker": "robinhood"})
            assert started.status_code == 200, started.text
            await _await_ticks(scheduler)

            response = await client.post("/live/runner/stop", json={"broker": "robinhood"})

            assert response.status_code == 200
            assert response.json() == {"broker": "robinhood", "stopped": True, "was_running": True}

            await _await_stopped(scheduler)
            assert scheduler.running is False, "scheduler loop still running after stop"
            assert scheduler.stop_calls == 1, "scheduler.stop() was not awaited"
            assert "robinhood" not in api_server._runner_tasks

    asyncio.run(scenario())


def test_runner_stop_is_idempotent_after_a_real_stop(tmp_path: Path, monkeypatch) -> None:
    """A second stop on an already-stopped runner stays a no-op."""
    _live_env(tmp_path, monkeypatch)
    scheduler = _install_runner(monkeypatch)

    async def scenario() -> None:
        async with _live_client() as client:
            await client.post("/live/runner/start", json={"broker": "robinhood"})
            await _await_ticks(scheduler)

            first = await client.post("/live/runner/stop", json={"broker": "robinhood"})
            assert first.json() == {"broker": "robinhood", "stopped": True, "was_running": True}
            await _await_stopped(scheduler)

            second = await client.post("/live/runner/stop", json={"broker": "robinhood"})
            assert second.status_code == 200
            assert second.json() == {
                "broker": "robinhood",
                "stopped": False,
                "was_running": False,
            }
            assert scheduler.stop_calls == 1

    asyncio.run(scenario())


def test_runner_start_without_a_scheduler_task_still_returns(
    tmp_path: Path, monkeypatch
) -> None:
    """``run_loop`` declining to start (no mandate) must not pin the driver forever.

    ``LiveRunner.run_loop`` returns early when the mandate is missing or expired
    and never calls ``Scheduler.start()``, so there is no task to await and the
    driver must settle immediately — preserving the legacy behaviour.
    """
    _live_env(tmp_path, monkeypatch)
    scheduler = _install_runner(monkeypatch, start=False)

    async def scenario() -> None:
        async with _live_client() as client:
            response = await client.post("/live/runner/start", json={"broker": "robinhood"})
            assert response.status_code == 200, response.text

            for _ in range(200):
                if "robinhood" not in api_server._runner_tasks:
                    break
                await asyncio.sleep(0.001)
            assert "robinhood" not in api_server._runner_tasks
            assert scheduler.stop_calls == 0

            stopped = await client.post("/live/runner/stop", json={"broker": "robinhood"})
            assert stopped.json() == {
                "broker": "robinhood",
                "stopped": False,
                "was_running": False,
            }

    asyncio.run(scenario())


def test_drive_runner_awaits_an_async_run_loop_and_stops_the_scheduler() -> None:
    """An ``async def run_loop`` is awaited, and its scheduler is still stopped."""
    scheduler = _StubScheduler()

    class _AsyncRunner:
        broker = "robinhood"

        def __init__(self) -> None:
            self._scheduler = scheduler

        async def run_loop(self, jobs=None) -> None:
            await asyncio.sleep(0)
            self._scheduler.start()

    runner = _AsyncRunner()

    async def scenario() -> None:
        task = asyncio.ensure_future(api_server._drive_runner(runner))
        for _ in range(200):
            if scheduler.ticks >= 5:
                break
            await asyncio.sleep(0.001)
        assert scheduler.ticks >= 5, "scheduler loop never started"

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    assert scheduler.running is False
    assert scheduler.stop_calls == 1


# --------------------------------------------------------------------------- #
# C1 — propose_mandate_profiles tool_result -> mandate.proposal SSE frame
# --------------------------------------------------------------------------- #


def _seed_proposal(tmp_path: Path, proposal_id: str, broker: str = "robinhood") -> dict:
    proposal = {
        "type": "mandate.proposal",
        "proposal_id": proposal_id,
        "session_id": "s1",
        "profiles": [{"ordinal": 1, "label": "稳健", "max_order_usd": 250}],
    }
    proposals_dir = tmp_path / ".vibe-trading" / "live" / broker / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    (proposals_dir / f"{proposal_id}.json").write_text(
        json.dumps(proposal), encoding="utf-8"
    )
    return proposal


def test_c1_relay_builds_mandate_proposal_frame(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    proposal_id = "mp_" + "1" * 32
    _seed_proposal(tmp_path, proposal_id)

    event = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={
            "tool": "propose_mandate_profiles",
            "status": "ok",
            "preview": json.dumps({"proposal_id": proposal_id})[:200],
        },
    )

    frame = api_server._mandate_proposal_frame_from_tool_result(event)

    assert frame is not None
    assert "mandate.proposal" in frame
    assert proposal_id in frame


def test_c1_relay_ignores_non_propose_events(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)

    # A different tool's result: no relay.
    other = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={"tool": "run_backtest", "status": "ok", "preview": "{}"},
    )
    assert api_server._mandate_proposal_frame_from_tool_result(other) is None

    # A non-tool_result event: no relay.
    thinking = SimpleNamespace(event_type="thinking", session_id="s1", data={})
    assert api_server._mandate_proposal_frame_from_tool_result(thinking) is None


def test_c1_relay_returns_none_when_proposal_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    # Preview references an id with no persisted proposal on disk.
    event = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={
            "tool": "propose_mandate_profiles",
            "status": "ok",
            "preview": json.dumps({"proposal_id": "mp_" + "2" * 32})[:200],
        },
    )
    assert api_server._mandate_proposal_frame_from_tool_result(event) is None


# --------------------------------------------------------------------------- #
# Real runner factory (R-INT): no more TypeError on the default path
# --------------------------------------------------------------------------- #


def test_build_live_runner_no_broker_configured_raises_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)
    # No robinhood MCP server in the default config → clean 503-class error,
    # NOT a TypeError (the audit's CRITICAL finding).
    import pytest

    with pytest.raises(api_server.LiveRunnerUnavailable):
        api_server._build_live_runner("robinhood")


def test_build_live_runner_wires_a_real_runner(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)

    # Stub the broker adapter + session service so no real broker/agent is hit.
    class _StubAdapter:
        def call_tool(self, name, args):
            return {"status": "ok", "result": {}}

    class _StubSession:
        session_id = "live-sess-1"

    class _StubSvc:
        event_bus = SimpleNamespace(emit=lambda *a, **k: None)

        def create_session(self, title=""):
            return _StubSession()

        async def send_message(self, sid, content, **kw):
            return {"message_id": "m1", "attempt_id": "a1"}

    monkeypatch.setattr(api_server, "_live_broker_adapter", lambda broker: _StubAdapter())
    monkeypatch.setattr(api_server, "_get_session_service", lambda: _StubSvc())
    monkeypatch.setattr(api_server, "_mandate_account_ref", lambda broker: "5QR00001")

    runner = api_server._build_live_runner("robinhood")
    # Constructs without TypeError and exposes the R2 contract.
    assert runner.broker == "robinhood"
    assert hasattr(runner, "run_once") and hasattr(runner, "run_loop")
    assert runner.runner_id == "robinhood"


def test_runner_start_returns_503_when_broker_unavailable(tmp_path, monkeypatch) -> None:
    client = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api_server, "_active_mandate_state", lambda broker: _valid_mandate_state(broker)
    )

    def _boom(broker):
        raise api_server.LiveRunnerUnavailable("no MCP server configured for live broker 'robinhood'")

    monkeypatch.setattr(api_server, "_runner_factory", _boom)

    response = client.post("/live/runner/start", json={"broker": "robinhood"})
    assert response.status_code == 503
    assert "configured" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# H5 — per-order live.action SSE relay (reload full record from the ledger)
# --------------------------------------------------------------------------- #


def _seed_ledger(tmp_path: Path, record: dict) -> None:
    ledger = tmp_path / ".vibe-trading" / "live" / "audit.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def test_live_action_relay_builds_frame_from_guard_result(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    audit_id = "la_01abcDEF"
    _seed_ledger(tmp_path, {"audit_id": audit_id, "kind": "order_placed", "outcome": "accepted",
                            "broker_request": {"symbol": "NVDA"}})

    event = SimpleNamespace(
        event_type="tool_result",
        session_id="s1",
        data={
            "tool": "mcp_robinhood_place_equity_order",
            "status": "ok",
            "preview": json.dumps({"status": "ok", "live_action": {"audit_id": audit_id}})[:200],
        },
    )

    frame = api_server._live_action_frame_from_tool_result(event)
    assert frame is not None
    assert "live.action" in frame
    assert audit_id in frame and "order_placed" in frame


def test_live_action_relay_ignores_non_live_results(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    # A normal tool_result with no live_action marker → no relay.
    event = SimpleNamespace(
        event_type="tool_result", session_id="s1",
        data={"tool": "run_backtest", "status": "ok", "preview": "{}"},
    )
    assert api_server._live_action_frame_from_tool_result(event) is None


# --------------------------------------------------------------------------- #
# H9 — broker-derived ceilings hook
# --------------------------------------------------------------------------- #


def test_fetch_broker_ceilings_derives_from_account(tmp_path, monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class _StubAdapter:
        def call_tool(self, name, args):
            calls.append((name, args))
            return rh.portfolio(total_value="9000.00", cash="1000.00", buying_power="4200.00")

    monkeypatch.setattr(api_server, "_live_broker_adapter", lambda broker: _StubAdapter())
    ceilings = api_server._fetch_broker_ceilings("robinhood", "5QR00001")
    # The read is scoped to the account being bound, and the value comes from
    # Robinhood's nested buying_power.buying_power, not a top-level key.
    assert calls == [("get_portfolio", {"account_number": "5QR00001"})]
    assert ceilings == {
        "account_funding_usd": 4200.0,
        "max_order_notional_usd": 4200.0,
        "max_total_exposure_usd": 4200.0,
    }


def test_fetch_broker_ceilings_falls_back_to_none(tmp_path, monkeypatch) -> None:
    # Unavailable broker → None (commit falls back to the proposal snapshot).
    def _unavail(broker):
        raise api_server.LiveRunnerUnavailable("x")

    monkeypatch.setattr(api_server, "_live_broker_adapter", _unavail)
    assert api_server._fetch_broker_ceilings("robinhood") is None


def test_build_live_runner_reads_normalize_adapter_payloads(tmp_path, monkeypatch) -> None:
    # The halt-sweep + reconcile reads must receive broker RECORDS, not the
    # adapter's {status: ok, data: ...} envelope — a successful positions
    # read must decode into the records list, not be rejected as a dict.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)

    class _StubAdapter:
        def __init__(self):
            self.calls: list[tuple[str, dict]] = []

        def call_tool(self, name, args):
            self.calls.append((name, args))
            if name.endswith("positions"):
                return rh.positions([rh.position("NVDA", "3")])
            if name.endswith("orders"):
                return rh.envelope("get_equity_orders", {"orders": [{"order_id": "o1"}]})
            if name.endswith("portfolio"):
                return rh.portfolio(buying_power="4200.00")
            raise AssertionError(f"unexpected tool {name}")

    class _StubSvc:
        session_id = "live-sess-2"
        event_bus = SimpleNamespace(emit=lambda *a, **k: None)

        def create_session(self, title=""):
            return self

        async def send_message(self, sid, content, **kw):
            return {"message_id": "m1", "attempt_id": "a1"}

    adapter = _StubAdapter()
    monkeypatch.setattr(api_server, "_live_broker_adapter", lambda broker: adapter)
    monkeypatch.setattr(api_server, "_get_session_service", lambda: _StubSvc())
    monkeypatch.setattr(api_server, "_mandate_account_ref", lambda broker: "5QR00001")

    runner = api_server._build_live_runner("robinhood")
    assert runner._read_positions() == [
        {"symbol": "NVDA", "quantity": "3", "average_cost": "100.00", "broker_type": "unobserved", "qty": "3"}
    ]
    assert runner._read_open_orders() == [{"order_id": "o1"}]
    assert runner._read_balance()["buying_power"] == "4200.00"
    assert {args.get("account_number") for _, args in adapter.calls} == {"5QR00001"}


def test_build_live_runner_read_error_envelope_raises(tmp_path, monkeypatch) -> None:
    # An error envelope at the boundary must raise (fail-closed: reconcile
    # ticks error, the sweep records a structured read error) — never be
    # consumed as an empty/mangled record list.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path), raising=False)
    monkeypatch.setattr(api_server, "_runner_factory", None, raising=False)

    class _StubAdapter:
        def call_tool(self, name, args):
            return {
                "status": "error",
                "server": "robinhood",
                "remote_tool": name,
                "tool": name,
                "error": "connection reset while reading positions",
                "error_type": "ConnectionError",
            }

    class _StubSvc:
        session_id = "live-sess-3"
        event_bus = SimpleNamespace(emit=lambda *a, **k: None)

        def create_session(self, title=""):
            return self

        async def send_message(self, sid, content, **kw):
            return {"message_id": "m1", "attempt_id": "a1"}

    monkeypatch.setattr(api_server, "_live_broker_adapter", lambda broker: _StubAdapter())
    monkeypatch.setattr(api_server, "_get_session_service", lambda: _StubSvc())
    monkeypatch.setattr(api_server, "_mandate_account_ref", lambda broker: "5QR00001")

    runner = api_server._build_live_runner("robinhood")
    with pytest.raises(RuntimeError, match="connection reset while reading positions"):
        runner._read_positions()
