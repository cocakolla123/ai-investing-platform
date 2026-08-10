"""polymarket_broker_for (ADR-0012): the ONE precondition chain to real
Polymarket execution. Sim broker for every environment except a
fully-satisfied LIVE chain; each individually missing precondition falls back
to sim with a loud structlog warning. No network anywhere."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import structlog.testing
from orchestration.assets.pm_trading import polymarket_broker_for

import dynamo.execution.polymarket_broker as broker_mod
from dynamo.config.settings import Settings
from dynamo.execution.gate import Readiness, sign_approval_body
from dynamo.execution.polymarket_broker import PolymarketBroker

KEY = "test-approval-key"
READY = Readiness(
    clean_paper_months=4.0, calibration_ok=True, risk_suite_green=True, backup_drill_ok=True
)

SIM_SENTINEL = object()


def sim_factory() -> Any:
    return SIM_SENTINEL


@pytest.fixture
def wallet_key_file(tmp_path: Path) -> Path:
    f = tmp_path / "polymarket-wallet.key"
    f.write_text("0xdeadbeefwalletkey")
    return f


def make_settings(
    tmp_path: Path,
    *,
    env: str = "PAPER",
    live_enabled: bool = False,
    credentialed: bool = False,
    wallet_key_file: Path | None = None,
) -> Settings:
    values: dict[str, Any] = {
        "DYNAMO_ENV": env,
        "polymarket_live_enabled": live_enabled,
        "DYNAMO_APPROVALS_DIR": str(tmp_path / "approvals"),
        "DYNAMO_KILL_SWITCH_PATH": str(tmp_path / "KILL_SWITCH"),
        "_env_file": None,
    }
    if credentialed:
        values.update(
            POLYMARKET_API_KEY="key",
            POLYMARKET_API_SECRET="secret",
            POLYMARKET_PASSPHRASE="pass",
            POLYMARKET_WALLET_KEY_PATH=str(wallet_key_file),
            POLYMARKET_FUNDER_ADDRESS="0xfunder",
        )
    return Settings(**values)  # type: ignore[call-arg]


def write_approval(
    settings: Settings,
    *,
    expiry: date = date(2027, 1, 1),
    key: str = KEY,
    scope: tuple[str, ...] = ("all",),
) -> None:
    from orchestration.resources import utcnow

    approvals = Path(settings.approvals_dir)
    approvals.mkdir(parents=True, exist_ok=True)
    body = {"scope": list(scope), "expiry": expiry}
    signature = sign_approval_body(body, key.encode())
    lines = ["scope:"] + [f"  - {s}" for s in scope] + [
        f"expiry: {expiry.isoformat()}",
        f"signature: {signature}",
    ]
    path = approvals / f"live-{utcnow().date().isoformat()}.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_factory(settings: Settings, readiness: Readiness | None = None):
    with structlog.testing.capture_logs() as logs:
        broker, live_active = polymarket_broker_for(
            settings, sim_broker_factory=sim_factory, readiness=readiness
        )
    warnings = [e for e in logs if e.get("log_level") == "warning"]
    return broker, live_active, warnings


@pytest.fixture
def live_ready(monkeypatch, tmp_path, wallet_key_file):
    """Everything satisfied: LIVE + enabled + configured + passing gate."""
    monkeypatch.setenv("DYNAMO_APPROVAL_KEY", KEY)
    monkeypatch.setattr(broker_mod, "_clob_import_error", lambda: None)
    settings = make_settings(
        tmp_path, env="LIVE", live_enabled=True, credentialed=True,
        wallet_key_file=wallet_key_file,
    )
    write_approval(settings)
    return settings


class TestSimIsTheDefault:
    @pytest.mark.parametrize("env", ["DEV", "TEST", "PAPER"])
    def test_non_live_env_returns_sim_silently(self, tmp_path, env) -> None:
        broker, live_active, warnings = run_factory(make_settings(tmp_path, env=env))
        assert broker is SIM_SENTINEL
        assert not live_active
        assert warnings == []  # the expected paper state: no noise

    @pytest.mark.parametrize("env", ["DEV", "TEST", "PAPER", "LIVE"])
    def test_regression_sim_for_every_env_without_full_chain(self, tmp_path, env) -> None:
        # Even with the master switch on, no env yields the real broker
        # without credentials + gate: sim always, fail closed.
        settings = make_settings(tmp_path, env=env, live_enabled=True)
        broker, live_active, _warnings = run_factory(settings, readiness=READY)
        assert broker is SIM_SENTINEL
        assert not live_active

    def test_live_env_alone_returns_sim_with_warning(self, tmp_path) -> None:
        broker, live_active, warnings = run_factory(make_settings(tmp_path, env="LIVE"))
        assert broker is SIM_SENTINEL and not live_active
        assert any("polymarket_live_enabled_false" in w.get("reason", "") for w in warnings)

    def test_enabled_outside_live_returns_sim_with_warning(self, tmp_path) -> None:
        settings = make_settings(tmp_path, env="PAPER", live_enabled=True)
        broker, live_active, warnings = run_factory(settings)
        assert broker is SIM_SENTINEL and not live_active
        assert any(w.get("reason", "") == "env_not_live" for w in warnings)


class TestEachMissingPreconditionFallsBack:
    def test_unconfigured_broker_falls_back(self, tmp_path) -> None:
        settings = make_settings(tmp_path, env="LIVE", live_enabled=True)
        broker, live_active, warnings = run_factory(settings, readiness=READY)
        assert broker is SIM_SENTINEL and not live_active
        assert any(w.get("reason", "").startswith("not_configured") for w in warnings)

    def test_kill_switch_falls_back(self, live_ready) -> None:
        Path(live_ready.kill_switch_path).write_text("engaged")
        broker, live_active, warnings = run_factory(live_ready, readiness=READY)
        assert broker is SIM_SENTINEL and not live_active
        assert any("kill_switch_engaged" in w.get("reason", "") for w in warnings)

    def test_missing_approval_falls_back(self, live_ready) -> None:
        approvals = Path(live_ready.approvals_dir)
        for f in approvals.glob("*.yaml"):
            f.unlink()
        broker, live_active, warnings = run_factory(live_ready, readiness=READY)
        assert broker is SIM_SENTINEL and not live_active
        assert any("approval_missing" in w.get("reason", "") for w in warnings)

    def test_bad_signature_falls_back(self, live_ready) -> None:
        write_approval(live_ready, key="wrong-key")
        broker, live_active, warnings = run_factory(live_ready, readiness=READY)
        assert broker is SIM_SENTINEL and not live_active
        assert any("approval_bad_signature" in w.get("reason", "") for w in warnings)

    def test_expired_approval_falls_back(self, live_ready) -> None:
        write_approval(live_ready, expiry=date(2020, 1, 1))
        broker, live_active, warnings = run_factory(live_ready, readiness=READY)
        assert broker is SIM_SENTINEL and not live_active
        assert any("approval_expired" in w.get("reason", "") for w in warnings)

    def test_scoped_approval_without_all_falls_back(self, live_ready) -> None:
        write_approval(live_ready, scope=("KALSHI:FED-26DEC",))
        broker, live_active, warnings = run_factory(live_ready, readiness=READY)
        assert broker is SIM_SENTINEL and not live_active
        assert any("approval_out_of_scope" in w.get("reason", "") for w in warnings)

    def test_unmet_readiness_falls_back(self, live_ready) -> None:
        # Readiness defaults to unset → refuse, same as the Kalshi flow.
        broker, live_active, warnings = run_factory(live_ready, readiness=None)
        assert broker is SIM_SENTINEL and not live_active
        assert any("readiness_unmet" in w.get("reason", "") for w in warnings)


class TestFullChain:
    def test_full_live_chain_returns_real_broker(self, live_ready) -> None:
        broker, live_active, warnings = run_factory(live_ready, readiness=READY)
        assert isinstance(broker, PolymarketBroker)
        assert live_active
        # Activation is itself loudly logged.
        assert any(w["event"] == "polymarket_live_broker_active" for w in warnings)
