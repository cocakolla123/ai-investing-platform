"""Stage-7 human approval is a real HMAC check, not string truthiness.

Design: doc 08 §7 (promotion pipeline), doc 13 §4 risk #3. The regression this
locks down: a YAML file containing nothing but ``signature: x`` used to pass
the shadow -> production gate.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from dynamo.cli import approve, promotion
from dynamo.execution.gate import APPROVAL_KEY_ENV, sign_approval_body

KEY = "test-approval-key-0123456789abcdef"


def _git_init(tmp_path: Path) -> None:
    """A clean git tree — the promotion CLI refuses a dirty one."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    # Everything the tests scribble (registry/evidence/approval files) is
    # ignored, so the tree stays clean and the CLI's dirty-tree refusal —
    # which is a *different* refusal — never masks the approval check.
    (tmp_path / ".gitignore").write_text("*\n!.gitignore\n")
    subprocess.run(["git", "add", "-A", "-f", ".gitignore"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=tmp_path, check=True)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git_init(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _registry_at_shadow(path: Path, target: str) -> None:
    path.write_text(
        json.dumps({"status": {target: "shadow"}, "trials": {}, "records": []})
    )


def _evidence(path: Path) -> None:
    path.write_text(json.dumps({"engine": "event_driven", "config_hash": "abc"}))


def _run_stage7(workspace: Path, approval: Path | None, target: str = "strat-a") -> int:
    registry = workspace / "registry.json"
    _registry_at_shadow(registry, target)
    evidence = workspace / "evidence.json"
    _evidence(evidence)
    argv = [
        "run", "--target", target, "--stage", "7",
        "--evidence", str(evidence), "--registry", str(registry),
        "--actor", "owner", "--evidence-bundle", "artifacts/backtests/run-1",
    ]
    if approval is not None:
        argv += ["--approval", str(approval)]
    return promotion.main(argv)


def _write_approval(path: Path, body: dict[str, Any], signature: str) -> None:
    path.write_text(yaml.safe_dump({**body, "signature": signature}, sort_keys=True))


def _body(target: str, days: int = 30) -> dict[str, Any]:
    return {
        "kind": "promotion",
        "target_id": target,
        "scope": [target],
        "expiry": (datetime.now(UTC).date() + timedelta(days=days)).isoformat(),
        "approver": "owner",
    }


def test_bogus_signature_is_refused(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    approval = workspace / "approval.yaml"
    _write_approval(approval, _body("strat-a"), "x")

    assert _run_stage7(workspace, approval) == 2
    out = capsys.readouterr().out
    assert "refused" in out and "approval_bad_signature" in out
    # The registry must NOT have moved.
    assert json.loads((workspace / "registry.json").read_text())["status"]["strat-a"] == "shadow"


def test_string_truthiness_no_longer_passes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact historical bug: target_id + any signature value."""
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    approval = workspace / "approval.yaml"
    approval.write_text("target_id: strat-a\nscope: [strat-a]\nexpiry: 2099-01-01\nsignature: x\n")
    assert _run_stage7(workspace, approval) == 2


def test_correctly_signed_approval_passes(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    body = _body("strat-a")
    approval = workspace / "approval.yaml"
    _write_approval(approval, body, sign_approval_body(body, KEY.encode()))

    assert _run_stage7(workspace, approval) == 0
    assert "PASS" in capsys.readouterr().out
    assert (
        json.loads((workspace / "registry.json").read_text())["status"]["strat-a"] == "production"
    )


def test_edited_body_after_signing_is_refused(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    body = _body("strat-a")
    signature = sign_approval_body(body, KEY.encode())
    tampered = {**body, "approver": "someone-else"}
    approval = workspace / "approval.yaml"
    _write_approval(approval, tampered, signature)
    assert _run_stage7(workspace, approval) == 2


def test_approval_for_another_target_is_out_of_scope(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    body = _body("strat-b")
    approval = workspace / "approval.yaml"
    _write_approval(approval, body, sign_approval_body(body, KEY.encode()))
    assert _run_stage7(workspace, approval, target="strat-a") == 2
    assert "approval_out_of_scope" in capsys.readouterr().out


def test_expired_approval_is_refused(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    body = _body("strat-a", days=-1)
    approval = workspace / "approval.yaml"
    _write_approval(approval, body, sign_approval_body(body, KEY.encode()))
    assert _run_stage7(workspace, approval) == 2


def test_missing_key_is_loud_and_instructive(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(APPROVAL_KEY_ENV, raising=False)
    body = _body("strat-a")
    approval = workspace / "approval.yaml"
    _write_approval(approval, body, sign_approval_body(body, KEY.encode()))

    assert _run_stage7(workspace, approval) == 2
    out = capsys.readouterr().out
    assert APPROVAL_KEY_ENV in out
    assert "dynamo.cli.approve keygen" in out  # tells the operator what to do


def test_missing_approval_file_is_refused(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    assert _run_stage7(workspace, workspace / "nope.yaml") == 2
    assert "no such approval artifact" in capsys.readouterr().out


def test_no_approval_flag_still_refuses_stage7(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    assert _run_stage7(workspace, None) == 2
    assert "requires a signed approval artifact" in capsys.readouterr().out


# --- the approvals tool itself ------------------------------------------


def test_approve_sign_then_promotion_accepts_it(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the tool that writes the artifact and the gate that reads it agree."""
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    out = workspace / "approvals" / "promotion-strat-a.yaml"
    expiry = (datetime.now(UTC).date() + timedelta(days=10)).isoformat()
    assert approve.main(
        ["sign", "--kind", "promotion", "--target", "strat-a",
         "--expiry", expiry, "--approver", "owner", "--out", str(out)]
    ) == 0
    assert out.exists()
    assert approve.main(["verify", "--file", str(out), "--target", "strat-a"]) == 0
    assert _run_stage7(workspace, out) == 0


def test_approve_refuses_without_key(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(APPROVAL_KEY_ENV, raising=False)
    assert approve.main(
        ["sign", "--kind", "promotion", "--target", "x",
         "--expiry", "2099-01-01", "--approver", "owner"]
    ) == 2
    assert "keygen" in capsys.readouterr().err


def test_approve_keygen_emits_a_usable_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert approve.main(["keygen"]) == 0
    captured = capsys.readouterr()
    assert len(captured.out.strip()) == 64  # 32 random bytes, hex
    assert APPROVAL_KEY_ENV in captured.err


def test_approve_verify_detects_tampering(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    out = workspace / "a.yaml"
    assert approve.main(
        ["sign", "--target", "strat-a", "--expiry", "2099-01-01",
         "--approver", "owner", "--out", str(out)]
    ) == 0
    data = yaml.safe_load(out.read_text())
    data["expiry"] = "2100-01-01"
    out.write_text(yaml.safe_dump(data))
    assert approve.main(["verify", "--file", str(out), "--target", "strat-a"]) == 1


def test_approve_sign_refuses_to_clobber(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(APPROVAL_KEY_ENV, KEY)
    out = workspace / "a.yaml"
    args = ["sign", "--target", "s", "--expiry", "2099-01-01", "--approver", "o", "--out", str(out)]
    assert approve.main(args) == 0
    assert approve.main(args) == 2
    assert approve.main([*args, "--force"]) == 0
