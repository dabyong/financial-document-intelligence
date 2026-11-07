"""Offline smoke tests. Run: python -m pytest tests/test_public_api_guard.py -q"""
from pathlib import Path
import importlib.util
import pytest

MODULE = Path(__file__).resolve().parents[1] / 'app' / 'public_api_guard.py'
spec = importlib.util.spec_from_file_location('public_api_guard_under_test', MODULE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_default_disabled(monkeypatch):
    monkeypatch.delenv('P5_LIVE_AI_ENABLED', raising=False)
    assert guard.live_ai_enabled() is False
    monkeypatch.setenv('P5_LIVE_AI_ENABLED', 'true')
    assert guard.live_ai_enabled() is True


def test_shared_cooldown(monkeypatch, tmp_path):
    monkeypatch.setenv('P5_APP_DAILY_LIMIT', '3')
    monkeypatch.setenv('P5_APP_COOLDOWN_SECONDS', '60')
    monkeypatch.setattr(guard.time, 'time', lambda: 1_800_000_000)
    assert guard.reserve_request(tmp_path) == (1, 3)
    with pytest.raises(guard.RequestDenied, match='recently'):
        guard.reserve_request(tmp_path)


def test_rolling_budget(monkeypatch, tmp_path):
    monkeypatch.setenv('P5_APP_DAILY_LIMIT', '2')
    monkeypatch.setenv('P5_APP_COOLDOWN_SECONDS', '1')
    clock = [1_800_000_000]
    monkeypatch.setattr(guard.time, 'time', lambda: clock[0])
    assert guard.reserve_request(tmp_path) == (1, 2)
    clock[0] += 2
    assert guard.reserve_request(tmp_path) == (2, 2)
    clock[0] += 2
    with pytest.raises(guard.RequestDenied, match='shared AI request budget'):
        guard.reserve_request(tmp_path)
    clock[0] += 86401
    assert guard.reserve_request(tmp_path) == (1, 2)
