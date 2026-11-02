"""Tests for the Windows display-affinity wrapper."""

from types import SimpleNamespace

from classbot.ui import capture_affinity


class _RecordingUser32:
    def __init__(self, result: int) -> None:
        self.result = result
        self.calls: list[tuple[int, int]] = []

    def SetWindowDisplayAffinity(self, hwnd: int, affinity: int) -> int:
        self.calls.append((hwnd, affinity))
        return self.result


def test_set_capture_visible_uses_exclude_affinity(monkeypatch):
    fake = _RecordingUser32(1)
    monkeypatch.setattr(capture_affinity, "_load_user32", lambda: fake)

    assert capture_affinity.set_capture_visible(1234, visible=False) is True
    assert capture_affinity.set_capture_visible(1234, visible=True) is True

    assert fake.calls == [(1234, capture_affinity._WDA_EXCLUDEFROMCAPTURE), (1234, 0)]


def test_set_capture_visible_reports_api_rejection(monkeypatch):
    monkeypatch.setattr(
        capture_affinity, "_load_user32", lambda: _RecordingUser32(0)
    )

    assert capture_affinity.set_capture_visible(1, visible=False) is False


def test_set_capture_visible_without_platform_support(monkeypatch):
    monkeypatch.setattr(capture_affinity, "_load_user32", lambda: None)

    assert capture_affinity.set_capture_visible(1, visible=False) is False


def test_load_user32_requires_supported_windows(monkeypatch):
    monkeypatch.setattr(capture_affinity, "is_supported", lambda: False)

    assert capture_affinity._load_user32() is None


def test_is_supported_requires_windows(monkeypatch):
    monkeypatch.setattr(capture_affinity.sys, "platform", "linux")

    assert capture_affinity.is_supported() is False


def test_is_supported_requires_build_19041(monkeypatch):
    old_version = SimpleNamespace(build=18363)
    new_version = SimpleNamespace(build=19041)
    monkeypatch.setattr(capture_affinity.sys, "platform", "win32")

    monkeypatch.setattr(
        capture_affinity.sys,
        "getwindowsversion",
        lambda: old_version,
    )
    assert capture_affinity.is_supported() is False

    monkeypatch.setattr(
        capture_affinity.sys,
        "getwindowsversion",
        lambda: new_version,
    )
    assert capture_affinity.is_supported() is True


def test_is_supported_without_windows_version_helper(monkeypatch):
    monkeypatch.setattr(capture_affinity.sys, "platform", "win32")
    monkeypatch.delattr(capture_affinity.sys, "getwindowsversion", raising=False)

    assert capture_affinity.is_supported() is False
