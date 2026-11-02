"""Windows display-affinity helpers for hiding a window from screen capture."""

import ctypes
import sys

# Windows 10 2004 (build 19041) added WDA_EXCLUDEFROMCAPTURE: the window is
# displayed only on a physical monitor, and every capture path -- window and
# desktop sharing, recording, and screenshots -- shows the content behind it.
_WDA_NONE = 0x0
_WDA_EXCLUDEFROMCAPTURE = 0x11
_MINIMUM_BUILD = 19041


def is_supported() -> bool:
    """Return whether this OS can exclude a window from screen capture."""
    if sys.platform != "win32":
        return False
    get_windows_version = getattr(sys, "getwindowsversion", None)
    if get_windows_version is None:
        return False
    build = int(getattr(get_windows_version(), "build", 0))
    return build >= _MINIMUM_BUILD


def set_capture_visible(hwnd: int, *, visible: bool) -> bool:
    """Show or hide one top-level window from every screen-capture path.

    Returns True when the requested display affinity was accepted, and False
    when the platform cannot restrict capture or the window rejects the
    change. A failed call leaves the window's current affinity untouched.
    """
    user32 = _load_user32()
    if user32 is None:
        return False
    affinity = _WDA_NONE if visible else _WDA_EXCLUDEFROMCAPTURE
    try:
        return bool(user32.SetWindowDisplayAffinity(hwnd, affinity))
    except OSError:
        return False


def _load_user32() -> ctypes.WinDLL | None:
    """Return a typed user32 handle, or None when capture hiding is unavailable."""
    if not is_supported():
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
    except OSError:  # pragma: no cover - user32 is always present on win32
        return None
    from ctypes import wintypes  # importable only after the platform check

    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.SetWindowDisplayAffinity.restype = wintypes.BOOL
    return user32
