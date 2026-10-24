"""Atomic replacement with bounded retries for Windows sharing conflicts."""
from __future__ import annotations

import os
import time
from os import PathLike

_WINDOWS_SHARING_ERRORS = frozenset({5, 32, 33})
_RETRY_DELAYS = (0.01, 0.02, 0.04, 0.08, 0.16)


def replace_atomic(src: str | PathLike[str], dst: str | PathLike[str]) -> None:
    """Replace a closed temporary file without ever deleting the current target.

    Windows readers can briefly deny replacement by omitting FILE_SHARE_DELETE.
    Retry only those sharing/access failures, at most six attempts and 0.31s of
    total backoff. The final exception is preserved; permissions are unchanged.
    The caller retains ownership of flushing, fsync, and temporary-file cleanup.
    """
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            os.replace(src, dst)
            return
        except OSError as error:
            if (os.name != "nt"
                    or getattr(error, "winerror", None) not in _WINDOWS_SHARING_ERRORS
                    or attempt == len(_RETRY_DELAYS)):
                raise
            time.sleep(_RETRY_DELAYS[attempt])
