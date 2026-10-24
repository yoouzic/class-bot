"""Bound owned streaming writes before bytes reach the cache filesystem.

The cache mutation lock excludes concurrent Classbot writers. Each writer takes
one allowance, including the old snapshot and any temporary files already on
disk; seeking back to patch a ZIP header does not consume space twice.
"""

from __future__ import annotations

from .cache_manager import CacheLimitError, cache_available_bytes


class BinaryQuotaWriter:
    def __init__(self, output, path):
        self.output = output
        self.allowance = cache_available_bytes(path)

    def write(self, value):
        end = self.output.tell() + len(value)
        if self.allowance is not None and end > self.allowance:
            raise CacheLimitError("缓存总占用将超过上限；请清理不用的索引或提高上限后重试。原始资料与上一完整快照已保留。")
        return self.output.write(value)

    def __getattr__(self, name):
        return getattr(self.output, name)


class TextQuotaWriter:
    def __init__(self, output, path, *, maximum=None):
        self.output = output
        self.allowance = cache_available_bytes(path)
        self.maximum = maximum
        self.written = 0

    def write(self, value):
        required = self.written + len(value.encode("utf-8"))
        if self.maximum is not None and required > self.maximum:
            raise ValueError(f"index cache size exceeds {self.maximum} bytes")
        if self.allowance is not None and required > self.allowance:
            raise CacheLimitError("缓存总占用将超过上限；请清理不用的索引或提高上限后重试。原始资料与上一完整快照已保留。")
        result = self.output.write(value)
        self.written = required
        return result

    def __getattr__(self, name):
        return getattr(self.output, name)
