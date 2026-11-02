"""Finish HTTPX's nested response iterators in their owning task."""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

import httpx

_TAIL_TIMEOUT_SECONDS = 0.25


class _ResponseLines:
    def __init__(self, iterator: AsyncIterator[str]):
        self._iterator = iterator
        self._started = False
        self._exhausted = False
        self._deadline: float | None = None

    def __aiter__(self):
        return self

    def mark_terminal(self) -> None:
        if self._deadline is None:
            self._deadline = asyncio.get_running_loop().time() + _TAIL_TIMEOUT_SECONDS

    async def __anext__(self) -> str:
        if self._exhausted:
            raise StopAsyncIteration
        self._started = True
        try:
            if self._deadline is None:
                return await anext(self._iterator)
            timeout = asyncio.timeout_at(self._deadline)
            try:
                async with timeout:
                    return await anext(self._iterator)
            except TimeoutError:
                if not timeout.expired():
                    raise
                # Cancellation entered the active HTTP read and unwound its
                # nested iterators, instead of abandoning them to GC tasks.
                self._exhausted = True
                raise StopAsyncIteration from None
        except httpx.TransportError:
            self._exhausted = True
            if self._deadline is not None:
                # The application already accepted DONE or selected fallback;
                # an error in discarded bytes must not replace that decision.
                raise StopAsyncIteration from None
            raise
        except BaseException:
            self._exhausted = True
            raise

    async def aclose(self) -> None:
        try:
            if self._started and not self._exhausted:
                timeout = asyncio.timeout(_TAIL_TIMEOUT_SECONDS)
                try:
                    async with timeout:
                        async for _ in self._iterator:
                            pass
                except TimeoutError:
                    if not timeout.expired():
                        raise
                finally:
                    self._exhausted = True
        finally:
            await self._iterator.aclose()


@asynccontextmanager
async def owned_sse_lines(response):
    """Discard a bounded tail on early exit; never replace the body's exception.

    Closing only ``aiter_lines`` abandons HTTPX's nested async generators.
    Exhausting the active iterator, or timing out its active read, unwinds the
    installed transport chain without a background task or retained tail body.
    """
    lines = _ResponseLines(response.aiter_lines())
    try:
        yield lines
    except BaseException:
        # Cleanup must preserve the original protocol error, cancellation/close.
        with suppress(BaseException):
            await lines.aclose()
        raise
    else:
        await lines.aclose()
