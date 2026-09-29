import asyncio
import inspect
import logging
import os
import sys
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

from PySide6.QtCore import QEventLoop
from PySide6.QtWidgets import QApplication

from classbot.asr.funasr_streaming import FunAsrStreamingRecognizer
from classbot.audio.capture import AudioCapture, AudioCaptureError
from classbot.audio.segmenter import SpeechSegmenter
from classbot.config import Settings
from classbot.course import load_course_profile
from classbot.llm.deepseek_client import DeepSeekClient
from classbot.pipeline import LivePipeline
from classbot.question_detector import QuestionDetector
from classbot.ui.main_window import MainWindow

_LOGGER = logging.getLogger(__name__)
_PRE_ROLL_FRAMES = 10
Factory = Callable[..., Any]


def _make_recognizer(device: str) -> FunAsrStreamingRecognizer:
    profile = load_course_profile(os.getenv("CLASSBOT_COURSE", ""))
    return FunAsrStreamingRecognizer(
        device=device,
        hotwords=profile.hotwords,
        corrections=profile.corrections,
        final_refinement=False,
    )


def _make_detector(subject: str) -> QuestionDetector:
    return QuestionDetector(subject)


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """Constructors for one independent listening session."""

    capture_factory: Factory = AudioCapture
    segmenter_factory: Factory = SpeechSegmenter
    recognizer_factory: Factory = _make_recognizer
    detector_factory: Factory = _make_detector
    client_factory: Factory = DeepSeekClient
    pipeline_factory: Factory = LivePipeline


@dataclass(slots=True)
class Runtime:
    """All resources that belong to one listening session."""

    capture: Any
    segmenter: Any
    recognizer: Any
    detector: Any
    client: Any
    pipeline: Any
    audio_task: asyncio.Task[None] | None = None
    pipeline_task: asyncio.Task[None] | None = None
    cleanup_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cleanup_started: bool = False
    cleaned: bool = False


class RuntimeController:
    """Own one disposable capture/ASR/pipeline runtime at a time."""

    def __init__(
        self,
        settings: Settings,
        window: Any,
        dependencies: RuntimeDependencies | None = None,
        *,
        factories: RuntimeDependencies | None = None,
        capture_factory: Factory | None = None,
        segmenter_factory: Factory | None = None,
        recognizer_factory: Factory | None = None,
        detector_factory: Factory | None = None,
        client_factory: Factory | None = None,
        pipeline_factory: Factory | None = None,
        quit_callback: Callable[[], Any] | None = None,
    ) -> None:
        if dependencies is not None and factories is not None:
            raise TypeError("pass either dependencies or factories, not both")

        base = dependencies or factories or RuntimeDependencies()
        overrides = {
            name: value
            for name, value in {
                "capture_factory": capture_factory,
                "segmenter_factory": segmenter_factory,
                "recognizer_factory": recognizer_factory,
                "detector_factory": detector_factory,
                "client_factory": client_factory,
                "pipeline_factory": pipeline_factory,
            }.items()
            if value is not None
        }
        self.settings = settings
        self.window = window
        self.dependencies = replace(base, **overrides)
        self.quit_callback = quit_callback

        self._lifecycle_lock = asyncio.Lock()
        self._current_runtime: Runtime | None = None
        self._start_task: asyncio.Task[Runtime] | None = None
        self._starting_recognizer: Any | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._shutdown_requested = False
        self._shutdown_complete = False
        self._signals_connected = False
        self._application_signals_connected = False
        self._application_quitting = False
        self._application: Any | None = None
        self._signal_tasks: set[asyncio.Task[Any]] = set()

    @property
    def current_runtime(self) -> Runtime | None:
        return self._current_runtime

    @property
    def runtime(self) -> Runtime | None:
        """Alias useful to callers that treat the active session as runtime."""
        return self._current_runtime

    async def start(self) -> Runtime:
        """Prepare and start a fresh runtime, once per active session."""
        while True:
            wait_task: asyncio.Task[Any] | None = None
            cleanup_runtime: Runtime | None = None
            recheck_after_wait = False

            async with self._lifecycle_lock:
                if self._shutdown_requested:
                    raise RuntimeError("classbot is shutting down")

                start_task = self._start_task
                if start_task is not None and not start_task.done():
                    wait_task = start_task
                    recheck_after_wait = True
                else:
                    stop_task = self._stop_task
                    if stop_task is not None and not stop_task.done():
                        wait_task = stop_task
                        recheck_after_wait = True
                    else:
                        runtime = self._current_runtime
                        if runtime is None:
                            start_task = asyncio.create_task(
                                self._start_owner(),
                                name="classbot-runtime-start",
                            )
                            self._start_task = start_task
                            wait_task = start_task
                            recheck_after_wait = False
                        elif runtime.cleaned:
                            self._current_runtime = None
                        elif runtime.cleanup_started:
                            cleanup_runtime = runtime
                        else:
                            return runtime

            if cleanup_runtime is not None:
                await asyncio.shield(
                    self._wait_for_runtime_cleanup(cleanup_runtime)
                )
                continue

            if wait_task is not None:
                result = await asyncio.shield(wait_task)
                if not recheck_after_wait:
                    return result
            # A completed existing start or stop changes the state inspected
            # above. Loop rather than returning a possibly cleaned runtime.

    async def _start_owner(self) -> Runtime:
        parts: dict[str, Any] = {}
        runtime: Runtime | None = None
        owner_task = asyncio.current_task()
        try:
            parts["capture"] = await self._make(
                self.dependencies.capture_factory
            )
            parts["segmenter"] = await self._make(
                self.dependencies.segmenter_factory
            )
            parts["recognizer"] = await self._make(
                self.dependencies.recognizer_factory,
                self.settings.asr_device,
            )
            self._starting_recognizer = parts["recognizer"]
            parts["detector"] = await self._make(
                self.dependencies.detector_factory,
                self.settings.subject,
            )
            parts["client"] = await self._make(
                self.dependencies.client_factory,
                self.settings,
            )
            parts["pipeline"] = await self._make(
                self.dependencies.pipeline_factory,
                parts["recognizer"],
                parts["detector"],
                parts["client"],
                self.window.upsert_card,
                self.window.update_answer,
                self.window.set_status,
                self.window.set_transcript,
            )

            runtime = Runtime(
                capture=parts["capture"],
                segmenter=parts["segmenter"],
                recognizer=parts["recognizer"],
                detector=parts["detector"],
                client=parts["client"],
                pipeline=parts["pipeline"],
            )

            # Model loading happens before the microphone is opened.
            await self._maybe_await(runtime.pipeline.prepare())
            await self._maybe_await(runtime.capture.start())
            runtime.audio_task = asyncio.create_task(
                self._audio_loop(runtime),
                name="classbot-audio-loop",
            )
            runtime.pipeline_task = asyncio.create_task(
                self._pipeline_loop(runtime),
                name="classbot-pipeline-loop",
            )

            # Do not expose a partially prepared runtime. A concurrent stop
            # waits on _start_task until this complete runtime is available.
            async with self._lifecycle_lock:
                self._current_runtime = runtime
            return runtime
        except BaseException as error:
            status = (
                "mic_error"
                if isinstance(error, AudioCaptureError)
                else "asr_error"
            )
            if runtime is not None:
                await asyncio.shield(
                    self._cleanup_runtime(runtime, status=status)
                )
            else:
                await asyncio.shield(
                    self._cleanup_partial(parts, status=status)
                )
            raise
        finally:
            self._starting_recognizer = None
            async with self._lifecycle_lock:
                if self._start_task is owner_task:
                    self._start_task = None

    async def stop(self) -> None:
        """Stop the active runtime and release every owned resource."""
        async with self._lifecycle_lock:
            task = self._stop_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._stop_owner(),
                    name="classbot-runtime-stop",
                )
                self._stop_task = task

        try:
            await asyncio.shield(task)
        finally:
            if task.done():
                async with self._lifecycle_lock:
                    if self._stop_task is task:
                        self._stop_task = None

    async def _stop_owner(self) -> None:
        owner_task = asyncio.current_task()
        try:
            start_task = self._start_task
            if start_task is not None and start_task is not owner_task:
                await self._cancel_pending_recognizer_start(start_task)
                try:
                    await asyncio.shield(start_task)
                except BaseException as error:
                    _LOGGER.debug("start failed while stopping", exc_info=error)

            runtime = self._current_runtime
            if runtime is None:
                await self._set_status("stopped")
                return

            await self._cleanup_runtime(runtime, status="stopped")
        finally:
            async with self._lifecycle_lock:
                if self._stop_task is owner_task:
                    self._stop_task = None

    async def shutdown(self) -> None:
        """Wait for lifecycle work, clean the runtime, and quit Qt once."""
        async with self._lifecycle_lock:
            if self._shutdown_complete:
                return

            task = self._shutdown_task
            if task is None or task.done():
                self._shutdown_requested = True
                task = asyncio.create_task(
                    self._shutdown_owner(),
                    name="classbot-runtime-shutdown",
                )
                self._shutdown_task = task

        await asyncio.shield(task)

    async def _shutdown_owner(self) -> None:
        try:
            start_task = self._start_task
            if (
                start_task is not None
                and start_task is not asyncio.current_task()
            ):
                await self._cancel_pending_recognizer_start(start_task)
                try:
                    await asyncio.shield(start_task)
                except BaseException as error:
                    _LOGGER.debug(
                        "start failed while shutting down",
                        exc_info=error,
                    )
            await self.stop()
        finally:
            async with self._lifecycle_lock:
                self._shutdown_complete = True
            await self._quit()

    def connect_window_signals(self) -> None:
        """Connect Qt signals to tasks without starting work at import time."""
        if self._signals_connected:
            return
        self.window.start_requested.connect(self._schedule_start)
        self.window.stop_requested.connect(self._schedule_stop)
        self.window.close_requested.connect(self._schedule_shutdown)
        self._signals_connected = True

    def connect_application_signals(self, application: Any | None = None) -> None:
        """Schedule runtime cleanup when Qt begins shutting down externally."""
        if self._application_signals_connected:
            return
        if application is None:
            application = QApplication.instance()
        if application is None:
            return
        about_to_quit = getattr(application, "aboutToQuit", None)
        if about_to_quit is None:
            return
        about_to_quit.connect(self._handle_about_to_quit)
        self._application = application
        self._application_signals_connected = True

    def connect_signals(self, application: Any | None = None) -> None:
        self.connect_window_signals()
        self.connect_application_signals(application)

    def _handle_about_to_quit(self) -> None:
        self._application_quitting = True
        if not self._shutdown_complete:
            task = self._schedule(self.shutdown())
            self._drain_shutdown_task(task)

    def _drain_shutdown_task(self, task: asyncio.Task[Any]) -> None:
        application = self._application
        process_events = getattr(application, "processEvents", None)
        if process_events is None:
            return
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if (
            task.get_loop() is not running_loop
            or type(running_loop).__module__ != "qasync"
        ):
            return
        # qasync exits Qt immediately after aboutToQuit; let its queued task
        # steps finish while Qt still owns the event loop.
        while not task.done():
            process_events(QEventLoop.ProcessEventsFlag.AllEvents, 10)

    def _schedule_start(self) -> None:
        self._schedule(self.start())

    def _schedule_stop(self) -> None:
        self._schedule(self.stop())

    def _schedule_shutdown(self) -> None:
        self._schedule(self.shutdown())

    def _schedule(self, awaitable: Awaitable[Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(awaitable)
        self._signal_tasks.add(task)
        task.add_done_callback(self._forget_signal_task)
        return task

    def _forget_signal_task(self, task: asyncio.Task[Any]) -> None:
        self._signal_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            _LOGGER.error("classbot UI operation failed", exc_info=error)

    async def _audio_loop(self, runtime: Runtime) -> None:
        in_speech = False
        pre_roll: deque[bytes] = deque(maxlen=_PRE_ROLL_FRAMES)
        try:
            async for frame in runtime.capture.frames():
                pre_roll.append(frame)
                event = runtime.segmenter.accept(frame)
                if event.speech_started:
                    in_speech = True
                    runtime.detector.begin_segment()
                    for buffered_frame in tuple(pre_roll)[:-1]:
                        runtime.recognizer.push_audio(buffered_frame)
                if in_speech:
                    runtime.recognizer.push_audio(frame)
                if event.speech_ended:
                    runtime.recognizer.finish_segment()
                    in_speech = False
        except asyncio.CancelledError:
            raise
        except AudioCaptureError:
            _LOGGER.exception("microphone runtime failed")
            if not runtime.cleanup_started:
                await asyncio.shield(
                    self._cleanup_runtime(runtime, status="mic_error")
                )
        except Exception:
            _LOGGER.exception("audio runtime failed")
            if not runtime.cleanup_started:
                await asyncio.shield(
                    self._cleanup_runtime(runtime, status="asr_error")
                )

    async def _pipeline_loop(self, runtime: Runtime) -> None:
        error: BaseException | None = None
        cancelled = False
        try:
            await runtime.pipeline.run()
        except asyncio.CancelledError:
            cancelled = True
            raise
        except BaseException as caught:
            error = caught
            _LOGGER.exception("pipeline runtime failed")
        finally:
            if not runtime.cleanup_started:
                status = "asr_error" if error is not None else "stopped"
                if not cancelled:
                    await asyncio.shield(
                        self._cleanup_runtime(runtime, status=status)
                    )

    async def _cleanup_runtime(self, runtime: Runtime, *, status: str) -> None:
        current_task = asyncio.current_task()
        async with runtime.cleanup_lock:
            if runtime.cleanup_started:
                already_started = True
            else:
                already_started = False
                runtime.cleanup_started = True

                # Stopping capture first wakes ``frames()`` before its task is
                # awaited or cancelled.
                await self._safe_call(runtime.capture, "stop")
                await self._cancel_task(runtime.audio_task, current_task)

                pipeline_error = await self._safe_call(
                    runtime.pipeline,
                    "stop",
                )
                if pipeline_error is not None:
                    await self._safe_call(runtime.recognizer, "stop")

                await self._close_client(runtime.client)
                async with self._lifecycle_lock:
                    if self._current_runtime is runtime:
                        self._current_runtime = None
                runtime.cleaned = True
                await self._set_status(status)

        if not already_started:
            await self._settle_pipeline_task(runtime.pipeline_task, current_task)

    async def _cleanup_partial(
        self,
        parts: dict[str, Any],
        *,
        status: str,
    ) -> None:
        await self._safe_call(parts.get("capture"), "stop")
        pipeline = parts.get("pipeline")
        pipeline_error = await self._safe_call(pipeline, "stop")
        if pipeline is None or pipeline_error is not None:
            await self._safe_call(parts.get("recognizer"), "stop")
        await self._close_client(parts.get("client"))
        await self._set_status(status)

    async def _cancel_pending_recognizer_start(
        self,
        start_task: asyncio.Task[Any],
    ) -> None:
        if start_task.done():
            return

        recognizer = self._starting_recognizer
        if recognizer is None:
            return

        recognizer_start = getattr(recognizer, "_start_task", None)
        model_load = getattr(recognizer, "_model_load_task", None)
        start_pending = (
            recognizer_start is not None and not recognizer_start.done()
        )
        load_pending = model_load is not None and not model_load.done()
        if not start_pending and not load_pending:
            return

        error = await self._safe_call(recognizer, "stop")
        if error is not None:
            _LOGGER.warning("cancel pending recognizer start failed: %s", error)

    async def _settle_pipeline_task(
        self,
        task: asyncio.Task[None] | None,
        current_task: asyncio.Task[Any] | None,
    ) -> None:
        if task is None or task is current_task:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _cancel_task(
        self,
        task: asyncio.Task[Any] | None,
        current_task: asyncio.Task[Any] | None,
    ) -> None:
        if task is None or task is current_task:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def _close_client(self, client: Any) -> None:
        if client is None:
            return
        method = getattr(client, "aclose", None)
        if method is None:
            method = getattr(client, "close", None)
        if method is not None:
            try:
                await self._safe_awaitable(method())
            except BaseException as error:  # noqa: BLE001 - cleanup continues
                _LOGGER.warning("client close failed: %s", error)

    @staticmethod
    async def _wait_for_runtime_cleanup(runtime: Runtime) -> None:
        async with runtime.cleanup_lock:
            return

    async def _set_status(self, status: str) -> None:
        await self._safe_awaitable(self.window.set_status(status))

    async def _quit(self) -> None:
        if self._application_quitting:
            return
        self._application_quitting = True
        if self.quit_callback is not None:
            await self._safe_awaitable(self.quit_callback())
            return
        application = self._application
        if application is None:
            application = QApplication.instance()
        if application is not None:
            application.quit()

    @staticmethod
    async def _make(factory: Factory, *args: Any) -> Any:
        return await RuntimeController._maybe_await(factory(*args))

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @classmethod
    async def _safe_call(
        cls,
        owner: Any,
        method_name: str,
    ) -> BaseException | None:
        if owner is None:
            return None
        method = getattr(owner, method_name, None)
        if method is None:
            return None
        try:
            await cls._safe_awaitable(method())
        except BaseException as error:  # noqa: BLE001 - cleanup must continue
            _LOGGER.warning(
                "cleanup call %s failed: %s",
                method_name,
                error,
            )
            return error
        return None

    @staticmethod
    async def _safe_awaitable(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value


def main() -> None:
    """Run the qasync desktop application."""
    settings = Settings.from_env()
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)
    application.setQuitOnLastWindowClosed(False)

    import qasync

    loop = qasync.QEventLoop(application)
    asyncio.set_event_loop(loop)
    window = MainWindow(settings.subject)
    controller = RuntimeController(settings, window)
    controller.connect_signals(application)
    window.show()

    with loop:
        loop.run_forever()
