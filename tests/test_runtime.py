import asyncio
from collections.abc import Callable

import pytest

import classbot.main as main_module
from classbot.audio.segmenter import SegmentEvent
from classbot.config import Settings


class FakeSignal:
    def __init__(self):
        self.callbacks: list[Callable[..., object]] = []

    def connect(self, callback):
        self.callbacks.append(callback)

    def emit(self):
        for callback in tuple(self.callbacks):
            callback()


class FakeWindow:
    def __init__(self):
        self.start_requested = FakeSignal()
        self.stop_requested = FakeSignal()
        self.close_requested = FakeSignal()
        self.statuses: list[str] = []
        self.cards = []
        self.answers = []

    def upsert_card(self, card):
        self.cards.append(card)

    def update_answer(self, question_id, answer):
        self.answers.append((question_id, answer))

    def set_status(self, status):
        self.statuses.append(status)


_STOP = object()


class FakeCapture:
    def __init__(self, order, *, start_error=None):
        self.order = order
        self.start_error = start_error
        self.started = False
        self.stop_calls = 0
        self._queue = asyncio.Queue()

    def start(self):
        self.order.append("capture.start")
        if self.start_error is not None:
            raise self.start_error
        self.started = True

    def stop(self):
        self.stop_calls += 1
        if self.stop_calls == 1:
            self._queue.put_nowait(_STOP)

    def frames(self):
        async def stream():
            while True:
                frame = await self._queue.get()
                if frame is _STOP:
                    return
                yield frame

        return stream()

    def feed(self, frame):
        self._queue.put_nowait(frame)


class FakeSegmenter:
    def __init__(self, events=None):
        self.events = list(events or [])

    def accept(self, _frame):
        if self.events:
            return self.events.pop(0)
        return SegmentEvent()


class FakeRecognizer:
    def __init__(self):
        self.start_calls = 0
        self.stop_calls = 0
        self.push_calls = []
        self.finish_calls = 0
        self._stopped = asyncio.Event()

    async def start(self):
        self.start_calls += 1

    def push_audio(self, frame):
        self.push_calls.append(frame)

    def finish_segment(self):
        self.finish_calls += 1

    def events(self):
        async def stream():
            await self._stopped.wait()

        return stream()

    async def stop(self):
        self.stop_calls += 1
        self._stopped.set()


class FakeDetector:
    def __init__(self):
        self.begin_calls = 0

    def begin_segment(self):
        self.begin_calls += 1


class FakeClient:
    def __init__(self):
        self.close_calls = 0

    async def aclose(self):
        self.close_calls += 1


class FakePipeline:
    def __init__(
        self,
        order,
        *,
        prepare_error=None,
        run_returns=False,
    ):
        self.order = order
        self.prepare_error = prepare_error
        self.run_returns = run_returns
        self.prepare_calls = 0
        self.run_calls = 0
        self.stop_calls = 0
        self.on_card = None
        self.on_delta = None
        self.on_status = None
        self._stop_event = asyncio.Event()

    async def prepare(self):
        self.prepare_calls += 1
        self.order.append("pipeline.prepare")
        if self.prepare_error is not None:
            raise self.prepare_error

    async def run(self):
        self.run_calls += 1
        if self.run_returns:
            return
        await self._stop_event.wait()

    async def stop(self):
        self.stop_calls += 1
        self._stop_event.set()


class RuntimeFactory:
    def __init__(
        self,
        *,
        prepare_error=None,
        capture_start_error=None,
        run_returns=False,
        segment_events=None,
    ):
        self.prepare_error = prepare_error
        self.capture_start_error = capture_start_error
        self.run_returns = run_returns
        self.segment_events = segment_events
        self.sessions = []
        self._current = None

    def _session(self):
        if self._current is None:
            self._current = {"order": []}
        return self._current

    def capture(self):
        session = self._session()
        capture = FakeCapture(
            session["order"],
            start_error=self.capture_start_error,
        )
        session["capture"] = capture
        return capture

    def segmenter(self):
        session = self._session()
        segmenter = FakeSegmenter(self.segment_events)
        session["segmenter"] = segmenter
        return segmenter

    def recognizer(self, _device):
        session = self._session()
        recognizer = FakeRecognizer()
        session["recognizer"] = recognizer
        return recognizer

    def detector(self, _subject):
        session = self._session()
        detector = FakeDetector()
        session["detector"] = detector
        return detector

    def client(self, _settings):
        session = self._session()
        client = FakeClient()
        session["client"] = client
        return client

    def pipeline(
        self,
        _recognizer,
        _detector,
        _client,
        on_card,
        on_delta,
        on_status,
    ):
        session = self._session()
        pipeline = FakePipeline(
            session["order"],
            prepare_error=self.prepare_error,
            run_returns=self.run_returns,
        )
        pipeline.on_card = on_card
        pipeline.on_delta = on_delta
        pipeline.on_status = on_status
        session["pipeline"] = pipeline
        self.sessions.append(session)
        self._current = None
        return pipeline


def settings():
    return Settings(
        deepseek_api_key="test-key",
        subject="历史",
        model="deepseek-flash",
        base_url="https://api.deepseek.com",
        asr_device="cpu",
    )


def make_controller(factory, window=None, *, quit_callback=None):
    dependencies = main_module.RuntimeDependencies(
        capture_factory=factory.capture,
        segmenter_factory=factory.segmenter,
        recognizer_factory=factory.recognizer,
        detector_factory=factory.detector,
        client_factory=factory.client,
        pipeline_factory=factory.pipeline,
    )
    return main_module.RuntimeController(
        settings(),
        window or FakeWindow(),
        dependencies=dependencies,
        quit_callback=quit_callback,
    )


async def wait_until(predicate):
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


@pytest.mark.asyncio
async def test_start_prepares_pipeline_before_starting_capture_and_builds_runtime():
    factory = RuntimeFactory()
    window = FakeWindow()
    controller = make_controller(factory, window)

    runtime = await controller.start()

    assert runtime is controller.current_runtime
    assert runtime.capture is factory.sessions[0]["capture"]
    assert runtime.segmenter is factory.sessions[0]["segmenter"]
    assert runtime.recognizer is factory.sessions[0]["recognizer"]
    assert runtime.detector is factory.sessions[0]["detector"]
    assert runtime.client is factory.sessions[0]["client"]
    assert runtime.pipeline is factory.sessions[0]["pipeline"]
    assert runtime.audio_task is not None
    assert runtime.pipeline_task is not None
    assert factory.sessions[0]["order"] == [
        "pipeline.prepare",
        "capture.start",
    ]

    await controller.stop()


@pytest.mark.asyncio
async def test_stop_cleans_runtime_and_clears_current_reference():
    factory = RuntimeFactory()
    window = FakeWindow()
    controller = make_controller(factory, window)

    await controller.start()
    session = factory.sessions[0]
    await controller.stop()

    assert session["capture"].stop_calls >= 1
    assert session["pipeline"].stop_calls == 1
    assert session["client"].close_calls == 1
    assert controller.current_runtime is None
    assert window.statuses[-1] == "stopped"


@pytest.mark.asyncio
async def test_start_after_stop_builds_a_new_runtime_and_pipeline():
    factory = RuntimeFactory()
    controller = make_controller(factory)

    first = await controller.start()
    await controller.stop()
    second = await controller.start()
    await controller.stop()

    assert len(factory.sessions) == 2
    assert second is not first
    assert second.pipeline is not first.pipeline
    assert factory.sessions[0]["pipeline"] is not factory.sessions[1]["pipeline"]
    assert factory.sessions[0]["client"].close_calls == 1
    assert factory.sessions[1]["client"].close_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prepare_error", "capture_start_error"),
    [
        (RuntimeError("prepare failed"), None),
        (None, RuntimeError("capture failed")),
    ],
)
async def test_start_failure_cleans_components_and_reports_asr_error(
    prepare_error,
    capture_start_error,
):
    factory = RuntimeFactory(
        prepare_error=prepare_error,
        capture_start_error=capture_start_error,
    )
    window = FakeWindow()
    controller = make_controller(factory, window)

    with pytest.raises(RuntimeError):
        await controller.start()

    session = factory.sessions[0]
    assert session["capture"].stop_calls >= 1
    assert session["pipeline"].stop_calls == 1
    assert session["client"].close_calls == 1
    assert controller.current_runtime is None
    assert "asr_error" in window.statuses


@pytest.mark.asyncio
async def test_audio_loop_begins_segment_pushes_audio_and_finishes_segment():
    factory = RuntimeFactory(
        segment_events=[
            SegmentEvent(speech_started=True),
            SegmentEvent(speech_ended=True),
        ]
    )
    controller = make_controller(factory)

    runtime = await controller.start()
    capture = factory.sessions[0]["capture"]
    detector = factory.sessions[0]["detector"]
    recognizer = factory.sessions[0]["recognizer"]
    capture.feed(b"speech")
    capture.feed(b"silence")

    await wait_until(lambda: recognizer.finish_calls == 1)
    assert detector.begin_calls == 1
    assert recognizer.push_calls == [b"speech", b"silence"]

    await controller.stop()
    assert runtime.audio_task is not None
    assert runtime.audio_task.done()


@pytest.mark.asyncio
async def test_shutdown_quits_once_and_repeated_shutdown_is_idempotent():
    factory = RuntimeFactory()
    quit_calls = []
    controller = make_controller(
        factory,
        quit_callback=lambda: quit_calls.append(True),
    )

    await controller.start()
    await controller.shutdown()
    await controller.shutdown()

    assert quit_calls == [True]
    assert controller.current_runtime is None


@pytest.mark.asyncio
async def test_close_signal_is_connected_and_closes_runtime_client():
    factory = RuntimeFactory()
    window = FakeWindow()
    quit_calls = []
    controller = make_controller(
        factory,
        window,
        quit_callback=lambda: quit_calls.append(True),
    )
    controller.connect_window_signals()

    await controller.start()
    window.close_requested.emit()
    await wait_until(lambda: quit_calls == [True])
    await controller.shutdown()

    assert factory.sessions[0]["client"].close_calls == 1
    assert controller.current_runtime is None
