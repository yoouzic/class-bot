import asyncio
import threading

import numpy as np
import pytest

from classbot.asr import funasr_streaming
from classbot.asr.funasr_streaming import FunAsrStreamingRecognizer


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = self.outputs.pop(0) if self.outputs else ""
        return [{"text": text}]


class PhaseModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = "final" if kwargs["is_final"] else ""
        return [{"text": text}]


class BlockingInferenceModel:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def generate(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=3)
        self.finished.set()
        return [{"text": "完成"}]


class RaisingModel:
    def generate(self, **kwargs):
        raise RuntimeError("fake inference failure")


def _pcm(samples: int, value: int = 1000) -> bytes:
    return (np.full(samples, value, dtype="<i2")).tobytes()


async def _next_event(events):
    return await asyncio.wait_for(events.__anext__(), timeout=1)


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1)


@pytest.mark.asyncio
async def test_partial_and_final_events_keep_segment_id_and_use_streaming_parameters(
    monkeypatch,
):
    model = FakeModel(["你好", "你好啊"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    partial = await _next_event(events)

    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert partial.text == "你好"
    assert partial.is_final is False
    assert final.text == "你好啊"
    assert final.is_final is True
    assert partial.segment_id == final.segment_id == 0
    assert model.calls[0]["is_final"] is False
    assert model.calls[1]["is_final"] is True
    assert model.calls[0]["chunk_size"] == [0, 10, 5]
    assert model.calls[0]["encoder_chunk_look_back"] == 4
    assert model.calls[0]["decoder_chunk_look_back"] == 1
    assert model.calls[0]["cache"] is model.calls[1]["cache"]
    assert model.calls[0]["input"].dtype == np.float32
    assert np.allclose(model.calls[0]["input"], 1000 / 32768)

    await recognizer.stop()


@pytest.mark.asyncio
async def test_streaming_chunks_accumulate_within_one_segment(monkeypatch):
    model = FakeModel(["马克思的", "生辰是什么", ""])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    first = await _next_event(events)
    recognizer.push_audio(_pcm(160))
    second = await _next_event(events)
    recognizer.finish_segment()
    final = await _next_event(events)

    assert first.text == "马克思的"
    assert second.text == "马克思的生辰是什么"
    assert final.text == "马克思的生辰是什么"
    assert first.segment_id == second.segment_id == final.segment_id == 0
    assert final.is_final is True

    await recognizer.stop()


@pytest.mark.asyncio
async def test_finish_increments_segment_id_and_new_segment_emits_new_event(monkeypatch):
    model = FakeModel(["第一段", "第二段"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    first = await _next_event(events)
    recognizer.finish_segment()
    first_final = await _next_event(events)

    recognizer.push_audio(_pcm(160, value=-2000))
    second = await _next_event(events)

    assert first.segment_id == first_final.segment_id == 0
    assert first_final.is_final is True
    assert model.calls[1]["is_final"] is False
    assert model.calls[1]["input"].size == 160
    assert second.text == "第二段"
    assert second.segment_id == 1
    assert second.segment_id != first.segment_id

    await recognizer.stop()


@pytest.mark.asyncio
async def test_empty_final_inference_replays_last_partial_text(monkeypatch):
    model = FakeModel(["保留文本", ""])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    partial = await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert partial.text == final.text == "保留文本"
    assert partial.is_final is False
    assert final.is_final is True
    assert final.segment_id == 0

    await recognizer.stop()


@pytest.mark.asyncio
async def test_long_segment_is_refined_with_full_final_decode(monkeypatch):
    class RefiningModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["is_final"] and kwargs["input"].size > 160:
                return [{"text": "完整最终文本"}]
            if kwargs["is_final"]:
                return [{"text": ""}]
            return [{"text": "实时片段"}]

    model = RefiningModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        final_refinement=True,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(320))
    await _next_event(events)
    await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert final.text == "完整最终文本"
    assert final.is_final is True
    assert model.calls[-1]["is_final"] is True
    assert model.calls[-1]["input"].size == 400
    assert model.calls[-1]["cache"] is not model.calls[0]["cache"]

    await recognizer.stop()


@pytest.mark.asyncio
async def test_final_refinement_passes_course_hotwords_and_corrections(monkeypatch):
    class CourseAwareModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["is_final"] and kwargs["input"].size > 160:
                return [{"text": "主观违心主义"}]
            return [{"text": "实时片段" if not kwargs["is_final"] else ""}]

    model = CourseAwareModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        hotwords=("主观唯心主义",),
        corrections={"违心": "唯心"},
        final_refinement=True,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(320))
    await _next_event(events)
    await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    await _next_event(events)

    final_call = model.calls[-1]
    assert final_call["hotword"] == "主观唯心主义"
    assert final_call["postprocess_hotwords"] == {"违心": "唯心"}
    assert final_call["postprocess_hotword_fuzzy"] is False

    await recognizer.stop()


@pytest.mark.asyncio
async def test_streaming_output_applies_course_corrections(monkeypatch):
    model = FakeModel(["可助论"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        corrections={"可助论": "唯物论"},
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    event = await _next_event(events)

    assert event.text == "唯物论"
    await recognizer.stop()


@pytest.mark.asyncio
async def test_streaming_output_corrects_terms_split_across_chunks(monkeypatch):
    model = FakeModel(["随心一", "意"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        corrections={"随心一意": "唯心主义"},
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    await _next_event(events)
    recognizer.push_audio(_pcm(160))
    second = await _next_event(events)

    assert second.text == "唯心主义"
    await recognizer.stop()


@pytest.mark.asyncio
async def test_live_mode_skips_blocking_full_segment_refinement(monkeypatch):
    class RefiningModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["is_final"] and kwargs["input"].size > 160:
                return [{"text": "完整最终文本"}]
            if kwargs["is_final"]:
                return [{"text": ""}]
            return [{"text": "实时片段"}]

    model = RefiningModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(320))
    await _next_event(events)
    await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert final.text == "实时片段"
    assert final.is_final is True
    assert all(call["input"].size <= 160 for call in model.calls)

    await recognizer.stop()


@pytest.mark.asyncio
async def test_audio_after_finish_marker_is_processed_as_next_segment(monkeypatch):
    model = FakeModel(["第一段", "第二段"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    first = await _next_event(events)
    recognizer.finish_segment()
    recognizer.push_audio(_pcm(160, value=-2000))
    recognizer.finish_segment()

    first_final = await _next_event(events)
    second = await _next_event(events)
    second_final = await _next_event(events)

    assert first.segment_id == first_final.segment_id == 0
    assert first_final.is_final is True
    assert second.segment_id == 1
    assert second.text == "第二段"
    assert second_final.segment_id == 1
    assert second_final.text == "第二段"
    assert second_final.is_final is True
    await recognizer.stop()


@pytest.mark.asyncio
async def test_stop_reliably_ends_events(monkeypatch):
    model = FakeModel([])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    await recognizer.stop()

    with pytest.raises(StopAsyncIteration):
        await _next_event(events)


@pytest.mark.asyncio
async def test_stop_during_model_load_invalidates_old_start_and_allows_restart(
    monkeypatch,
):
    load_started = threading.Event()
    release_load = threading.Event()
    models = []

    def build(device):
        load_started.set()
        release_load.wait(timeout=3)
        model = FakeModel(["重新开始"])
        models.append(model)
        return model

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    start_task = asyncio.create_task(recognizer.start())
    await asyncio.wait_for(asyncio.to_thread(load_started.wait, 1), timeout=1)

    stop_task = asyncio.create_task(recognizer.stop())
    await asyncio.sleep(0)
    release_load.set()
    await asyncio.wait_for(stop_task, timeout=2)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=2)

    assert recognizer._worker is None
    assert recognizer._stopped is True

    await recognizer.start()
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))
    event = await _next_event(events)
    assert event.text == "重新开始"
    assert event.segment_id == 0
    await recognizer.stop()
    assert len(models) == 2


@pytest.mark.asyncio
async def test_stop_cancels_pending_model_load_without_waiting_for_build_thread(
    monkeypatch,
):
    load_started = threading.Event()
    release_load = threading.Event()

    def build(device):
        load_started.set()
        release_load.wait()
        return FakeModel([])

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    start_task = asyncio.create_task(recognizer.start())
    await asyncio.wait_for(asyncio.to_thread(load_started.wait, 1), timeout=1)

    stop_task = asyncio.create_task(recognizer.stop())
    stopped_without_building = True
    try:
        await asyncio.wait_for(asyncio.shield(stop_task), timeout=0.2)
    except TimeoutError:
        stopped_without_building = False
    finally:
        release_load.set()

    await asyncio.wait_for(stop_task, timeout=2)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start_task, timeout=2)

    assert stopped_without_building is True
    assert recognizer._worker is None
    assert recognizer._start_task is None


@pytest.mark.asyncio
async def test_cancelled_start_then_immediate_restart_does_not_return_empty_worker(
    monkeypatch,
):
    load_started = threading.Event()
    release_load = threading.Event()
    calls = 0

    def build(device):
        nonlocal calls
        calls += 1
        if calls == 1:
            load_started.set()
            release_load.wait(timeout=3)
        return FakeModel(["新会话"])

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    old_start = asyncio.create_task(recognizer.start())
    await asyncio.wait_for(asyncio.to_thread(load_started.wait, 1), timeout=1)

    old_start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await old_start

    new_start = asyncio.create_task(recognizer.start())
    await asyncio.sleep(0)
    release_load.set()
    await asyncio.wait_for(new_start, timeout=2)

    assert calls == 2
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))
    assert (await _next_event(events)).text == "新会话"
    await recognizer.stop()


@pytest.mark.asyncio
async def test_stop_cleans_session_state_and_is_idempotent(monkeypatch):
    models = []

    def build(device):
        model = FakeModel([])
        models.append(model)
        return model

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)

    await recognizer.start()
    recognizer.push_audio(_pcm(80))
    await _wait_until(lambda: recognizer._residual.size == 80)
    old_cache = recognizer._cache
    recognizer._last_text = "旧文本"
    recognizer._worker_error = RuntimeError("old error")
    events = recognizer.events()
    await recognizer.stop()
    with pytest.raises(RuntimeError, match="old error"):
        await _next_event(events)
    second_events = recognizer.events()
    await recognizer.stop()
    with pytest.raises(StopAsyncIteration):
        await _next_event(second_events)

    assert recognizer._residual.size == 0
    assert recognizer._cache == {}
    assert recognizer._cache is not old_cache
    assert recognizer._last_text == ""
    assert recognizer._segment_samples == 0
    assert recognizer._worker is None
    assert recognizer._worker_error is None
    assert recognizer._input_queue.empty()
    assert recognizer._events_queue.empty()

    await recognizer.start()
    assert recognizer._segment_id == 0
    await recognizer.stop()
    assert len(models) == 2


@pytest.mark.asyncio
async def test_worker_failure_stops_accepting_audio_and_can_restart(monkeypatch):
    models = [RaisingModel(), FakeModel(["恢复"])]

    def build(device):
        return models.pop(0)

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    await recognizer.start()
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))

    with pytest.raises(RuntimeError, match="fake inference failure"):
        await _next_event(events)

    assert recognizer._stopped is True
    assert recognizer._worker is None
    with pytest.raises(RuntimeError, match="not running"):
        recognizer.push_audio(_pcm(160))

    await recognizer.start()
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))
    assert (await _next_event(events)).text == "恢复"
    await recognizer.stop()


@pytest.mark.asyncio
async def test_stop_waits_for_inference_thread_before_session_cleanup(monkeypatch):
    model = BlockingInferenceModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    await asyncio.wait_for(asyncio.to_thread(model.started.wait, 1), timeout=1)

    stop_task = asyncio.create_task(recognizer.stop())
    await asyncio.sleep(0.05)
    stop_finished_early = stop_task.done()

    model.release.set()
    await asyncio.wait_for(stop_task, timeout=2)
    assert stop_finished_early is False
    assert model.finished.is_set()
    assert recognizer._inference_tasks == set()


@pytest.mark.asyncio
async def test_live_queue_buffers_a_short_inference_burst(monkeypatch):
    model = BlockingInferenceModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    await asyncio.wait_for(asyncio.to_thread(model.started.wait, 1), timeout=1)

    for _ in range(400):
        recognizer.push_audio(_pcm(160))

    assert recognizer._input_queue.qsize() >= 400

    model.release.set()
    await recognizer.stop()


@pytest.mark.asyncio
async def test_finish_marker_survives_full_queue_in_live_mode(monkeypatch):
    model = FakeModel(["第一段"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    for _ in range(32):
        recognizer.push_audio(_pcm(160))
    recognizer.finish_segment()
    for _ in range(32):
        recognizer.push_audio(_pcm(160, value=-1000))

    final = await _next_event(events)
    while not final.is_final:
        final = await _next_event(events)
    await _wait_until(lambda: recognizer._segment_id == 1)

    assert final.text == "第一段"
    assert final.is_final is True
    assert final.segment_id == 0
    assert len([call for call in model.calls if call["is_final"]]) == 0
    assert recognizer._closing is False
    await recognizer.stop()


@pytest.mark.asyncio
async def test_full_audio_queue_rejects_overload_without_evicting_finish_boundary(monkeypatch):
    model = FakeModel(["不能被误标为完整"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    await recognizer.start()
    events = recognizer.events()
    try:
        for number in range(funasr_streaming._AUDIO_QUEUE_SIZE):
            recognizer.push_audio(_pcm(160, value=number))
        recognizer.finish_segment()
        accepted = tuple(recognizer._input_queue._queue)
        assert len(accepted) == funasr_streaming._INPUT_QUEUE_SIZE
        assert accepted[-1] is funasr_streaming._FINISH

        with pytest.raises(RuntimeError, match="音频.*积压") as caught:
            recognizer.push_audio(_pcm(160, value=-1000))

        assert type(caught.value).__name__ == "AsrOverloadError"
        assert tuple(recognizer._input_queue._queue) == accepted
        assert recognizer._stopped is True
        with pytest.raises(RuntimeError, match="音频.*积压") as streamed:
            await _next_event(events)
        assert streamed.value is caught.value
        with pytest.raises(RuntimeError, match="not running"):
            recognizer.push_audio(_pcm(160))
        with pytest.raises(RuntimeError, match="not running"):
            recognizer.finish_segment()
        assert model.calls == []
    finally:
        await recognizer.stop()


@pytest.mark.asyncio
async def test_finish_marker_queue_is_bounded_and_never_evicts_accepted_audio(monkeypatch):
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: FakeModel([]))
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    await recognizer.start()
    try:
        for _ in range(funasr_streaming._AUDIO_QUEUE_SIZE):
            recognizer.push_audio(_pcm(160))
        recognizer.finish_segment()
        accepted = tuple(recognizer._input_queue._queue)

        with pytest.raises(RuntimeError, match="音频.*积压") as caught:
            recognizer.finish_segment()

        assert type(caught.value).__name__ == "AsrOverloadError"
        assert recognizer._input_queue.maxsize == funasr_streaming._INPUT_QUEUE_SIZE
        assert tuple(recognizer._input_queue._queue) == accepted
    finally:
        await recognizer.stop()


@pytest.mark.asyncio
async def test_overload_suppresses_inflight_inference_and_final_transcript(monkeypatch):
    model = BlockingInferenceModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    await recognizer.start()
    events = recognizer.events()
    try:
        recognizer.push_audio(_pcm(160))
        await asyncio.wait_for(asyncio.to_thread(model.started.wait, 1), timeout=1)
        for _ in range(funasr_streaming._AUDIO_QUEUE_SIZE):
            recognizer.push_audio(_pcm(160))
        recognizer.finish_segment()

        with pytest.raises(RuntimeError, match="音频.*积压"):
            recognizer.push_audio(_pcm(160))

        model.release.set()
        await _wait_until(lambda: not recognizer._inference_tasks)
        with pytest.raises(RuntimeError, match="音频.*积压"):
            await _next_event(events)
        with pytest.raises(StopAsyncIteration):
            await _next_event(events)
        assert recognizer._last_text == ""
        assert recognizer._segment_id == 0
    finally:
        model.release.set()
        await recognizer.stop()


@pytest.mark.asyncio
async def test_audio_overload_can_restart_with_clean_segment_and_no_old_frames(monkeypatch):
    old_model = FakeModel(["旧会话不能继续"])
    new_model = FakeModel(["恢复后的新片段"])
    models = iter((old_model, new_model))
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: next(models))
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    await recognizer.start()
    old_events = recognizer.events()
    try:
        for _ in range(funasr_streaming._AUDIO_QUEUE_SIZE):
            recognizer.push_audio(_pcm(160))
        with pytest.raises(RuntimeError, match="音频.*积压"):
            recognizer.push_audio(_pcm(160))
        with pytest.raises(RuntimeError, match="音频.*积压"):
            await _next_event(old_events)

        await recognizer.start()
        recognizer.push_audio(_pcm(160, value=-2000))
        recognizer.finish_segment()
        events = recognizer.events()
        partial = await _next_event(events)
        final = await _next_event(events)

        assert partial.text == final.text == "恢复后的新片段"
        assert partial.segment_id == final.segment_id == 0
        assert final.is_final is True
        assert old_model.calls == []
        assert len(new_model.calls) == 1
        assert np.allclose(new_model.calls[0]["input"], -2000 / 32768)
    finally:
        await recognizer.stop()


@pytest.mark.asyncio
async def test_live_recognizer_does_not_retain_full_segment_audio(monkeypatch):
    model = FakeModel([])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160, final_refinement=False)
    await recognizer.start()
    try:
        recognizer.push_audio(_pcm(160 * 40))
        await _wait_until(lambda: recognizer._segment_samples == 160 * 40)

        assert recognizer._segment_audio == []
        assert recognizer._residual.size < recognizer.chunk_samples

        recognizer.finish_segment()
        await _wait_until(lambda: recognizer._segment_id == 1)
        assert recognizer._segment_audio == []
    finally:
        await recognizer.stop()


@pytest.mark.asyncio
async def test_continuous_speech_bounds_retained_refinement_audio(monkeypatch):
    from classbot.audio.segmenter import SpeechSegmenter

    peak_retained_samples = []

    class MeasuringModel:
        def generate(self, **kwargs):
            peak_retained_samples.append(sum(audio.size for audio in recognizer._segment_audio))
            return []

    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: MeasuringModel())
    recognizer = FunAsrStreamingRecognizer(chunk_samples=320, final_refinement=True)
    segmenter = SpeechSegmenter(max_segment_ms=100)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: True)
    frame = _pcm(320)
    await recognizer.start()
    try:
        for _ in range(30):
            event = segmenter.accept(frame)
            recognizer.push_audio(frame)
            if event.speech_ended:
                recognizer.finish_segment()
        await _wait_until(lambda: recognizer._segment_id == 6)

        assert peak_retained_samples
        assert max(peak_retained_samples) <= 16000 * 100 // 1000
        assert recognizer._segment_audio == []
        assert recognizer._input_queue.empty()
    finally:
        await recognizer.stop()


class FakeRefiner:
    def __init__(self, *, loaded=False, text="", error=None):
        self._loaded = loaded
        self.text = text
        self.error = error
        self.refine_calls = []
        self.load_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self._loaded = True

    def refine(self, samples):
        self.refine_calls.append(samples)
        if self.error is not None:
            raise self.error
        return self.text


class NeverLoadedRefiner:
    loaded = False

    def __init__(self):
        self.refine_calls = []
        self.load_calls = 0

    def load(self):
        self.load_calls += 1

    def refine(self, samples):
        self.refine_calls.append(samples)
        raise AssertionError("offline refine must not run while unloaded")


class _StreamingRefinementModel:
    """Streams partials, then offers a full-segment decode on final."""

    def __init__(self, refined_text="完整最终文本"):
        self.calls = []
        self.refined_text = refined_text

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["is_final"] and kwargs["input"].size > 160:
            return [{"text": self.refined_text}]
        if kwargs["is_final"]:
            return [{"text": ""}]
        return [{"text": "实时片段"}]


@pytest.mark.asyncio
async def test_loaded_offline_refiner_replaces_streaming_refinement(monkeypatch):
    model = _StreamingRefinementModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    refiner = FakeRefiner(loaded=True, text="主观违心主义，带标点。")
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        corrections={"违心": "唯心"},
        final_refinement=True,
        refiner=refiner,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(320))
    await _next_event(events)
    await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert final.text == "主观唯心主义，带标点。"
    assert final.is_final is True
    assert len(refiner.refine_calls) == 1
    assert refiner.refine_calls[0].size == 400
    # The offline refiner answered, so no full-segment streaming re-decode ran.
    assert all(call["input"].size <= 160 for call in model.calls)

    await recognizer.stop()


@pytest.mark.asyncio
async def test_unloaded_refiner_falls_back_to_streaming_refinement(monkeypatch):
    model = _StreamingRefinementModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    refiner = NeverLoadedRefiner()
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        final_refinement=True,
        refiner=refiner,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(320))
    await _next_event(events)
    await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert final.text == "完整最终文本"
    assert refiner.refine_calls == []

    await recognizer.stop()


@pytest.mark.asyncio
async def test_failing_offline_refinement_falls_back_to_streaming(monkeypatch):
    model = _StreamingRefinementModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    refiner = FakeRefiner(loaded=True, text="被忽略", error=RuntimeError("offline broken"))
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        final_refinement=True,
        refiner=refiner,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(320))
    await _next_event(events)
    await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert final.text == "完整最终文本"
    assert len(refiner.refine_calls) == 1

    await recognizer.stop()


@pytest.mark.asyncio
async def test_session_start_prewarms_refiner_load(monkeypatch):
    model = FakeModel(["你好"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    refiner = FakeRefiner(loaded=False, text="")
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        final_refinement=True,
        refiner=refiner,
    )

    await recognizer.start()
    try:
        await _wait_until(lambda: refiner.load_calls == 1)
    finally:
        await recognizer.stop()


@pytest.mark.asyncio
async def test_refiner_is_not_prewarmed_without_final_refinement(monkeypatch):
    model = FakeModel(["你好"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    refiner = FakeRefiner(loaded=False, text="")
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        final_refinement=False,
        refiner=refiner,
    )

    await recognizer.start()
    try:
        assert refiner.load_calls == 0
        assert recognizer._refiner_load_task is None
    finally:
        await recognizer.stop()


@pytest.mark.asyncio
async def test_stop_cancels_pending_refiner_load(monkeypatch):
    model = FakeModel(["你好"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)

    class SlowLoadingRefiner:
        def __init__(self):
            self._loaded = False
            self.started = threading.Event()
            self.release = threading.Event()

        @property
        def loaded(self):
            return self._loaded

        def load(self):
            self.started.set()
            self.release.wait(timeout=3)
            self._loaded = True

    refiner = SlowLoadingRefiner()
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        final_refinement=True,
        refiner=refiner,
    )

    await recognizer.start()
    try:
        await _wait_until(lambda: refiner.started.is_set())
        await asyncio.wait_for(recognizer.stop(), timeout=2)
        assert recognizer._refiner_load_task is None
    finally:
        refiner.release.set()
        await recognizer.stop()


class FakePunctuator:
    def __init__(self, *, loaded=False, suffix="，", error=None):
        self._loaded = loaded
        self.suffix = suffix
        self.error = error
        self.punctuate_calls = []
        self.load_calls = 0

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        self.load_calls += 1
        self._loaded = True

    def punctuate(self, text):
        self.punctuate_calls.append(text)
        if self.error is not None:
            raise self.error
        return text + self.suffix


class NeverLoadingPunctuator:
    loaded = False

    def __init__(self):
        self.punctuate_calls = []
        self.load_calls = 0

    def load(self):
        self.load_calls += 1

    def punctuate(self, text):
        self.punctuate_calls.append(text)
        raise AssertionError("partial punctuation must not run while unloaded")


@pytest.mark.asyncio
async def test_loaded_punctuator_marks_partial_events_only(monkeypatch):
    model = FakeModel(["你好", "你好啊"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    punctuator = FakePunctuator(loaded=True, suffix="，")
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        partial_punctuator=punctuator,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    partial = await _next_event(events)
    # internal accumulation stays raw; only the emitted partial is punctuated
    assert recognizer._segment_text == "你好"
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert partial.text == "你好，"
    assert partial.is_final is False
    assert final.text == "你好啊"
    assert final.is_final is True
    assert punctuator.punctuate_calls == ["你好"]

    await recognizer.stop()


@pytest.mark.asyncio
async def test_unloaded_punctuator_keeps_raw_partials(monkeypatch):
    model = FakeModel(["你好"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    punctuator = NeverLoadingPunctuator()
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        partial_punctuator=punctuator,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    partial = await _next_event(events)

    assert partial.text == "你好"
    assert punctuator.punctuate_calls == []

    await recognizer.stop()


@pytest.mark.asyncio
async def test_failing_punctuation_degrades_to_raw_partial(monkeypatch):
    model = FakeModel(["你好"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    punctuator = FakePunctuator(loaded=True, error=RuntimeError("punc broken"))
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        partial_punctuator=punctuator,
    )
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    partial = await _next_event(events)

    assert partial.text == "你好"
    assert len(punctuator.punctuate_calls) == 1

    await recognizer.stop()


@pytest.mark.asyncio
async def test_punctuator_load_starts_even_without_final_refinement(monkeypatch):
    model = FakeModel(["你好"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    punctuator = FakePunctuator(loaded=False)
    recognizer = FunAsrStreamingRecognizer(
        chunk_samples=160,
        final_refinement=False,
        partial_punctuator=punctuator,
    )

    await recognizer.start()
    try:
        await _wait_until(lambda: punctuator.load_calls == 1)
    finally:
        await recognizer.stop()
