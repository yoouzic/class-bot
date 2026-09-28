# 课堂实时问答助手 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Windows 上实现一个带明显监听状态的课堂实时问答悬浮窗：本地流式中文 ASR 识别问题，使用 DeepSeek Flash 流式生成简短答案，并按问题分卡片显示。

**Architecture:** 音频采集、VAD、FunASR `paraformer-zh-streaming` 和问题检测在本地运行；检测到候选问题后，只把文本和学科上下文发送给 DeepSeek `deepseek-flash`，通过 SSE 增量更新答案卡片。PySide6 UI 通过 qasync 驱动 asyncio pipeline，所有耗时工作都不在 UI 线程执行。

**Tech Stack:** Python 3.11、PySide6、qasync、sounddevice、webrtcvad-wheels、FunASR/Paraformer streaming、httpx、DeepSeek Chat Completions SSE、pytest、pytest-asyncio、ruff。

---

## 文件结构

实现完成后，工程应具有以下边界清晰的文件：

- Create: `pyproject.toml` — Python 包、运行依赖、可选 ASR 依赖、测试和 lint 配置。
- Create: `.gitignore` — Python、模型缓存、密钥和运行日志忽略规则。
- Create: `.env.example` — 仅列出配置键名，不包含真实密钥。
- Create: `README.md` — Windows 安装、模型首次下载、API Key 配置、启动和课堂使用说明。
- Create: `src/classbot/config.py` — 环境变量和默认设置。
- Create: `src/classbot/domain.py` — transcript、候选问题和答案卡片数据类型。
- Create: `src/classbot/__main__.py` — `python -m classbot` 入口。
- Create: `src/classbot/audio/capture.py` — sounddevice 输入流与有界帧队列。
- Create: `src/classbot/audio/segmenter.py` — 20 ms PCM 帧的 VAD 与语音段结束判断。
- Create: `src/classbot/asr/protocol.py` — ASR 协议和事件类型别名。
- Create: `src/classbot/asr/funasr_streaming.py` — FunASR 流式模型适配器。
- Create: `src/classbot/question_detector.py` — 中文问题候选检测、去重和修订。
- Create: `src/classbot/llm/deepseek_client.py` — DeepSeek 请求、SSE 解析、取消和错误转换。
- Create: `src/classbot/pipeline.py` — ASR 事件到问题卡片和答案流的协调器。
- Create: `src/classbot/ui/main_window.py` — 置顶悬浮窗和问题卡片控件。
- Create: `src/classbot/main.py` — 依赖组装、qasync 事件循环和进程入口。
- Create: `tests/test_question_detector.py` — 问题识别和修订测试。
- Create: `tests/test_deepseek_client.py` — fake SSE、请求载荷和错误测试。
- Create: `tests/test_segmenter.py` — VAD 片段边界测试。
- Create: `tests/test_funasr_streaming.py` — mock FunASR 模型和事件测试。
- Create: `tests/test_pipeline.py` — fake ASR/LLM 的端到端协调测试。
- Create: `tests/test_config.py` — 设置解析测试。

本计划保持一个可独立验收的 MVP；说话人识别、教材检索、移动端和语音播报不进入本轮。

### Task 1: 初始化 Python 工程和领域类型

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `src/classbot/__init__.py`
- Create: `src/classbot/config.py`
- Create: `src/classbot/domain.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 写设置解析的失败测试**

```python
# tests/test_config.py
from classbot.config import Settings


def test_settings_from_env_uses_deepseek_defaults(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("CLASSBOT_SUBJECT", "历史")

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "test-key"
    assert settings.subject == "历史"
    assert settings.model == "deepseek-flash"
    assert settings.base_url == "https://api.deepseek.com"


def test_settings_allows_missing_key_for_ui_configuration(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.deepseek_api_key is None

```

- [ ] **Step 2: 安装开发依赖并运行测试，确认测试先失败**

Run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\pytest.exe tests/test_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'classbot'` or `No module named 'classbot.config'` because the package files are not implemented yet.

- [ ] **Step 3: 写工程配置和最小领域类型**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=75,<76"]
build-backend = "setuptools.build_meta"

[project]
name = "classbot"
version = "0.1.0"
description = "Low-latency classroom question assistant"
requires-python = ">=3.11,<3.13"
dependencies = [
    "PySide6>=6.8,<7",
    "qasync>=0.27,<1",
    "httpx>=0.28,<1",
    "numpy>=2,<3",
    "python-dotenv>=1,<2",
    "sounddevice>=0.5,<1",
    "webrtcvad-wheels>=2.0.14,<3",
]

[project.optional-dependencies]
asr = [
    "funasr>=1.2,<2",
    "torch>=2.5,<3",
    "torchaudio>=2.5,<3",
]
dev = [
    "pytest>=8,<9",
    "pytest-asyncio>=0.24,<1",
    "ruff>=0.9,<1",
]

[project.scripts]
classbot = "classbot.main:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"

[tool.ruff]
line-length = 100
target-version = "py311"
```

```gitignore
# .gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
*.py[cod]
.env
*.log
models/
.model_cache/
dist/
build/
*.egg-info/
```

```dotenv
# .env.example
DEEPSEEK_API_KEY=
CLASSBOT_SUBJECT=历史
CLASSBOT_MODEL=deepseek-flash
CLASSBOT_BASE_URL=https://api.deepseek.com
CLASSBOT_ASR_DEVICE=cpu
```

```python
# src/classbot/config.py
from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    deepseek_api_key: str | None
    subject: str
    model: str
    base_url: str
    asr_device: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            subject=os.getenv("CLASSBOT_SUBJECT", "历史"),
            model=os.getenv("CLASSBOT_MODEL", "deepseek-flash"),
            base_url=os.getenv("CLASSBOT_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            asr_device=os.getenv("CLASSBOT_ASR_DEVICE", "cpu"),
        )
```

```python
# src/classbot/domain.py
from dataclasses import dataclass
from enum import StrEnum


class CardStatus(StrEnum):
    DETECTED = "detected"
    ANSWERING = "answering"
    ANSWERED = "answered"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    text: str
    is_final: bool
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 0.0
    pause_ms: int = 0
    segment_id: int | str | None = None


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    question_id: str
    text: str
    subject: str
    confidence: float
    is_final: bool


@dataclass(slots=True)
class AnswerCard:
    question_id: str
    question: str
    subject: str
    answer: str = ""
    status: CardStatus = CardStatus.DETECTED
    error: str | None = None
```

Create an empty `src/classbot/__init__.py` and ensure it contains only:

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: 运行设置测试，确认通过**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_config.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: 提交工程骨架**

```powershell
git add pyproject.toml .gitignore .env.example src tests/test_config.py
git commit -m "chore: bootstrap classbot python project"
```

### Task 2: 实现中文问题检测和去重

**Files:**
- Create: `src/classbot/question_detector.py`
- Test: `tests/test_question_detector.py`

- [ ] **Step 1: 写失败测试，覆盖普通陈述、partial 提问、final 修订和重复事件**

```python
# tests/test_question_detector.py
from classbot.domain import TranscriptEvent
from classbot.question_detector import QuestionDetector


def test_statement_does_not_create_candidate():
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(
        TranscriptEvent("马克思出生于德国特里尔", is_final=True, pause_ms=500)
    )

    assert candidates == []


def test_partial_question_creates_one_candidate_early():
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(TranscriptEvent("马克思的生辰是什么", is_final=False))

    assert len(candidates) == 1
    assert candidates[0].text == "马克思的生辰是什么"
    assert candidates[0].is_final is False


def test_final_revision_keeps_question_id_and_updates_text():
    detector = QuestionDetector(subject="历史")
    first = detector.feed(TranscriptEvent("马克思的生辰是什么", is_final=False))[0]

    revised = detector.feed(
        TranscriptEvent("马克思的生辰是什么？", is_final=True, pause_ms=400)
    )

    assert len(revised) == 1
    assert revised[0].question_id == first.question_id
    assert revised[0].is_final is True


def test_identical_final_event_is_not_submitted_twice():
    detector = QuestionDetector(subject="历史")
    event = TranscriptEvent("马克思的生辰是什么？", is_final=True, pause_ms=400)

    assert len(detector.feed(event)) == 1
    assert detector.feed(event) == []

```

- [ ] **Step 2: 运行检测器测试，确认实现不存在**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_question_detector.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'classbot.question_detector'`.

- [ ] **Step 3: 实现归一化、评分、候选复用和修订逻辑**

```python
# src/classbot/question_detector.py
from __future__ import annotations

import re
from difflib import SequenceMatcher
from uuid import uuid4

from classbot.domain import QuestionCandidate, TranscriptEvent


_MARKERS = (
    "什么",
    "哪年",
    "哪一年",
    "哪个",
    "谁",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "多少",
    "是否",
    "吗",
    "呢",
    "？",
    "?",
)


def normalize_text(text: str) -> str:
    return re.sub(r"[\s，。！!、；;：:（）()【】\[\]]", "", text).strip()


class QuestionDetector:
    def __init__(self, subject: str, minimum_length: int = 8) -> None:
        self.subject = subject
        self.minimum_length = minimum_length
        self._active: dict[str, tuple[str, bool]] = {}

    def feed(self, event: TranscriptEvent) -> list[QuestionCandidate]:
        text = normalize_text(event.text)
        if len(text) < self.minimum_length:
            return []

        marker_score = 0.65 if any(marker in text for marker in _MARKERS) else 0.0
        length_score = 0.15 if len(text) >= self.minimum_length else 0.0
        final_score = 0.10 if event.is_final else 0.0
        pause_score = 0.10 if event.pause_ms >= 300 else 0.0
        score = min(marker_score + length_score + final_score + pause_score, 1.0)

        early_ready = marker_score > 0 and len(text) >= self.minimum_length
        final_ready = event.is_final and score >= 0.65
        if not early_ready and not final_ready:
            return []

        matching_id: str | None = None
        for question_id, (previous, previous_final) in self._active.items():
            ratio = SequenceMatcher(None, previous, text).ratio()
            if previous in text or text in previous or ratio >= 0.72:
                matching_id = question_id
                break

        if matching_id is None:
            matching_id = uuid4().hex
        else:
            previous, previous_final = self._active[matching_id]
            if previous == text and previous_final >= event.is_final:
                return []

        self._active[matching_id] = (text, event.is_final)
        return [
            QuestionCandidate(
                question_id=matching_id,
                text=text,
                subject=self.subject,
                confidence=score,
                is_final=event.is_final,
            )
        ]

```

- [ ] **Step 4: 运行检测器测试，确认通过**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_question_detector.py -q
```

Expected: `4 passed`.

- [ ] **Step 5: 提交问题检测器**

```powershell
git add src/classbot/question_detector.py tests/test_question_detector.py
git commit -m "feat: detect and deduplicate classroom questions"
```

### Task 3: 实现 DeepSeek Flash 流式客户端

**Files:**
- Create: `src/classbot/llm/__init__.py`
- Create: `src/classbot/llm/deepseek_client.py`
- Test: `tests/test_deepseek_client.py`

- [ ] **Step 1: 写 fake SSE 测试，先锁定请求和增量输出契约**

```python
# tests/test_deepseek_client.py
import json

import httpx

from classbot.config import Settings
from classbot.llm.deepseek_client import DeepSeekClient


def sse(*contents: str) -> bytes:
    chunks = []
    for content in contents:
        chunks.append(
            "data: "
            + json.dumps(
                {"choices": [{"delta": {"content": content}}]},
                ensure_ascii=False,
            )
            + "\n\n"
        )
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


async def test_stream_answer_sends_short_answer_request_and_yields_deltas():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/chat/completions"
        assert body["model"] == "deepseek-flash"
        assert body["stream"] is True
        assert "马克思的生辰是什么" in body["messages"][-1]["content"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse("马克思", "出生于", "1818 年。"),
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(
        base_url="https://api.deepseek.com",
        transport=transport,
    )
    settings = Settings("test-key", "历史", "deepseek-flash", "https://api.deepseek.com", "cpu")
    client = DeepSeekClient(settings, http_client=http_client)

    chunks = [
        chunk
        async for chunk in client.stream_answer("马克思的生辰是什么", "历史")
    ]

    assert chunks == ["马克思", "出生于", "1818 年。"]
    await client.aclose()


async def test_missing_api_key_fails_before_network_call():
    settings = Settings(None, "历史", "deepseek-flash", "https://api.deepseek.com", "cpu")
    client = DeepSeekClient(settings, http_client=httpx.AsyncClient())

    try:
        result = [chunk async for chunk in client.stream_answer("问题", "历史")]
    except ValueError as exc:
        assert "DEEPSEEK_API_KEY" in str(exc)
    else:
        raise AssertionError(f"expected ValueError, got {result!r}")
    finally:
        await client.aclose()

```

- [ ] **Step 2: 运行客户端测试，确认模块不存在**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_deepseek_client.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'classbot.llm'`.

- [ ] **Step 3: 实现请求载荷、SSE 解析和连接关闭**

```python
# src/classbot/llm/deepseek_client.py
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from classbot.config import Settings


SYSTEM_PROMPT = (
    "你是课堂简答助手。请根据给定学科回答问题。"
    "只给出直接、准确、简短的答案，优先使用 1～3 句中文。"
    "如果事实不确定，请明确说‘不确定’，不要编造。"
    "不要输出推理过程、寒暄、标题或 Markdown 代码块。"
)


class DeepSeekClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        self._http = http_client or httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def stream_answer(self, question: str, subject: str) -> AsyncIterator[str]:
        if not self._settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is not configured")

        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"学科：{subject}\n问题：{question}",
                },
            ],
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 120,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.deepseek_api_key}",
            "Accept": "text/event-stream",
        }

        async with self._http.stream(
            "POST",
            "/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    async def aclose(self) -> None:
        await self._http.aclose()
```

Create `src/classbot/llm/__init__.py` with:

```python
from classbot.llm.deepseek_client import DeepSeekClient

__all__ = ["DeepSeekClient"]
```

- [ ] **Step 4: 运行 DeepSeek 客户端测试，确认通过**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_deepseek_client.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: 提交流式客户端**

```powershell
git add src/classbot/llm tests/test_deepseek_client.py
git commit -m "feat: add streaming deepseek answer client"
```

### Task 4: 接入 PCM 采集、VAD 和 FunASR 流式识别

**Files:**
- Create: `src/classbot/audio/__init__.py`
- Create: `src/classbot/audio/capture.py`
- Create: `src/classbot/audio/segmenter.py`
- Create: `src/classbot/asr/__init__.py`
- Create: `src/classbot/asr/protocol.py`
- Create: `src/classbot/asr/funasr_streaming.py`
- Test: `tests/test_segmenter.py`
- Test: `tests/test_funasr_streaming.py`

- [ ] **Step 1: 写不依赖真实麦克风的 VAD 测试**

```python
# tests/test_segmenter.py
from classbot.audio.segmenter import SpeechSegmenter


def pcm_frame(value: int = 1000, samples: int = 320) -> bytes:
    return (value.to_bytes(2, "little", signed=True)) * samples


def test_speech_segmenter_reports_end_after_silence(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=60)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: frame != b"silent")

    assert segmenter.accept(pcm_frame()).speech_started is True
    assert segmenter.accept(b"silent").speech_ended is False
    assert segmenter.accept(b"silent").speech_ended is False
    assert segmenter.accept(b"silent").speech_ended is True

```

- [ ] **Step 2: 写 mock FunASR 测试，锁定 partial/final 事件**

```python
# tests/test_funasr_streaming.py
import asyncio

from classbot.asr.funasr_streaming import FunAsrStreamingRecognizer


class FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [{"text": "马克思的生辰是什么" if kwargs["is_final"] else "马克思的生辰"}]


async def test_funasr_adapter_emits_partial_and_final(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(
        "classbot.asr.funasr_streaming.build_model",
        lambda device: fake_model,
    )
    recognizer = FunAsrStreamingRecognizer(device="cpu", chunk_samples=160)
    await recognizer.start()

    recognizer.push_audio(b"\x00\x00" * 160)
    partial = await asyncio.wait_for(anext(recognizer.events()), timeout=1)
    assert partial.is_final is False

    recognizer.push_audio(b"\x00\x00" * 80)
    recognizer.finish_segment()
    final = await asyncio.wait_for(anext(recognizer.events()), timeout=1)
    assert final.is_final is True

    await recognizer.stop()

```

- [ ] **Step 3: 运行音频/ASR 测试，确认实现不存在**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_segmenter.py tests/test_funasr_streaming.py -q
```

Expected: FAIL with missing module errors.

- [ ] **Step 4: 实现 16 kHz、20 ms 音频帧和有界队列**

```python
# src/classbot/audio/capture.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import sounddevice as sd


class AudioCapture:
    def __init__(self, device: int | None = None, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.blocksize = sample_rate // 50
        self._device = device
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.RawInputStream | None = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            dtype="int16",
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, indata, frames, time_info, status) -> None:
        del frames, time_info, status
        if not self._loop or not self._running:
            return
        data = bytes(indata)

        def enqueue() -> None:
            if not self._running:
                return
            if self._queue.full():
                self._queue.get_nowait()
            self._queue.put_nowait(data)

        self._loop.call_soon_threadsafe(enqueue)

    async def frames(self) -> AsyncIterator[bytes]:
        while self._running:
            yield await self._queue.get()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

```

```python
# src/classbot/audio/segmenter.py
from __future__ import annotations

from dataclasses import dataclass

import webrtcvad


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    speech_started: bool = False
    speech_ended: bool = False


class SpeechSegmenter:
    def __init__(self, sample_rate: int = 16_000, end_silence_ms: int = 320) -> None:
        self._vad = webrtcvad.Vad(2)
        self._sample_rate = sample_rate
        self._frame_ms = 20
        self._end_silence_ms = end_silence_ms
        self._silent_ms = 0
        self._in_speech = False

    def _is_speech(self, frame: bytes) -> bool:
        return self._vad.is_speech(frame, self._sample_rate)

    def accept(self, frame: bytes) -> SegmentEvent:
        if self._is_speech(frame):
            started = not self._in_speech
            self._in_speech = True
            self._silent_ms = 0
            return SegmentEvent(speech_started=started)

        if not self._in_speech:
            return SegmentEvent()

        self._silent_ms += self._frame_ms
        if self._silent_ms < self._end_silence_ms:
            return SegmentEvent()

        self._in_speech = False
        self._silent_ms = 0
        return SegmentEvent(speech_ended=True)

```

- [ ] **Step 5: 实现 ASR 协议和 FunASR cache worker**

```python
# src/classbot/asr/protocol.py
from collections.abc import AsyncIterator, Protocol

from classbot.domain import TranscriptEvent


class SpeechRecognizer(Protocol):
    async def start(self) -> None: ...

    def push_audio(self, frame: bytes) -> None: ...

    def finish_segment(self) -> None: ...

    def events(self) -> AsyncIterator[TranscriptEvent]: ...

    async def stop(self) -> None: ...
```

```python
# src/classbot/asr/funasr_streaming.py
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np

from classbot.domain import TranscriptEvent


def build_model(device: str):
    from funasr import AutoModel

    return AutoModel(model="paraformer-zh-streaming", device=device)


class FunAsrStreamingRecognizer:
    def __init__(self, device: str = "cpu", chunk_samples: int = 9_600) -> None:
        self._device = device
        self._chunk_samples = chunk_samples
        self._model = None
        self._cache: dict = {}
        self._audio = bytearray()
        self._input: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue(maxsize=100)
        self._output: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._running = False
        self._segment_ms = 0
        self._segment_id = 0
        self._last_text = ""

    async def start(self) -> None:
        self._model = await asyncio.to_thread(build_model, self._device)
        self._running = True
        self._worker = asyncio.create_task(self._run())

    def push_audio(self, frame: bytes) -> None:
        if not self._running:
            return
        if self._input.full():
            self._input.get_nowait()
        self._input.put_nowait(("audio", frame))

    def finish_segment(self) -> None:
        if self._running:
            self._input.put_nowait(("finish", b""))

    async def _run(self) -> None:
        while self._running:
            kind, data = await self._input.get()
            if kind == "audio":
                self._audio.extend(data)
                await self._infer_ready_chunks()
            elif kind == "finish":
                emitted = False
                if self._audio:
                    emitted = await self._infer(bytes(self._audio), is_final=True)
                if not emitted and self._last_text:
                    await self._output.put(
                        TranscriptEvent(
                            text=self._last_text,
                            is_final=True,
                            start_ms=self._segment_ms,
                            end_ms=self._segment_ms,
                            confidence=0.0,
                            segment_id=self._segment_id,
                        )
                    )
                self._audio.clear()
                self._cache = {}
                self._last_text = ""
                self._segment_ms = 0
                self._segment_id += 1

    async def _infer_ready_chunks(self) -> None:
        chunk_bytes = self._chunk_samples * 2
        while len(self._audio) >= chunk_bytes:
            chunk = bytes(self._audio[:chunk_bytes])
            del self._audio[:chunk_bytes]
            await self._infer(chunk, is_final=False)

    async def _infer(self, data: bytes, is_final: bool) -> bool:
        if self._model is None:
            return False
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        result = await asyncio.to_thread(
            self._model.generate,
            input=samples,
            cache=self._cache,
            is_final=is_final,
            chunk_size=[0, 10, 5],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )
        text = (result[0].get("text") if result else "") or ""
        if not text:
            return False
        self._last_text = text
        duration_ms = int(len(samples) / 16)
        event = TranscriptEvent(
            text=text,
            is_final=is_final,
            start_ms=self._segment_ms,
            end_ms=self._segment_ms + duration_ms,
            confidence=0.0,
            segment_id=self._segment_id,
        )
        self._segment_ms += duration_ms
        await self._output.put(event)
        return True

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        while self._running or not self._output.empty():
            event = await self._output.get()
            if event is None:
                return
            yield event

    async def stop(self) -> None:
        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        await self._output.put(None)
```

Create `src/classbot/audio/__init__.py` and `src/classbot/asr/__init__.py` as empty package markers. In `src/classbot/asr/__init__.py`, export `FunAsrStreamingRecognizer`.

FunASR 产生的每个 `TranscriptEvent` 都必须携带当前递增的 `segment_id`；`start_ms` 和 `end_ms` 只表示段内时间元数据，不能作为段身份。`QuestionDetector.begin_segment(segment_id=None)` 只为没有显式 ID 的调用方切换 fallback identity，不得清理已有 per-segment active 状态或迟到事件的可复用状态。

- [ ] **Step 6: 运行 mock 测试并修复接口细节**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_segmenter.py tests/test_funasr_streaming.py -q
```

Expected: `2 passed`. The test must not download a model or access a microphone; it replaces `build_model` with `FakeModel`.

- [ ] **Step 7: 提交音频和 ASR 层**

```powershell
git add src/classbot/audio src/classbot/asr tests/test_segmenter.py tests/test_funasr_streaming.py
git commit -m "feat: add local streaming audio recognition layer"
```

### Task 5: 协调候选问题、取消旧请求和答案卡片

**Files:**
- Create: `src/classbot/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: 写 fake ASR/LLM 协调测试**

```python
# tests/test_pipeline.py
import asyncio

from classbot.domain import QuestionCandidate, TranscriptEvent, CardStatus
from classbot.pipeline import LivePipeline


class FakeRecognizer:
    def __init__(self, events):
        self._events = events
        self.pushed = []

    async def start(self):
        return None

    def push_audio(self, frame):
        self.pushed.append(frame)

    def finish_segment(self):
        return None

    async def events(self):
        for event in self._events:
            yield event

    async def stop(self):
        return None


class FakeDetector:
    def __init__(self):
        self.calls = []

    def feed(self, event):
        self.calls.append(event)
        if event.text == "马克思的生辰是什么":
            return [QuestionCandidate("q1", event.text, "历史", 0.9, event.is_final)]
        return []


class FakeClient:
    async def stream_answer(self, question, subject):
        assert question == "马克思的生辰是什么"
        assert subject == "历史"
        for chunk in ("马克思", "出生于", "1818 年。"):
            await asyncio.sleep(0)
            yield chunk


async def test_pipeline_creates_card_and_accumulates_streaming_answer():
    recognizer = FakeRecognizer(
        [TranscriptEvent("马克思的生辰是什么", is_final=True, pause_ms=400)]
    )
    detector = FakeDetector()
    cards = []
    deltas = []
    pipeline = LivePipeline(
        recognizer=recognizer,
        detector=detector,
        client=FakeClient(),
        on_card=lambda card: cards.append(card),
        on_delta=lambda question_id, delta: deltas.append((question_id, delta)),
    )

    await pipeline.run()

    assert len(cards) == 1
    assert cards[0].status is CardStatus.ANSWERED
    assert cards[0].answer == "马克思出生于1818 年。"
    assert [answer for _, answer in deltas] == [
        "马克思",
        "马克思出生于",
        "马克思出生于1818 年。",
    ]

```

- [ ] **Step 2: 运行 pipeline 测试，确认协调器不存在**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_pipeline.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'classbot.pipeline'`.

- [ ] **Step 3: 实现卡片回调、增量拼接和同 ID 修订取消**

```python
# src/classbot/pipeline.py
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from classbot.domain import AnswerCard, CardStatus, QuestionCandidate, TranscriptEvent


Callback = Callable[..., None | Awaitable[None]]


async def invoke(callback: Callback, *args) -> None:
    result = callback(*args)
    if asyncio.iscoroutine(result):
        await result


class LivePipeline:
    def __init__(
        self,
        recognizer,
        detector,
        client,
        on_card: Callback,
        on_delta: Callback,
        on_status: Callback | None = None,
    ) -> None:
        self.recognizer = recognizer
        self.detector = detector
        self.client = client
        self.on_card = on_card
        self.on_delta = on_delta
        self.on_status = on_status or (lambda status: None)
        self._cards: dict[str, AnswerCard] = {}
        self._submitted_text: dict[str, str] = {}
        self._answer_tasks: dict[str, asyncio.Task] = {}
        self._stopped = False
        self._started = False

    async def run(self) -> None:
        await self.prepare()
        await invoke(self.on_status, "listening")
        try:
            async for event in self.recognizer.events():
                await self._handle_transcript(event)
        finally:
            await self.stop()

    async def prepare(self) -> None:
        if not self._started:
            self._stopped = False
            await self.recognizer.start()
            self._started = True

    async def _handle_transcript(self, event: TranscriptEvent) -> None:
        for candidate in self.detector.feed(event):
            await self._handle_candidate(candidate)

    async def _handle_candidate(self, candidate: QuestionCandidate) -> None:
        card = self._cards.get(candidate.question_id)
        previous_text = self._submitted_text.get(candidate.question_id)

        if card is None:
            card = AnswerCard(
                question_id=candidate.question_id,
                question=candidate.text,
                subject=candidate.subject,
            )
            self._cards[candidate.question_id] = card
            await invoke(self.on_card, card)
        elif previous_text == candidate.text:
            return
        else:
            card.question = candidate.text
            old_task = self._answer_tasks.get(candidate.question_id)
            if old_task and not old_task.done():
                old_task.cancel()
                await asyncio.gather(old_task, return_exceptions=True)
            card.answer = ""
            card.status = CardStatus.DETECTED
            await invoke(self.on_card, card)

        self._submitted_text[candidate.question_id] = candidate.text
        task = asyncio.create_task(self._answer(card, candidate))
        self._answer_tasks[candidate.question_id] = task
        if candidate.is_final:
            await asyncio.gather(task, return_exceptions=False)

    async def _answer(self, card: AnswerCard, candidate: QuestionCandidate) -> None:
        card.status = CardStatus.ANSWERING
        await invoke(self.on_card, card)
        try:
            async for delta in self.client.stream_answer(candidate.text, candidate.subject):
                card.answer += delta
                await invoke(self.on_delta, candidate.question_id, card.answer)
            card.status = CardStatus.ANSWERED
            await invoke(self.on_card, card)
        except asyncio.CancelledError:
            card.status = CardStatus.CANCELLED
            await invoke(self.on_card, card)
            raise
        except Exception as exc:
            card.status = CardStatus.ERROR
            card.error = str(exc)
            await invoke(self.on_card, card)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        for task in self._answer_tasks.values():
            if not task.done():
                task.cancel()
        if self._answer_tasks:
            await asyncio.gather(*self._answer_tasks.values(), return_exceptions=True)
        await self.recognizer.stop()
        self._started = False
        await invoke(self.on_status, "stopped")
```

- [ ] **Step 4: 运行 pipeline 测试，确认通过**

Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_pipeline.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: 增加修订问题测试并提交 pipeline**

Add a second fake-client test where the first candidate yields one delayed chunk, then a candidate with the same `question_id` and expanded text arrives. Assert the first task is cancelled, only the expanded text is sent to the client, and the card has one final answer. Run:

```powershell
.\.venv\Scripts\pytest.exe tests/test_pipeline.py -q
git add src/classbot/pipeline.py tests/test_pipeline.py
git commit -m "feat: coordinate question cards and streaming answers"
```

Expected: all pipeline tests pass.

### Task 6: 创建可读的 Windows 悬浮窗

**Files:**
- Create: `src/classbot/ui/__init__.py`
- Create: `src/classbot/ui/main_window.py`
- Test: manual UI smoke test, no Qt snapshot test required for the first MVP

- [ ] **Step 1: 实现不阻塞的窗口布局和状态更新 API**

```python
# src/classbot/ui/main_window.py
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from classbot.domain import AnswerCard, CardStatus


class MainWindow(QMainWindow):
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, subject: str) -> None:
        super().__init__()
        self.setWindowTitle(f"Classbot · {subject}")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(420, 560)
        self._items: dict[str, QListWidgetItem] = {}
        self._status = QLabel("已停止")
        self._transcript = QLabel("等待转写…")
        self._transcript.setWordWrap(True)
        self._cards = QListWidget()
        self._toggle = QPushButton("开始监听")
        self._toggle.clicked.connect(self._toggle_clicked)

        root = QWidget()
        layout = QVBoxLayout(root)
        top = QHBoxLayout()
        top.addWidget(self._status)
        top.addStretch()
        top.addWidget(self._toggle)
        layout.addLayout(top)
        layout.addWidget(QLabel("当前转写"))
        layout.addWidget(self._transcript)
        layout.addWidget(QLabel("问题与回答"))
        layout.addWidget(self._cards, 1)
        self.setCentralWidget(root)

    def _toggle_clicked(self) -> None:
        if self._toggle.text() == "开始监听":
            self._toggle.setText("暂停监听")
            self.start_requested.emit()
        else:
            self._toggle.setText("开始监听")
            self.stop_requested.emit()

    def set_status(self, status: str) -> None:
        labels = {
            "listening": "● 正在监听",
            "stopped": "已停止",
            "asr_error": "ASR 不可用",
            "api_error": "API 错误",
        }
        self._status.setText(labels.get(status, status))

    def set_transcript(self, text: str) -> None:
        self._transcript.setText(text or "等待转写…")

    def upsert_card(self, card: AnswerCard) -> None:
        text = f"问题：{card.question}\n回答：{card.answer or '等待回答…'}"
        if card.status is CardStatus.ERROR:
            text += f"\n错误：{card.error}"
        if card.question_id not in self._items:
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, card.question_id)
            self._items[card.question_id] = item
            self._cards.insertItem(0, item)
        else:
            self._items[card.question_id].setText(text)

    def update_answer(self, question_id: str, answer: str) -> None:
        item = self._items.get(question_id)
        if item is not None:
            current = item.text()
            question_line = current.split(chr(10), 1)[0]
            item.setText(f"{question_line}{chr(10)}回答：{answer or '等待回答…'}")
```

The callback receives the full accumulated answer, not only the latest token. This prevents the initial `等待回答…` text from remaining in the card and makes repeated partial updates idempotent.

- [ ] **Step 2: 手工运行最小 Qt 窗口，确认置顶、暂停按钮和卡片排版可读**

Run:

```powershell
.\.venv\Scripts\python.exe -c "from PySide6.QtWidgets import QApplication; import sys; from classbot.ui.main_window import MainWindow; app=QApplication(sys.argv); w=MainWindow('历史'); w.show(); app.exec()"
```

Expected: a small always-on-top window appears; clicking the button toggles between `开始监听` and `暂停监听`; no terminal traceback occurs.

- [ ] **Step 3: 提交 UI**

```powershell
git add src/classbot/ui
git commit -m "feat: add classroom floating answer window"
```

### Task 7: 组装 qasync 运行入口和可诊断配置

**Files:**
- Create: `src/classbot/main.py`
- Modify: `src/classbot/ui/main_window.py`
- Create: `README.md`
- Modify: `.env.example`

- [ ] **Step 1: 写启动入口，明确依赖组装顺序**

`main.py` must construct these objects in order: `Settings` → `MainWindow` → `AudioCapture` → `SpeechSegmenter` → `FunAsrStreamingRecognizer` → `QuestionDetector` → `DeepSeekClient` → `LivePipeline`. The audio loop must call `detector.begin_segment()` when `SpeechSegmenter.accept(frame).speech_started` is true, call `recognizer.push_audio(frame)` for every frame, and call `recognizer.finish_segment()` when `speech_ended` is true. `start_ms` and `end_ms` are timing metadata only and must not be used as a segment identity.

Implement the qasync entry point as:

```python
# src/classbot/main.py
from __future__ import annotations

import asyncio
import sys

from PySide6.QtWidgets import QApplication
from qasync import QEventLoop

from classbot.asr.funasr_streaming import FunAsrStreamingRecognizer
from classbot.audio.capture import AudioCapture
from classbot.audio.segmenter import SpeechSegmenter
from classbot.config import Settings
from classbot.llm.deepseek_client import DeepSeekClient
from classbot.pipeline import LivePipeline
from classbot.question_detector import QuestionDetector
from classbot.ui.main_window import MainWindow


async def run_app(window: MainWindow, pipeline: LivePipeline, capture: AudioCapture, segmenter: SpeechSegmenter) -> None:
    async def feed_audio() -> None:
        capture.start()
        async for frame in capture.frames():
            recognizer = pipeline.recognizer
            recognizer.push_audio(frame)
            segment_event = segmenter.accept(frame)
            if segment_event.speech_started:
                pipeline.detector.begin_segment()
            if segment_event.speech_ended:
                recognizer.finish_segment()

    audio_task: asyncio.Task | None = None

    async def start() -> None:
        nonlocal audio_task
        if audio_task is None or audio_task.done():
            try:
                await pipeline.prepare()
            except Exception:
                window.set_status("asr_error")
                return
            audio_task = asyncio.create_task(feed_audio())
            asyncio.create_task(pipeline.run())

    def stop() -> None:
        capture.stop()
        asyncio.create_task(pipeline.stop())

    window.start_requested.connect(lambda: asyncio.create_task(start()))
    window.stop_requested.connect(stop)
    window.show()


def main() -> None:
    settings = Settings.from_env()
    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow(settings.subject)
    capture = AudioCapture()
    segmenter = SpeechSegmenter()
    recognizer = FunAsrStreamingRecognizer(device=settings.asr_device)
    detector = QuestionDetector(settings.subject)
    client = DeepSeekClient(settings)
    pipeline = LivePipeline(
        recognizer=recognizer,
        detector=detector,
        client=client,
        on_card=window.upsert_card,
        on_delta=window.update_answer,
        on_status=window.set_status,
    )
    with loop:
        loop.run_until_complete(run_app(window, pipeline, capture, segmenter))
        loop.run_forever()


if __name__ == "__main__":
    main()
```

Create the module entry point required by the README command:

```python
# src/classbot/__main__.py
from classbot.main import main


if __name__ == "__main__":
    main()
```

The `on_delta` binding above receives the full accumulated answer and calls `window.update_answer`, so the initial text is replaced idempotently on every streamed update. Do not append raw token deltas directly to the list item.

- [ ] **Step 2: 加入 README 的真实安装和安全说明**

`README.md` must include these exact commands and facts:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,asr]"
Copy-Item .env.example .env
notepad .env
.\.venv\Scripts\python.exe -m classbot
```


The README must state that the first ASR launch downloads `paraformer-zh-streaming`, that the DeepSeek key is read from `.env`, that raw audio is not written by the application, and that classroom use requires appropriate consent.

- [ ] **Step 3: 运行 fake-free import smoke test**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import classbot.main; import classbot.ui.main_window; print('imports ok')"
```

Expected: `imports ok` and no import-time model download. Model construction must happen only after the user presses `开始监听`.

- [ ] **Step 4: 提交运行入口和使用文档**

```powershell
git add src/classbot/main.py src/classbot/__main__.py src/classbot/ui/main_window.py README.md .env.example
git commit -m "feat: wire desktop runtime and setup documentation"
```

### Task 8: 集成验证、延迟测量和最终修复

**Files:**
- Modify: `src/classbot/question_detector.py` when detector tests or lint expose a defect
- Modify: `src/classbot/llm/deepseek_client.py` when SSE tests or lint expose a defect
- Modify: `src/classbot/asr/funasr_streaming.py` when ASR tests or lint expose a defect
- Modify: `src/classbot/pipeline.py` when coordination tests or lint expose a defect
- Modify: `src/classbot/ui/main_window.py` when UI smoke checks expose a defect
- Modify: `src/classbot/main.py` when runtime smoke checks expose a defect
- Create: `tests/fixtures/transcripts.json`

- [ ] **Step 1: 添加不含真实课堂音频的转写 fixture**

Create `tests/fixtures/transcripts.json` with three authorized synthetic cases:

```json
[
  {"text": "马克思的生辰是什么", "is_final": false},
  {"text": "马克思的生辰是什么？", "is_final": true},
  {"text": "马克思出生于德国特里尔", "is_final": true}
]
```

- [ ] **Step 2: 运行完整自动化检查并修复所有失败**

Run:

```powershell
.\.venv\Scripts\pytest.exe -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m compileall src
```

Expected: all tests pass, ruff reports no errors, and compileall exits with code 0. Do not claim the microphone path works from these commands alone.

- [ ] **Step 3: 做一次设备和 API 手工验收**

With a valid key and the approved classroom/test recording:

1. Start the app and verify the initial state is `已停止` and no model download begins.
2. Click `开始监听`; verify the state becomes `● 正在监听` and the microphone device opens.
3. Say “马克思的生辰是什么”; verify one question card appears and its answer streams into the same card.
4. Say a second question before the first answer finishes; verify two cards remain separate.
5. Click `暂停监听`; verify new audio does not start a request.
6. Revoke or remove the API key; verify the card shows an understandable error and the process remains alive.
7. Search the project directory after exit; verify no `.wav`, `.mp3`, or raw audio file was created.

- [ ] **Step 4: 记录并检查延迟预算**

Use the existing `candidate_detected_at`, `request_started_at`, and `first_answer_delta_at` timestamps. Run 10 synthetic or explicitly authorized spoken questions on the same network and record median candidate-to-first-delta time. The MVP acceptance target is a median no higher than 2 seconds; if it is higher, report the measured ASR, network, and model components separately instead of masking the result.

- [ ] **Step 5: 提交最终验证结果**

```powershell
git status --short
git log --oneline --decorate -8
git add tests/fixtures/transcripts.json
git commit -m "test: add classroom question fixtures"
```

The final handoff must link `README.md`, the design document, and this plan, and must distinguish automated test success from real microphone/API acceptance.










