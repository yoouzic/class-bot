# Live ASR Latency and Answer Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** Keep microphone frames flowing across speech-segment boundaries, remove the blocking full-segment re-decode from the live path, and give high-thinking answers enough output budget to avoid truncated classroom replies.

**Architecture:** The recognizer will emit the streaming/final-cache transcript immediately in low-latency mode and accept audio queued after a finish marker; the expensive full-segment refinement remains available only when explicitly enabled for offline accuracy tests. The DeepSeek client will use a larger completion budget only when thinking is enabled, while retaining the short-answer system instruction.

**Tech Stack:** Python 3.11, asyncio queues/tasks, FunASR Paraformer streaming, PySide6 runtime, httpx SSE client, pytest/pytest-asyncio, PowerShell replay evaluation.

---

### Task 1: Add failing ASR boundary tests

**Files:**
- Modify: \`tests/test_funasr_streaming.py\`

- [x] **Step 1: Add a regression test for audio queued after a finish marker**

\`\`\`python
@pytest.mark.asyncio
async def test_audio_after_finish_marker_is_processed_as_next_segment(monkeypatch):
    model = FakeModel(["第一段", "", "第二段"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    first = await _next_event(events)
    recognizer.finish_segment()
    recognizer.push_audio(_pcm(160, value=-2000))

    first_final = await _next_event(events)
    second = await _next_event(events)

    assert first.segment_id == first_final.segment_id == 0
    assert first_final.is_final is True
    assert second.segment_id == 1
    assert second.text == "第二段"
    await recognizer.stop()
\`\`\`

- [x] **Step 2: Run the focused test and confirm the expected failure**

Run:

\`\`\`powershell
.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_funasr_streaming.py -k after_finish_marker
\`\`\`

Expected: FAIL because \`push_audio()\` currently returns while \`_closing\` is true, so the second segment never reaches the worker.

- [x] **Step 3: Make the existing full-refinement tests explicit**

Change the two tests that assert full-segment re-decode to construct:

\`\`\`python
recognizer = FunAsrStreamingRecognizer(
    chunk_samples=160,
    final_refinement=True,
)
\`\`\`

This keeps the expensive behavior covered while making the live default testable separately.

### Task 2: Implement non-blocking live segment finalization

**Files:**
- Modify: \`src/classbot/asr/funasr_streaming.py\`
- Modify: \`src/classbot/main.py\`

- [x] **Step 1: Add an explicit refinement mode**

Add a constructor parameter:

\`\`\`python
final_refinement: bool = False,
\`\`\`

Store it as \`self.final_refinement\`. In \`_finish_current_segment()\`, only call \`_refine_segment()\` when this flag is true:

\`\`\`python
if self.final_refinement:
    refined_text = await self._refine_segment()
    if refined_text:
        self._segment_text = refined_text
        self._last_text = refined_text
\`\`\`

- [x] **Step 2: Stop discarding frames merely because a finish marker is pending**

Remove the early return from \`push_audio()\`:

\`\`\`python
if self._closing:
    return
\`\`\`

The \`_FINISH\` marker remains FIFO-ordered. Frames pushed after it stay in the bounded queue and are consumed only after the current segment resets, so they belong to the next segment rather than being mixed into the previous one.

- [x] **Step 3: Keep the application on the low-latency path**

Construct the recognizer in \`_make_recognizer()\` with:

\`\`\`python
final_refinement=False,
\`\`\`

Course hotwords and exact corrections still apply to each streaming inference; only the blocking full-segment replay is disabled for the microphone session.

- [x] **Step 4: Run the focused ASR tests**

\`\`\`powershell
.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_funasr_streaming.py tests/test_main.py
\`\`\`

Expected: all focused tests pass, including the new boundary regression and the explicit offline refinement tests.

### Task 3: Prevent high-thinking answer truncation

**Files:**
- Modify: \`tests/test_deepseek_client.py\`
- Modify: \`src/classbot/llm/deepseek_client.py\`

- [x] **Step 1: Change the high-thinking contract test first**

Update the high-thinking request assertion from:

\`\`\`python
assert payload["max_tokens"] == 512
\`\`\`

to:

\`\`\`python
assert payload["max_tokens"] == 1024
\`\`\`

- [x] **Step 2: Verify the focused test fails before production code changes**

\`\`\`powershell
.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_deepseek_client.py -k textbook
\`\`\`

Expected: FAIL because the current thinking branch sends \`max_tokens=512\`.

- [x] **Step 3: Increase only the thinking branch budget**

Change the payload expression to:

\`\`\`python
"max_tokens": 1024 if thinking_enabled else 120,
\`\`\`

Keep non-thinking classroom responses at 120 tokens for low latency.

- [x] **Step 4: Run the DeepSeek client tests**

\`\`\`powershell
.\\.venv\\Scripts\\python.exe -m pytest -q tests/test_deepseek_client.py tests/test_config.py
\`\`\`

Expected: all tests pass, including enabled thinking, textbook context, and disabled thinking contracts.

### Task 4: Verify the real recordings and document the result

**Files:**
- Modify: \`README.md\` only if the live refinement behavior needs a user-facing note.
- Generated/ignored: \`tmp/eval_marxism_realtime_120s_after_fix.json\`, \`tmp/eval_marxism_tail_40m_after_fix.json\`

- [x] **Step 1: Run the complete automated suite and lint**

\`\`\`powershell
.\\.venv\\Scripts\\python.exe -m ruff check src tests
.\\.venv\\Scripts\\python.exe -m pytest -q
powershell -ExecutionPolicy Bypass -File .\\start_classbot.ps1 -CheckOnly
\`\`\`

- [x] **Step 2: Replay 120 seconds at real-time speed**

\`\`\`powershell
.\\.venv\\Scripts\\python.exe tmp\\evaluate_recordings.py --file '9月16日 马克思主义哲学与辩证法课程讨论.m4a' --start 5820 --duration 120 --realtime --course 'marxism_basic_principles_2023' --subject '马克思主义基本原理' --output tmp\\eval_marxism_realtime_120s_after_fix.json
\`\`\`

Record frame loss, final-transcript latency median/P90, final segment count, and question candidates. The expected direction is zero frames discarded during segment closing and materially lower latency than the previous 6.64-second median / 17.73-second P90.

- [x] **Step 3: Replay the 40-minute high-frequency tail without artificial frame loss**

\`\`\`powershell
.\\.venv\\Scripts\\python.exe tmp\\evaluate_recordings.py --file '9月16日 马克思主义哲学与辩证法课程讨论.m4a' --start 5820 --duration 2400 --wait-for-closing --course 'marxism_basic_principles_2023' --subject '马克思主义基本原理' --output tmp\\eval_marxism_tail_40m_after_fix.json
\`\`\`

Compare final transcript/question counts and representative terms against the previous 40-minute report. Do not present the offline no-drop result as live latency.

- [ ] **Step 4: Review the diff and commit the scoped fix**

\`\`\`powershell
git diff --check
git status --short
git add src/classbot/asr/funasr_streaming.py src/classbot/main.py src/classbot/llm/deepseek_client.py tests/test_funasr_streaming.py tests/test_main.py tests/test_deepseek_client.py README.md docs/superpowers/plans/2026-09-29-live-asr-latency.md
git commit -m "fix: keep live ASR responsive across segments"
\`\`\`

