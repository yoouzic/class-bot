# Classroom Question Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Filter live classroom transcription into answerable textbook/choice questions without sending lecture, management, or isolated confirmation speech to DeepSeek.

**Architecture:** Keep `QuestionDetector.feed()` as the only pipeline-facing interface. Add a local extraction/classification layer inside the detector that finds the last complete question clause, removes safe vocative/name prefixes, rejects management/confirmation/unfinished partials, and uses course hotwords as semantic evidence. Wire the existing course profile hotwords into the detector; do not add another network request or change UI/pipeline/LLM contracts.

**Tech Stack:** Python 3.11, dataclasses, regular expressions, pytest/pytest-asyncio, existing FunASR evaluator and course profiles.

---

## Files and responsibilities

- Modify: `src/classbot/question_detector.py` — local question-clause extraction, classroom-type filtering, safe suffix cleanup, course-hotword support.
- Modify: `src/classbot/main.py` — pass the selected course profile hotwords to the detector factory.
- Modify: `tmp/evaluate_recordings.py` — pass the same hotwords during offline recording evaluation so the report matches the desktop runtime.
- Modify: `tests/detection/test_question_detector.py` — recorded-speech regression tests and existing lifecycle assertions.
- Modify: `tests/core/test_main.py` — detector factory course-profile wiring test.
- Modify: `README.md` — document the answerable-question boundary and conservative filtering.
- Create: `docs/superpowers/specs/2026-09-29-question-detection-design.md` — approved design, already committed in `9334f36`.
- Create: `tmp/eval_marxism_tail_40m_question_filter.json` — ignored offline evidence report; never commit audio, API keys, or generated reports.

### Task 1: Add failing tests from the real recording

**Files:**

- Modify: `tests/detection/test_question_detector.py`

- [ ] **Step 1: Add tests for lecture, management, confirmation, extraction, and partial stability**

Add these tests after the existing question-shape tests:

```python
def test_embedded_explanation_phrase_is_not_a_question():
    detector = QuestionDetector(subject="哲学", hotwords=("世界观",))

    assert detector.feed(
        TranscriptEvent(
            text="其次呢我们又了解了什么是世界观",
            is_final=True,
        )
    ) == []


def test_incomplete_partial_is_waited_for():
    detector = QuestionDetector(subject="哲学", hotwords=("马克思主义",))

    assert detector.feed(
        TranscriptEvent(
            text="第二个重点就是马克思主义是如何",
            is_final=False,
        )
    ) == []


@pytest.mark.parametrize(
    "text",
    [
        "李信同学在吗",
        "声音大一点",
        "答案不就在这里吗",
        "对不对",
    ],
)
def test_management_and_isolated_confirmation_are_filtered(text):
    detector = QuestionDetector(subject="哲学", hotwords=("物质",))

    assert detector.feed(TranscriptEvent(text=text, is_final=True)) == []


def test_long_explanation_returns_only_the_final_academic_question():
    detector = QuestionDetector(subject="哲学", hotwords=("哲学", "世界观"))

    candidates = detector.feed(
        TranscriptEvent(
            text=(
                "哲学是系统化理论化的世界观然后所以诸位"
                "哲学的基本问题是什么呀"
            ),
            is_final=True,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].text == "哲学的基本问题是什么"


def test_called_student_question_removes_only_safe_name_prefix():
    detector = QuestionDetector(subject="哲学", hotwords=("哲学",))

    candidates = detector.feed(
        TranscriptEvent(text="李卓阳哲学的基本问题是什么", is_final=True)
    )

    assert len(candidates) == 1
    assert candidates[0].text == "哲学的基本问题是什么"


def test_choice_question_survives_name_prefix():
    detector = QuestionDetector(subject="哲学")

    candidates = detector.feed(
        TranscriptEvent(text="陈林西第三题选哪个", is_final=True)
    )

    assert len(candidates) == 1
    assert candidates[0].text == "第三题选哪个"


def test_confirmation_with_a_textbook_proposition_is_answerable():
    detector = QuestionDetector(subject="哲学", hotwords=("物质", "意识"))

    candidates = detector.feed(
        TranscriptEvent(
            text="物质和意识的辩证关系对不对",
            is_final=True,
        )
    )

    assert len(candidates) == 1
```

- [ ] **Step 2: Run only the new tests and verify they fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/detection/test_question_detector.py -q
```

Expected: the new tests fail because the current detector treats embedded `什么是`, management/confirmation speech, full lecture prefixes, and unsafe partials as candidates; the existing lifecycle tests should remain green.

### Task 2: Implement local extraction and classification

**Files:**

- Modify: `src/classbot/question_detector.py`

- [ ] **Step 1: Extend the detector constructor without breaking existing callers**

Change the signature to accept optional hotwords while preserving all current call sites:

```python
def __init__(
    self,
    subject: str,
    minimum_length: int = 8,
    hotwords: tuple[str, ...] = (),
) -> None:
    self.subject = subject
    self.minimum_length = minimum_length
    self.hotwords = tuple(word for word in hotwords if word.strip())
```

- [ ] **Step 2: Add recorded-speech constants and helper functions**

Add constants for discourse boundaries, safe topic prefixes, classroom management, confirmation-only phrases, and unfinished partial tails. Add these helpers without changing the public `QuestionCandidate` type:

```python
_QUESTION_BOUNDARY_RE = re.compile(
    r"[。！？?!；;]+|所以|那么|接着(?:呢)?|然后(?:呢)?|"
    r"也就是说|具体来说|总的来说|回顾一下|刚刚已经|"
    r"(?:我们(?:上节课)?(?:大致)?学习了)|(?:我们(?:又)?了解了)"
)
_SAFE_TOPIC_PREFIXES = (
    "哲学", "世界观", "物质", "意识", "唯物", "唯心", "马克思",
    "辩证", "第一性", "运动", "时间", "空间", "方法论",
)
_MANAGEMENT_RE = re.compile(
    r"(?:同学|大家|各位)?(?:在吗|到了吗|声音(?:大|小)(?:一点)?|"
    r"听(?:清|题目)?了?吗|能听到吗|请安静|打开教材|翻到)"
)
_CONFIRMATION_ONLY = ("对不对", "对吧", "是吧", "答案不就在这里吗")
_INCOMPLETE_PARTIAL_TAILS = (
    "是如何", "是怎么", "是怎样", "是为什么", "是哪些", "是哪个",
)
_TRAILING_FILLERS = ("嗯", "呃", "呀", "啊", "吧", "各位", "同学们")


def _strip_trailing_fillers(text: str) -> str:
    for filler in _TRAILING_FILLERS:
        while text.endswith(filler):
            text = text[: -len(filler)]
    return text


def _strip_safe_name_prefix(text: str) -> str:
    match = re.match(
        r"^[\u4e00-\u9fff]{2,4}(?=(?:第?\d+题|"
        + "|".join(map(re.escape, _SAFE_TOPIC_PREFIXES))
        + r"))",
        text,
    )
    if match:
        return text[match.end() :]
    return text
```

`_strip_safe_name_prefix` must only remove a prefix immediately followed by a known topic or question number; never remove arbitrary leading Chinese characters.

- [ ] **Step 3: Add the question-span extractor**

Implement `_extract_question_text(raw_text: str) -> str` with this order:

1. strip explicit vocatives and safe name prefixes;
2. split on `_QUESTION_BOUNDARY_RE` and inspect clauses from right to left;
3. strip trailing fillers from each clause;
4. keep a clause when it has a terminal question ending/mark or a standalone question shape such as `为什么说…` or `哲学是什么样的世界观`;
5. reject an embedded `什么是…` clause when the original text has a teaching lead such as `学习了/了解了` and the clause does not itself end as a question;
6. return normalized text with trailing question particles removed.

The function returns `""` when no answerable question span exists. This makes the detector conservative without changing transcript callbacks.

- [ ] **Step 4: Add the answerability gate and use it from `feed()`**

Replace the current context-noise-only gate with a local answerability check:

```python
normalized_text = _extract_question_text(event.text)
confidence = self._score(event, normalized_text)
if not normalized_text or not self._is_answerable(event, normalized_text):
    if event.is_final:
        self._active_questions.pop(segment_id, None)
    return []
```

`_is_answerable` must reject `_MANAGEMENT_RE`, isolated `_CONFIRMATION_ONLY`, and incomplete partial tails. It must accept explicit terminal question forms, choice forms such as `第3题选哪个`, and proposition-plus-confirmation forms when the normalized text contains a course hotword or a built-in academic term. It must continue to enforce `minimum_length` and the existing score threshold.

- [ ] **Step 5: Run the detector tests and fix only behavior covered by the plan**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/detection/test_question_detector.py -q
```

Expected: all detector tests pass, including the new recording-derived cases. Do not loosen the filter to make an unrelated test pass; update the specific extraction or classification rule instead.

### Task 3: Wire course hotwords into runtime and evaluator

**Files:**

- Modify: `src/classbot/main.py`
- Modify: `tmp/evaluate_recordings.py`
- Modify: `tests/core/test_main.py`

- [ ] **Step 1: Add a failing factory wiring test**

Add a test that patches `load_course_profile`, calls `_make_detector`, and asserts the detector receives the profile hotwords:

```python
def test_default_detector_loads_selected_course_hotwords(monkeypatch):
    profile = CourseProfile("demo", hotwords=("物质", "意识"))
    monkeypatch.setenv("CLASSBOT_COURSE", "demo")
    monkeypatch.setattr(main_module, "load_course_profile", lambda _course: profile)

    detector = main_module._make_detector("哲学")

    assert detector.hotwords == ("物质", "意识")
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/core/test_main.py::test_default_detector_loads_selected_course_hotwords -q
```

Expected: FAIL because `_make_detector` currently constructs `QuestionDetector(subject)` without the course profile.

- [ ] **Step 3: Implement the runtime and evaluator wiring**

Update `_make_detector` to load `CLASSBOT_COURSE` and pass `profile.hotwords`. In `tmp/evaluate_recordings.py`, pass `hotwords=profile.hotwords` to `QuestionDetector` so offline reports exercise the same rule path as the desktop app.

- [ ] **Step 4: Run runtime-related tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/core/test_main.py tests/core/test_runtime.py tests/pipeline/test_pipeline.py -q
```

Expected: PASS. Existing dependency-injection factories that accept only `subject` must remain untouched; only the default production factory loads hotwords.

### Task 4: Document and verify the behavior

**Files:**

- Modify: `README.md`

- [ ] **Step 1: Update the runtime behavior paragraph**

Document that the app only creates answer cards for academic/choice questions and keeps management, confirmation, and lecture speech in the transcript stream. State that partial questions are conservative and may wait for a final segment.

- [ ] **Step 2: Run the complete automated checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
git diff --check
.\start_classbot.ps1 -CheckOnly
```

Expected: all tests pass, Ruff reports `All checks passed!`, `git diff --check` is silent, and launcher self-check reports `[Classbot] Check passed`.

- [ ] **Step 3: Run the 40-minute recording evaluation**

Run the existing evaluator against the same tail used for the baseline:

```powershell
.\.venv\Scripts\python.exe tmp/evaluate_recordings.py `
  --file "9月16日 马克思主义哲学与辩证法课程讨论.m4a" `
  --start 5820 `
  --duration 2400 `
  --output tmp/eval_marxism_tail_40m_question_filter.json `
  --course marxism_basic_principles_2023 `
  --subject "马克思主义基本原理" `
  --wait-for-closing
```

Inspect the report fields `question_candidate_count`, `question_candidates`, `final_transcript_count`, `dropped_audio_frames`, and `reader_error`. Manually classify representative candidates into answerable, management, confirmation, lecture, and unfinished partial. Report candidate-quality changes separately from offline processing speed.

- [ ] **Step 4: Commit the implementation and documentation**

After all checks and the recording report are complete, stage only the source/tests/README files and commit:

```powershell
git add src/classbot/question_detector.py src/classbot/main.py `
  tmp/evaluate_recordings.py tests/detection/test_question_detector.py `
  tests/core/test_main.py README.md
git commit -m "feat: filter classroom question candidates"
```

Do not stage `.env`, recordings, PDFs, or files under `tmp/`.

