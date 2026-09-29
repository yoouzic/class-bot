# Live Question Timestamps and Context Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show each newly detected classroom question with a session-relative timestamp while applying conservative context filtering and vocative cleanup before an answer request.

**Architecture:** Keep every question from a new speech segment; only revisions inside the same ASR segment reuse a card. `LivePipeline` assigns the timestamp when a candidate first becomes active, `QuestionDetector` filters obvious lecture scaffolding and cleans explicit vocative prefixes, and `MainWindow` renders a dedicated current-question label plus timestamped history cards.

**Tech Stack:** Python 3.11, PySide6, asyncio, pytest/pytest-asyncio, existing ASR and question-detector domain objects.

---

### Task 1: Timestamped domain and pipeline cards

**Files:**
- Modify: `src/classbot/domain.py`
- Modify: `src/classbot/pipeline.py`
- Test: `tests/test_domain.py`
- Test: `tests/test_pipeline.py`

- [x] **Step 1: Write failing tests** for non-negative `AnswerCard.timestamp_ms` and a pipeline-created card carrying a session-relative timestamp.
- [x] **Step 2: Run the focused tests** with `python -m pytest -q tests/test_domain.py tests/test_pipeline.py`; confirm the new timestamp assertions fail because the field is absent.
- [x] **Step 3: Add `timestamp_ms: int = 0` to `AnswerCard`, validate it is non-negative, start a monotonic session clock in `LivePipeline._run_owner`, and set the timestamp only when a new question card is created.
- [x] **Step 4: Run the focused tests** and confirm they pass while revision events retain the original timestamp.

### Task 2: Conservative context filtering and vocative cleanup

**Files:**
- Modify: `src/classbot/question_detector.py`
- Test: `tests/test_question_detector.py`

- [x] **Step 1: Write failing tests** for removing explicit prefixes such as `各位同学，` and suppressing long lecture-scaffolding text such as `现在打开教材看一下什么是运动什么是静止`, while preserving `我们来看看，哲学的基本问题是什么？`.
- [x] **Step 2: Run the focused detector tests** and confirm they fail with the current normalized text and marker-only scoring.
- [x] **Step 3: Add a conservative `clean_question_text` path for explicit vocatives, add question-ending recognition, and reject only long narrative scaffolding without a question ending/mark. Do not deduplicate across segment IDs.
- [x] **Step 4: Run all detector tests** and confirm existing partial/final revision behavior remains unchanged.

### Task 3: Current-question label and timestamped card rendering

**Files:**
- Modify: `src/classbot/ui/main_window.py`
- Test: `tests/test_main_window.py`

- [x] **Step 1: Write failing UI tests** for the initial current-question placeholder, timestamp rendering such as `[00:01:05]`, and updating the current label only when a new card or the current card changes.
- [x] **Step 2: Run the focused UI tests** and confirm they fail because the label and timestamp are not rendered.
- [x] **Step 3: Add `current_question_label`, a safe elapsed-time formatter, timestamp output in cards, and latest-card tracking without changing the bounded history behavior.
- [x] **Step 4: Run all UI tests** and confirm the existing close, status, card-update, and eviction behavior remains green.

### Task 4: Integration verification and delivery

**Files:**
- Modify: `README.md` only if the timestamp/filter behavior needs user-facing documentation.

- [x] **Step 1: Run `python -m pytest -q`, `python -m ruff check src tests`, and `python -m compileall src`.
- [x] **Step 2: Run a real API smoke test using the local `.env` without printing the key; verify a current question returns a concise answer.
- [ ] **Step 3: Inspect `git diff --check`, verify the worktree state, commit the implementation, and push the existing PR branch.
