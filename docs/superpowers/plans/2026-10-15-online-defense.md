# Online Defense Assistant Implementation Plan

> **For agentic workers:** Use subagent-driven-development for independent audio,
> knowledge, and interface components; root owns integration and review. Preserve
> existing uncommitted changes. No publication or bulk staging is authorized.

**Goal:** Answer live online defense questions from computer playback using the
selected scenario's local code and documents, with local semantic retrieval.

**Architecture:** Reuse the existing capture/ASR/detection/streaming lifecycle.
Inject an immutable scenario/index into a defense answer client and detector.
Keep audio, indexing, and interface changes in separate owned files.

**Tech Stack:** PySide6/qasync, PyAudioWPatch/SoXR, Transformers/PyTorch local BGE,
NumPy, python-docx, pypdf, python-pptx, existing DeepSeek/httpx SSE.

## File ownership and contracts

- Audio: `audio/loopback.py`, `tests/audio/test_loopback.py`.
- Knowledge: `knowledge/` package, `scenarios.py`, corresponding new tests.
- UI: `ui/main_window.py`, optional `ui/scenario_dialog.py`, UI tests.
- Integration: `main.py`, `config.py`, `domain.py`, `llm/deepseek_client.py`,
  `defense_detector.py`, `pyproject.toml`, launchers, README and integration tests.
- `ScenarioProfile(scenario_id, name, kind, library_path)` is immutable; kind is
  `group_meeting`, `competition`, or `defense`; `ScenarioStore.load/save` persists
  profiles under `.cache/classbot`.
- `KnowledgeIndex.build(root, cache_dir, model_name=..., embedder=None,
  progress=None)` returns an immutable searchable snapshot. `search(query,
  limit=5)` yields hits exposing `source`, `text`, `path`, `locator`, `score`.
  `report` exposes indexed/skipped/issues; model failures are raised.
- `LoopbackCapture(device=None)` implements `start/frames/stop` with the existing
  640-byte frames. `list_loopback_devices()` returns playback choices.
- Window emits `scenario_changed(str)`, `scenario_add_requested()`,
  `scenario_edit_requested()`, `index_requested()`, and audio selection changes;
  root owns dialogs/persistence and background work. Existing constructor/tests
  remain compatible.

## Task 1: Playback capture

- [x] Add failing tests with a fake PyAudioWPatch backend and real SoXR: no
  microphone fallback, valid endpoint selection, 44.1/48 kHz stereo conversion,
  exact 640-byte frames, late callbacks ignored, stop wakes consumers.
- [x] Run `.venv/Scripts/python -m pytest tests/audio/test_loopback.py -q` and confirm
  failures identify the missing backend.
- [x] Implement bounded callback capture, continuous resampling and cleanup;
  enumerate loopback endpoints without opening a recording stream.
- [x] Rerun targeted tests and inspect source/error handling.

## Task 2: Scenario folders and hybrid semantic retrieval

- [x] Add failing tests using temporary code/DOCX/PDF/PPTX files and a deterministic
  embedding double. Prove real locators, semantic-only retrieval, source-folder
  isolation, exclusions, parse failures and fingerprint cache invalidation.
- [x] Implement immutable profiles and atomic JSON persistence; safe folder
  traversal/parsers; normalized BGE embeddings with query instructions and
  persisted bounded chunks/vectors; hybrid ranking with provenance.
- [x] Run targeted tests, then exercise real BGE on two synthetic documents.
- [x] Check failed refresh does not replace a valid snapshot/cache.

## Task 3: Interface

- [x] Add failing Qt tests for scenario/device controls and immutable active
  session configuration, source displays, and old-card labels.
- [x] Implement named scenario selection, create/edit action, folder/index
  status with parse summary, playback endpoint selector, refresh button and
  ready/start status. Keep existing semantic cards and scrolling behavior.
- [x] Run UI tests offscreen and render the ready/answered states for inspection.

## Task 4: Runtime, detector and model integration

- [x] Add failing tests for defense requests/multiple subquestions and context;
  source excerpts in the prompt; no cross-scenario references; stop-before-switch
  and disabled startup controls; classroom compatibility.
- [x] Add settings for mode/model/cache; wire background indexing and scenario
  store into UI; inject playback capture, defense detector and knowledge-backed
  answer client into each runtime. Retrieve in `asyncio.to_thread`.
- [x] Define defense prompt as speakable answer + evidence + uncertainty, with
  explicit separation of facts and inference. Add deterministic source listings.
- [x] Update dependencies, launcher checks, `.env.example` and README with setup,
  privacy, refresh, supported formats and endpoint limitations.

## Task 5: Review and delivery

- [x] Review spec coverage, then review implementation quality independently.
- [x] Run full pytest, Ruff, `git diff --check`, dependency/launcher checks.
- [x] Observe real model inference and loopback startup without retaining audio;
  distinguish these from an unobserved DingTalk/Tencent Meeting session.
- [x] Report launch instructions, supported behavior and actual verification.

No commits that mix prior dirty files with this feature; any later publication
requires an explicit allowlist and review of staged contents.

