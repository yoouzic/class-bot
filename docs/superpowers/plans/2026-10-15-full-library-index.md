# Full Library Index Implementation Plan

**Goal:** Honor the selected full-library policy, including raw JSON/CSV and
historical results, without the balanced mode's 50,000-chunk omission.

**Architecture:** Persist an explicit per-scenario indexing mode. Full mode
parses every supported source within explicit safety bounds, fails visibly on
coverage limits, stores vectors in a resumable memory-mapped checkpoint, and
publishes only a complete, internally consistent snapshot. Existing balanced
mode remains available and its cache cannot satisfy a full-mode request.

**Tech Stack:** Existing local BGE/PyTorch, NumPy memory maps, atomic JSON
checkpoints, PyMuPDF, PySide6/qasync. Source files remain read-only.

## Contracts

- `ScenarioProfile(..., index_mode='balanced')`; valid modes `balanced/full`.
  Existing saved profiles remain readable. The selected profile uses `full`
  through ordinary configuration; code never branches on project names or paths.
- `KnowledgeIndex.build(..., mode='balanced')`; mode is part of cache identity.
- `parsing.scan(..., mode='balanced')` and `parse(..., mode='balanced')` preserve
  current balanced behavior. Full mode includes nested tmp/temp results and
  increases supported-file safety bounds; does not silently skip a budget hit.
- `EmbeddingCheckpoint(metadata_path, revision, rows)` resumes validated,
  contiguous normalized float32 batches. Metadata commits follow data flushes.
  Source/model/schema/mode changes use a new revision. An interrupted refresh
  never replaces the previous usable index.
- Full vector cache is a read-only `.npy` memory map after publication, with
  validated shape/type/digest and bounded header inspection.

## Tasks

- [x] Add failing tests for full-mode tail files beyond balanced limits, raw data retention,
  compatible scenario persistence, policy-isolated caches and explicit bounds.
- [x] Implement policy-aware scanning/parsing and profile mode selection.
- [x] Add failing tests for interrupted checkpoint resume, corrupt checkpoint
  recovery, incompatible revisions, atomic publication and mmap loading.
- [x] Implement checkpointed batches and exclusive build ownership.
- [x] Wire full mode into runtime, dialog and progress/readiness reporting.
- [x] Apply the authorized selected-profile policy; parse the real folder in full
  mode and measure actual coverage without calling hosted inference.
- [x] Run focused then full tests, Ruff, source-layout checks, real local model
  resume smoke, and document actual build status without premature success.

No simulator or project code in the library will be executed; no source file or
credential is modified. No publication, Git staging, or commit is part of this work.

## Expanded system audit

The large-library failure is a pressure sample for the whole system. The audit
also covers audio overflow and bounded segments, canceled retrieval ownership,
bounded answer concurrency, completion semantics, scene-state isolation,
corrupt-configuration recovery, shared models and LRU index retention. See
`docs/system-audit-2026-10-15.md` for evidence and remaining limitations.
