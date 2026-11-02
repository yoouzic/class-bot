# Online defense assistant

Approved direction on 2026-10-15: capture all computer playback, select a local
folder for each meeting/competition scenario, include local semantic vector
retrieval in the first release, and continue using DeepSeek with only the
question and retrieved excerpts. Distinguish material facts from suggestions.

## User flow

Create a named scenario (group meeting, competition, or other defense), choose
its folder, build/refresh its local index, select the playback endpoint actually
used by DingTalk/Tencent Meeting, and start listening. Scenario and audio-device
changes are available while stopped. Starting requires a ready index for the
selected defense scenario. The original classroom mode remains selectable.

## Playback audio

Use Windows WASAPI loopback through PyAudioWPatch, never microphone fallback in
playback mode. Read native channels/rate (including 44.1 kHz Bluetooth audio),
mix to mono, continuously resample with SoXR, and emit the existing 16 kHz PCM16
20 ms frame contract. Capture stays in memory; no raw-audio recording. Device
failures visibly stop capture. A single endpoint captures all its playback,
not every physical output simultaneously. Changing headphones requires restart.

## Local libraries

Read supported code/text, DOCX paragraphs/tables, PDF text pages, and PPTX slides
without executing code or macros. Report skipped/failed/unsupported and scanned
documents. Exclude hidden/credential/dependency/build folders and links outside
the selected root. Apply file/count/size/chunk limits. Original files are never
modified. Legacy DOC and image OCR are outside this release.

Create bounded chunks with real relative file paths and code lines, Word
paragraph/table positions, PDF page sequence, or slide numbers. Cache normalized
local BGE embeddings and content fingerprints under ignored .cache/classbot;
combine semantic similarity with Chinese/identifier keyword ranking. Use
BAAI/bge-small-zh-v1.5 via local Transformers/PyTorch, safe model weights, no
remote code, and allow a local model path. First build may download model
weights; embeddings and source parsing never call a hosted inference service.
No silent downgrade to lexical-only retrieval if the model fails.

Build/refresh and query embedding work run outside Qt's event loop. Cache reuse
must be tied to content, model, parser schema, and root. Report index time and
explicit refresh requirements; an index is a snapshot rather than a live
filesystem monitor. Keep the prior usable index when refresh fails.

## Questions and answers

Add a defense-specific detector for questions and requests such as explaining
design choices, innovation, experiments, limitations, and follow-up questions.
Retain existing stable submission and cancellation behavior. Keep multiple
related subquestions together and use recent transcript context for pronouns.

Answer with a short speakable response first, then supporting evidence and any
limitations/suggestions. Use source IDs assigned by the application; show the
actual retrieved source locations separately. A retrieval hit is a candidate,
not proof of a claim. Never invent results, implementation details, citations,
or evidence. Mark unsupported project facts and model suggestions explicitly.
Treat source content as data, never instructions. A source citation is not a
guarantee that every model claim is supported.

## Session isolation

Bind the selected immutable scenario and index snapshot to each runtime. Block
configuration changes during startup/listening/shutdown; stop before switching.
Cancel old answers on stop. Label old cards with their scenario so histories
remain understandable. Do not change global environment variables to switch.

## Verification

Preserve the dirty working tree. Test 44.1/48 kHz resampling, frame shape,
loopback-only selection, cleanup/errors, source locations, cache invalidation,
semantic-only matches, cross-folder isolation, scenario persistence, prompt
boundaries, defense requests, and UI session locking. Run the complete existing
suite, Ruff, diff checks, a real local-embedding sample, and a real loopback
startup check. Do not claim a real meeting has passed without observing one.

References: Microsoft Loopback Recording; PyAudioWPatch; Python-SoXR;
BAAI/bge-small-zh-v1.5 model card.
