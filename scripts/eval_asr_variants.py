"""Compare ASR variants on a real classroom recording.

Usage (project root):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/eval_asr_variants.py \
        --audio "9月16日 马克思主义哲学与辩证法课程讨论.m4a" \
        --course marxism_basic_principles_2023 \
        --out tmp/eval-asr-20261101 [--device cuda] [--max-seconds 180]

Decodes the recording to 16 kHz mono PCM once (cached under --out), segments
it with the production SpeechSegmenter settings, then transcribes every
segment with three variants of the current stack:

  streaming    chunked streaming decode, the live partial path
  refine       streaming full-segment re-decode, the current final refinement
  offline      SeACo-Paraformer + ct-punc, the new refinement

Each variant writes one line per segment ("[start_s end_s] text") plus timing
JSON. CER against an optional --reference transcript is printed at the end;
synthetic comparisons without a human reference measure agreement, not
accuracy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np


def decode_to_pcm(audio_path: Path, out_pcm: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(audio_path),
            "-ac", "1", "-ar", "16000", "-f", "s16le", str(out_pcm),
        ],
        check=True,
    )


def collect_segments(
    samples: np.ndarray,
    *,
    frame_samples: int,
    end_silence_ms: int = 320,
    max_segment_ms: int = 60_000,
    pre_roll_frames: int = 10,
) -> list[tuple[int, np.ndarray]]:
    """Mirror production _audio_loop framing: pre-roll plus in-speech frames."""
    from classbot.audio.segmenter import SpeechSegmenter

    segmenter = SpeechSegmenter(
        end_silence_ms=end_silence_ms, max_segment_ms=max_segment_ms,
    )
    pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
    buffer: list[np.ndarray] = []
    buffer_start = 0
    in_speech = False
    segments: list[tuple[int, np.ndarray]] = []

    for index in range(0, samples.size - frame_samples + 1, frame_samples):
        frame = samples[index : index + frame_samples]
        pre_roll.append(frame)
        event = segmenter.accept(frame.tobytes())
        if event.speech_started:
            in_speech = True
            if not event.continued:
                buffer = list(pre_roll)[:-1]
                buffer_start = index + 1 - len(buffer)
            elif not buffer:
                # a continued split resumes mid-speech without pre-roll
                buffer_start = index
        if in_speech:
            buffer.append(frame)
        if event.speech_ended:
            if buffer:
                segments.append((buffer_start, np.concatenate(buffer)))
            buffer = []
            in_speech = False
    if buffer:
        segments.append((buffer_start, np.concatenate(buffer)))
    return segments


def _texts(result: object) -> list[str]:
    from classbot.asr.funasr_streaming import _texts_from_result

    return [text for text in _texts_from_result(result) if text.strip()]


def _streaming_kwargs(
    hotword: str | None, corrections: dict[str, str], lookback: int,
):
    kwargs: dict[str, object] = {
        "chunk_size": [0, 10, 5],
        "encoder_chunk_look_back": lookback,
        "decoder_chunk_look_back": 1,
    }
    if hotword:
        kwargs["hotword"] = hotword
    if corrections:
        kwargs["postprocess_hotwords"] = dict(corrections)
        kwargs["postprocess_hotword_fuzzy"] = False
    return kwargs


def apply_corrections(text: str, corrections: dict[str, str]) -> str:
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    return text


def transcribe_streaming(
    model, segments, *, chunk_samples: int, hotword: str | None,
    corrections: dict[str, str], lookback: int,
) -> list[str]:
    kwargs = _streaming_kwargs(hotword, corrections, lookback)
    texts: list[str] = []
    for _start, seg in segments:
        parts: list[str] = []
        cache: dict = {}
        full = (seg.size // chunk_samples) * chunk_samples
        for offset in range(0, full, chunk_samples):
            result = model.generate(
                input=seg[offset : offset + chunk_samples],
                cache=cache, is_final=False, **kwargs,
            )
            parts.extend(_texts(result))
        if seg.size > full:
            result = model.generate(
                input=seg[full:], cache=cache, is_final=True, **kwargs,
            )
            parts.extend(_texts(result))
        texts.append(apply_corrections("".join(parts).strip(), corrections))
    return texts


def transcribe_refine(
    model, segments, *, hotword: str | None, corrections: dict[str, str],
    lookback: int,
) -> list[str]:
    kwargs = _streaming_kwargs(hotword, corrections, lookback)
    texts: list[str] = []
    for _start, seg in segments:
        result = model.generate(
            input=seg, cache={}, is_final=True, **kwargs,
        )
        texts.append(apply_corrections("".join(_texts(result)).strip(), corrections))
    return texts


def transcribe_offline(
    refiner, segments, *, corrections: dict[str, str],
) -> list[str]:
    return [
        apply_corrections(refiner.refine(seg), corrections)
        for _start, seg in segments
    ]


def strip_punctuation(text: str) -> str:
    keep = "".join(
        char for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )
    return keep


def edit_distance(left: str, right: str) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float | None:
    normalized_reference = strip_punctuation(reference)
    normalized_hypothesis = strip_punctuation(hypothesis)
    if not normalized_reference:
        return None
    return edit_distance(normalized_reference, normalized_hypothesis) / len(
        normalized_reference
    )


def load_reference_text(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        # tolerate "[12.3 45.6] text" segment lines from earlier runs
        if stripped.startswith("["):
            stripped = stripped.partition("]")[-1].strip()
        if stripped:
            lines.append(stripped)
    return "".join(lines)


def write_transcript(
    path: Path, segments: list[tuple[int, np.ndarray]], texts: list[str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for (start, seg), text in zip(segments, texts, strict=True):
            begin = start / 16000
            end = begin + seg.size / 16000
            handle.write(f"[{begin:07.1f} {end:07.1f}] {text}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--course", default="marxism_basic_principles_2023")
    parser.add_argument("--out", default="tmp/eval-asr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--reference", default="")
    parser.add_argument("--lookback", type=int, default=4)
    parser.add_argument("--variants", default="streaming,refine,offline")
    args = parser.parse_args()

    selected = {name.strip() for name in args.variants.split(",") if name.strip()}
    unknown = selected - {"streaming", "refine", "offline"}
    if unknown:
        parser.error(f"unknown variants: {sorted(unknown)}")

    from classbot.asr import funasr_streaming
    from classbot.asr.offline_refiner import OfflineRefiner
    from classbot.course import load_course_profile

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_course_profile(args.course)
    hotword = " ".join(profile.hotwords) if profile.hotwords else None
    corrections = profile.corrections or {}

    pcm_path = out_dir / "audio-16k.pcm"
    if not pcm_path.is_file():
        print(f"decoding {args.audio} -> {pcm_path}")
        started = time.perf_counter()
        decode_to_pcm(Path(args.audio), pcm_path)
        print(f"decoded in {time.perf_counter() - started:.0f}s")
    raw = np.fromfile(pcm_path, dtype="<i2")
    if args.max_seconds > 0:
        raw = raw[: int(args.max_seconds * 16000)]
    print(f"audio: {raw.size / 16000:.0f}s")

    print("collecting speech segments with production segmenter settings")
    started = time.perf_counter()
    segments = collect_segments(raw, frame_samples=320)
    segments = [
        (start, seg.astype(np.float32) / 32768.0) for start, seg in segments
    ]
    speech_seconds = sum(seg.size for _s, seg in segments) / 16000
    print(
        f"{len(segments)} segments, {speech_seconds:.0f}s speech, "
        f"segmentation took {time.perf_counter() - started:.0f}s"
    )
    (out_dir / "segments.json").write_text(
        json.dumps(
            [
                {"start_s": round(start / 16000, 2),
                 "end_s": round((start + seg.size) / 16000, 2)}
                for start, seg in segments
            ],
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )

    timings: dict[str, object] = {}
    streaming_model = None
    if "streaming" in selected or "refine" in selected:
        started = time.perf_counter()
        streaming_model = funasr_streaming.build_model(args.device)
        timings["streaming_load_s"] = round(time.perf_counter() - started, 1)

    for name, runner in (
        ("streaming", lambda: transcribe_streaming(
            streaming_model, segments, chunk_samples=9600,
            hotword=hotword, corrections=corrections, lookback=args.lookback,
        )),
        ("refine", lambda: transcribe_refine(
            streaming_model, segments, hotword=hotword, corrections=corrections,
            lookback=args.lookback,
        )),
    ):
        if name not in selected:
            continue
        print(f"variant: {name} (lookback {args.lookback})")
        started = time.perf_counter()
        texts = runner()
        elapsed = time.perf_counter() - started
        timings[f"{name}_s"] = round(elapsed, 1)
        timings[f"{name}_rtf"] = round(elapsed / max(speech_seconds, 1), 3)
        write_transcript(out_dir / f"transcript-{name}.txt", segments, texts)
        print(f"  {len(texts)} segments in {elapsed:.0f}s (rtf {timings[f'{name}_rtf']})")

    streaming_model = None  # release before the offline model loads

    if "offline" in selected:
        print("variant: offline (SeACo + ct-punc)")
        refiner = OfflineRefiner(device=args.device, hotwords=profile.hotwords)
        started = time.perf_counter()
        refiner.load()
        timings["offline_load_s"] = round(time.perf_counter() - started, 1)
        started = time.perf_counter()
        texts = transcribe_offline(refiner, segments, corrections=corrections)
        elapsed = time.perf_counter() - started
        timings["offline_s"] = round(elapsed, 1)
        timings["offline_rtf"] = round(elapsed / max(speech_seconds, 1), 3)
        write_transcript(out_dir / "transcript-offline.txt", segments, texts)
        print(f"  {len(texts)} segments in {elapsed:.0f}s (rtf {timings['offline_rtf']})")

    if args.reference:
        reference = load_reference_text(Path(args.reference))
        for name in sorted(selected):
            path = out_dir / f"transcript-{name}.txt"
            if not path.is_file():
                continue
            hypothesis = load_reference_text(path)
            score = cer(reference, hypothesis)
            print(f"CER vs reference [{name}]: {score:.4f}" if score is not None
                  else f"CER vs reference [{name}]: reference empty")
            timings[f"cer_{name}"] = score

    (out_dir / "timing.json").write_text(
        json.dumps(timings, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
