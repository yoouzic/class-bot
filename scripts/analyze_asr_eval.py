"""Summarize an eval_asr_variants run: agreement, wrong-form terms, samples.

Usage:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/analyze_asr_eval.py \
        --dir tmp/eval-asr-20261101 [--course marxism_basic_principles_2023]

All variants share one segment list, so line i of every transcript belongs to
the same audio. Without a human reference this measures agreement and known
error-form counts, not accuracy.
"""

from __future__ import annotations

import argparse
import sys
from difflib import SequenceMatcher
from pathlib import Path

VARIANTS = ("streaming", "refine", "offline")


def load_lines(path: Path) -> list[str]:
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            stripped = stripped.partition("]")[-1].strip()
        texts.append(stripped)
    return texts


def similarity(left: str, right: str) -> float:
    # Punctuation differs by design (offline adds it); compare content only.
    left = strip_punctuation(left)
    right = strip_punctuation(right)
    if not left and not right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def strip_punctuation(text: str) -> str:
    return "".join(
        char for char in text if char.isalnum() or "\u4e00" <= char <= "\u9fff"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--course", default="marxism_basic_principles_2023")
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()

    from classbot.course import load_course_profile

    directory = Path(args.dir)
    transcripts = {
        name: load_lines(directory / f"transcript-{name}.txt")
        for name in VARIANTS
    }
    lengths = {name: len(texts) for name, texts in transcripts.items()}
    if len(set(lengths.values())) != 1:
        print(f"variant line counts differ, analysis may be misaligned: {lengths}")
    count = min(lengths.values())
    print(f"segments compared: {count}")

    corpus = {
        name: "".join(texts[:count]) for name, texts in transcripts.items()
    }
    print("\n== characters (raw, incl. punctuation) ==")
    for name, text in corpus.items():
        print(f"  {name:>9}: {len(text)}")

    print("\n== pairwise mean similarity per segment ==")
    for left in VARIANTS:
        for right in VARIANTS:
            if left >= right:
                continue
            scores = [
                similarity(a, b)
                for a, b in zip(
                    transcripts[left][:count],
                    transcripts[right][:count],
                    strict=True,
                )
            ]
            print(f"  {left:>9} vs {right:<9}: {sum(scores) / len(scores):.3f}")

    profile = load_course_profile(args.course)
    corrections = profile.corrections or {}
    hotwords = profile.hotwords
    if corrections:
        # Transcripts were saved after applying corrections, so wrong-form
        # counting here would always read zero; kept for reference only.
        print("\n== known wrong forms (post-correction; expected 0) ==")
        for name, text in corpus.items():
            total = sum(text.count(wrong) for wrong in corrections)
            print(f"  {name:>9}: {total}")

    print("\n== empty segments (speech recognized as nothing) ==")
    for name, texts in transcripts.items():
        empty = sum(1 for text in texts[:count] if not text.strip())
        print(f"  {name:>9}: {empty} / {count}")

    if hotwords:
        print("\n== hotword occurrence totals ==")
        counts = {
            name: sum(text.count(word) for word in hotwords)
            for name, text in corpus.items()
        }
        for name, total in counts.items():
            print(f"  {name:>9}: {total}")

    print(f"\n== {args.samples} longest segments: streaming vs offline ==")
    ranked = sorted(range(count), key=lambda i: -len(transcripts["offline"][i]))
    for index in ranked[: args.samples]:
        print(f"\n--- segment {index} ---")
        for name in VARIANTS:
            text = transcripts[name][index]
            print(f"  {name:>9}: {text[:160]}{'…' if len(text) > 160 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
