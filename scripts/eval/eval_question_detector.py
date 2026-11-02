"""Measure classroom or defense question detection against annotated clauses.

Usage (project root):
    PYTHONIOENCODING=utf-8 PYTHONPATH=src .venv/Scripts/python.exe \
        scripts/eval/eval_question_detector.py \
        --sample tmp/eval-asr-20261101/clause-sample.json \
        --labels tmp/eval-asr-20261101/clause-labels.json \
        --detector classroom

`--detector defense` runs DefenseQuestionDetector over the same clauses. The
labels are AI-annotated from a classroom recording, so defense metrics are a
regression reference only, not an acceptance gate. Labels: q = a complete,
answerable question (including teacher self-posed questions and prompts to
students); n = statements, confirmation tails, fragments, noise. They are a
reviewable baseline for regression comparison, not ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from classbot.course import load_course_profile
from classbot.defense_detector import DefenseQuestionDetector
from classbot.domain import TranscriptEvent
from classbot.question_detector import QuestionDetector


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--course", default="marxism_basic_principles_2023")
    parser.add_argument(
        "--detector", choices=("classroom", "defense"), default="classroom"
    )
    args = parser.parse_args()
    print(f"detector: {args.detector}")

    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    if args.detector == "classroom":
        profile = load_course_profile(args.course)
    else:
        profile = None

    tp = fp = fn = tn = 0
    disagreements: list[tuple[str, int, str, str]] = []
    for index, item in enumerate(sample):
        label = labels.get(str(index))
        if label is None:
            continue
        if args.detector == "defense":
            detector = DefenseQuestionDetector("答辩")
        else:
            detector = QuestionDetector(
                "马克思主义基本原理",
                hotwords=profile.hotwords if profile else (),
            )
        event = TranscriptEvent(
            text=item["clause"],
            is_final=True,
            start_ms=0,
            end_ms=1000,
            segment_id=index,
        )
        predicted = bool(detector.feed(event))
        actual = label == "q"
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
            disagreements.append(("FP", index, item["clause"], ""))
        elif not predicted and actual:
            fn += 1
            disagreements.append(("FN", index, item["clause"], ""))
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    print(f"clauses: {tp + fp + fn + tn}  questions: {tp + fn}")
    print(f"precision: {precision:.3f}  recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}  TN: {tn}")
    for kind, index, clause, _extra in disagreements:
        print(f"  {kind} [{index:3d}] {clause[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
