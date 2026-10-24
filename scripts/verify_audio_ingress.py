"""Observe native playback -> production audio loop -> real ASR -> questions.

No cloud client is constructed. The original LivePipeline reads recognizer
events and invokes the actual detector, but answer scheduling is disabled.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from verify_live_defense import (
    ROOT,
    Receipt,
    asr_model_receipt,
    assess_audio_case,
    observe_dependencies,
    play_wav_sync,
    wav_receipt,
)


async def run(args, cases):
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from classbot.asr import funasr_streaming
    from classbot.audio.loopback import LoopbackCapture
    from classbot.audio.segmenter import SpeechSegmenter
    from classbot.config import Settings
    from classbot.defense_detector import DefenseQuestionDetector
    from classbot.main import RuntimeController, RuntimeDependencies
    from classbot.pipeline import LivePipeline

    torch.set_num_threads(args.cpu_threads)
    model_path = args.asr_model.resolve()
    if not (model_path / "model.pt").is_file():
        raise ValueError("The explicit cached streaming ASR model is unavailable")

    def cached_model(device):
        from funasr import AutoModel
        return AutoModel(model=str(model_path), device=device, disable_update=True, check_latest=False)
    if not args.production_model_loader:
        funasr_streaming.build_model = cached_model

    class NoCloudClient:
        async def prepare_answer(self, *_args):
            raise RuntimeError("Cloud is disabled for ingress observation")

        async def stream_answer(self, *_args, **_kwargs):
            raise RuntimeError("Cloud is disabled for ingress observation")
            yield  # pragma: no cover - preserve the async-generator protocol

        async def aclose(self):
            pass

    class IngressPipeline(LivePipeline):
        async def _handle_candidate(self, candidate):
            # _run_owner still consumes real ASR events and calls detector.feed.
            # The dependency observer records every real candidate before here.
            pass

    receipt = Receipt(args.output)
    receipt.record("ingress_session", asr_model=str(model_path), cpu_threads=torch.get_num_threads(),
                   raw_audio_saved=False, cloud_enabled=False,
                   final_refinement=args.final_refinement, hotwords=[], corrections={},
                   loader_override=None if args.production_model_loader else
                   "verification process only: same cached paraformer checkpoint, disable_update=True, check_latest=False",
                   checkpoint=asr_model_receipt(model_path))
    settings = Settings(None, "示例项目答辩", "cloud-disabled", "https://api.deepseek.com", "cpu",
                        audio_source="loopback")
    def recognizer_factory(device):
        recognizer = funasr_streaming.FunAsrStreamingRecognizer(
            device=device, final_refinement=args.final_refinement,
        )
        refine = recognizer._refine_segment

        async def observed_refine():
            before = recognizer._raw_segment_text
            retained_samples = sum(audio.size for audio in recognizer._segment_audio)
            retained_bytes = sum(audio.nbytes for audio in recognizer._segment_audio)
            started = time.monotonic()
            result = await refine()
            receipt.record("refinement", streaming_text_before=before, refined_text=result,
                           seconds_to_refine=round(time.monotonic() - started, 4),
                           retained_samples=retained_samples, retained_bytes=retained_bytes)
            return result
        recognizer._refine_segment = observed_refine
        return recognizer

    dependencies = RuntimeDependencies(
        capture_factory=lambda: LoopbackCapture(None),
        segmenter_factory=lambda: SpeechSegmenter(end_silence_ms=args.end_silence_ms),
        recognizer_factory=recognizer_factory,
        detector_factory=DefenseQuestionDetector,
        client_factory=lambda _settings: NoCloudClient(),
        pipeline_factory=IngressPipeline,
    )
    controller = RuntimeController(settings, receipt, observe_dependencies(dependencies, receipt))
    active = None
    results = []
    try:
        active = await asyncio.wait_for(controller.start(), args.startup_timeout)
        receipt.record("runtime_started", endpoint=asdict(active.capture.active_device),
                       torch_cpu_threads_after_asr=torch.get_num_threads(),
                       asr_ncpu=getattr(active.recognizer._model, "kwargs", {}).get("ncpu"),
                       capture_type=type(active.capture).__name__,
                       segmenter_type=type(active.segmenter).__name__,
                       recognizer_type=type(active.recognizer).__name__,
                       final_refinement=active.recognizer.final_refinement,
                       segment_max_ms=active.segmenter.max_segment_ms,
                       end_silence_ms=active.segmenter.end_silence_ms,
                       detector_type=type(active.detector).__name__)
        for case in cases:
            await asyncio.sleep(0.7)
            receipt.case_id = case["id"]
            wav = args.wav_dir / f"{case['id']}.wav"
            receipt.record("playback_start", intended_text=case["text"], **wav_receipt(wav))
            await asyncio.to_thread(play_wav_sync, wav)
            receipt.record("playback_end")
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                events = [event for event in receipt.events if event["case_id"] == case["id"]]
                result = assess_audio_case(events, case["expected_terms"],
                                           expected_question_groups=case.get("expected_question_groups", []))
                if result["passed"] or active.cleaned:
                    break
                await asyncio.sleep(0.2)
            await asyncio.sleep(0.4)
            events = [event for event in receipt.events if event["case_id"] == case["id"]]
            result = {"case_id": case["id"], "intended_text": case["text"],
                      **assess_audio_case(events, case["expected_terms"],
                                         expected_question_groups=case.get("expected_question_groups", []))}
            actual_finals = " ".join(event["text"] for event in events
                                     if event["kind"] == "transcript" and event.get("is_final"))
            result["actual_final_transcript"] = actual_finals
            result["recognition_term_review"] = {
                "missing_terms": [term for term in case.get("recognition_review_terms", [])
                                  if term.lower() not in actual_finals.lower()],
                "scope": "literal professional-term review; diagnostic only",
            }
            results.append(result)
            receipt.record("case_verdict", result=result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if active.cleaned:
                break
    finally:
        receipt.case_id = None
        await controller.stop()
        if active is not None:
            receipt.record("capture_summary", **active.capture.verification_stats)
            receipt.record("cleanup", cleaned=active.cleaned, capture_released=active.capture._stream is None,
                           pipeline_diagnostics=active.pipeline.diagnostics,
                           retained_segment_samples=sum(audio.size for audio in active.recognizer._segment_audio),
                           inference_tasks_remaining=len(active.recognizer._inference_tasks))
        summary = {"passed": len(results) == len(cases) and all(result["passed"] for result in results),
                   "cloud_enabled": False, "cases": results, "scope": "audio ingress only; no retrieval or answers"}
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-json", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-model", type=Path, required=True)
    parser.add_argument("--production-model-loader", action="store_true",
                        help="Call unmodified production build_model; explicit model path is receipt-only")
    parser.add_argument("--final-refinement", action="store_true",
                        help="Diagnostic-only existing full-segment decode; no hotwords or corrections")
    parser.add_argument("--end-silence-ms", type=int, default=320,
                        help="Explicit diagnostic segmenter pause threshold; maximum stays 60000ms")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=8)
    parser.add_argument("--startup-timeout", type=float, default=180)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cases = json.loads(args.questions_json.read_text(encoding="utf-8-sig"))
    return 0 if asyncio.run(run(args, cases)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
