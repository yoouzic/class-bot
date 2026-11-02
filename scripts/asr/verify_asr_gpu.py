"""Verify the ASR runtime: streaming model and offline refiner both load and infer.

Usage (project root):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/asr/verify_asr_gpu.py --device cuda

The offline refinement model downloads once on first use (about 1.4 GB through
ModelScope). Synthetic audio produces meaningless text on purpose: this script
checks that the devices, checkpoints, and inference paths work end to end, not
transcription quality.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def _probe_audio(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    sample_count = int(seconds * 16000)
    t = np.arange(sample_count) / 16000.0
    noise = rng.standard_normal(sample_count).astype(np.float32) * 0.01
    return (noise + 0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=3.0)
    args = parser.parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"FAIL: CUDA unavailable (torch={torch.__version__}, "
            f"build={torch.version.cuda}). Install the cu130 wheels first.",
        )
        return 1
    if args.device.startswith("cuda"):
        print(
            f"torch {torch.__version__} (cuda {torch.version.cuda}) on "
            f"{torch.cuda.get_device_name(0)}"
        )

    audio = _probe_audio(args.seconds)

    from classbot.asr import funasr_streaming
    from classbot.asr.offline_refiner import OfflineRefiner

    started = time.perf_counter()
    streaming = funasr_streaming.build_model(args.device)
    print(f"streaming model loaded in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    streaming.generate(
        input=audio[:9600],
        cache={},
        is_final=False,
        chunk_size=[0, 10, 5],
        encoder_chunk_look_back=4,
        decoder_chunk_look_back=1,
    )
    print(f"streaming chunk inference ok in {(time.perf_counter() - started) * 1000:.0f} ms")

    refiner = OfflineRefiner(device=args.device)
    started = time.perf_counter()
    refiner.load()  # includes the CUDA warm-up inference absorbed here
    print(f"offline refiner loaded and warmed in {time.perf_counter() - started:.1f}s")

    started = time.perf_counter()
    text = refiner.refine(audio)
    print(
        f"offline refine of {args.seconds:.0f}s audio ok in "
        f"{(time.perf_counter() - started) * 1000:.0f} ms -> text={text!r}"
    )

    if args.device.startswith("cuda"):
        print(f"peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MiB")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
