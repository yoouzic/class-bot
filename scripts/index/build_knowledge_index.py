"""Build a saved scenario's local index in an isolated computation process.

Only this process's Torch thread count is adjusted. Source files are never
executed. Progress is durable in the ordinary KnowledgeIndex checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="saved scenario id; defaults to the first")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    parser.add_argument("--cancel-file", type=Path, help="stop after a phase/batch when this file exists")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--receipt", type=Path, default=Path("tmp/full-index-build-result.json"))
    arguments = parser.parse_args()
    if arguments.threads < 1:
        parser.error("--threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(arguments.threads)
    os.environ["MKL_NUM_THREADS"] = str(arguments.threads)
    if arguments.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    from classbot.config import Settings
    from classbot.knowledge import KnowledgeIndex, LocalEmbedder
    from classbot.knowledge.embedding import EMBEDDING_ENCODING_VERSION
    from classbot.scenarios import ScenarioStore

    torch.set_num_threads(arguments.threads)
    torch.set_num_interop_threads(1)
    if arguments.device.startswith("cuda"):
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this Python environment")
        torch.cuda.get_device_properties(arguments.device)  # Fail before scanning on an invalid index.
        # Keep full float32 semantics when reusing batches computed on CPU.
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    settings = Settings.from_env()
    profiles = ScenarioStore(settings.cache_dir / "scenarios.json").load()
    profile = next((item for item in profiles if item.scenario_id == arguments.scenario), None) if arguments.scenario else next(iter(profiles), None)
    if profile is None:
        parser.error("no matching saved scenario")
    started = time.monotonic()
    last_report = 0.0
    first_vector: tuple[int, float] | None = None
    rows = None

    def progress(message: str) -> None:
        nonlocal last_report, first_vector, rows
        now = time.monotonic()
        match = re.search(r"本地语义向量(?:续建)?\s+(\d+)/(\d+)", message)
        statistics = {}
        if match:
            complete, total = map(int, match.groups())
            rows = (complete, total)
            if first_vector is None:
                first_vector = (complete, now)
            computed, elapsed = complete - first_vector[0], now - first_vector[1]
            if computed and elapsed:
                rate = computed / elapsed
                statistics = {"rows": complete, "total": total, "rows_per_second": round(rate, 2), "eta_seconds": round((total - complete) / rate)}
        if now - last_report < 15 and "完成：" not in message:
            return
        last_report = now
        print(json.dumps({"at": datetime.now(UTC).isoformat(), "elapsed_seconds": round(now - started, 2), "message": message, **statistics}, ensure_ascii=False), flush=True)

    embedder = LocalEmbedder(settings.embedding_model, cache_dir=settings.cache_dir / "models", device=arguments.device)
    print(json.dumps({"event": "start", "scenario_id": profile.scenario_id, "mode": profile.index_mode, "threads": arguments.threads, "device": arguments.device, "offline": arguments.offline}, ensure_ascii=False), flush=True)
    try:
        index = KnowledgeIndex.build(
            Path(profile.library_path), settings.cache_dir, model_name=settings.embedding_model,
            mode=profile.index_mode, progress=progress, embedder=embedder,
            cancelled=(arguments.cancel_file.exists if arguments.cancel_file is not None else None),
        )
    except BaseException as error:
        print(json.dumps({"event": "failed", "type": type(error).__name__, "message": str(error), "last_rows": rows}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise
    receipt = {
        "event": "complete", "scenario_id": profile.scenario_id, "mode": index.mode,
        "root": str(index.root), "revision": index.revision, "built_at": index.built_at,
        "indexed_files": index.report.indexed_files, "chunks": index.report.chunks,
        "limited_files": index.report.limited_files, "skipped_files": index.report.skipped_files,
        "duplicate_files": index.report.duplicate_files,
        "vector_shape": list(index._vectors.shape), "vector_type": type(index._vectors).__name__,
        "elapsed_seconds": time.monotonic() - started, "report_path": str(index.report_path),
        "is_stale": index.is_stale(), "cloud_calls": 0,
        "embedding_device": arguments.device, "embedding_cache": index.embedding_stats,
        "model_parameter_device": str(next(embedder._model.parameters()).device),
        "model_parameter_dtype": str(next(embedder._model.parameters()).dtype),
        "torch_version": torch.__version__, "model_revision": embedder.revision,
        "embedding_encoding": EMBEDDING_ENCODING_VERSION,
        "vectors_file": str(getattr(index._vectors, "filename", "")),
    }
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if receipt["is_stale"] or receipt["limited_files"]:
        raise RuntimeError("index did not pass final full-coverage freshness check")


if __name__ == "__main__":
    main()
