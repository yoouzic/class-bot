"""Pinned local BGE inference; only explicit build preparation may download."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np

from .atomic import replace_atomic
from .cache_manager import cache_available_bytes, require_cache_space

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
EMBEDDING_ENCODING_VERSION = "cls-window512-stride64-unit-pool-v2"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
_MODEL_MARKER = "classbot-snapshot.json"
_MAX_MODEL_METADATA = 4 * 1024 * 1024
_TOKENIZER_ASSETS = frozenset({
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt", "tokenizer.model",
    "spiece.model", "sentencepiece.bpe.model",
})


def _plain_model_path(root: Path, name: str) -> Path:
    """Managed downloads never follow links or accept repository path escapes."""
    parts = PurePosixPath(name).parts
    if not parts or any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part)
        or part.endswith(".")
        or part.split(".", 1)[0].upper() in {
            "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        for part in parts
    ) or "\\" in name or str(PurePosixPath(name)) != name:
        raise ValueError("invalid managed model asset path")
    target = root.joinpath(*parts)
    # Check the models directory as well as every descendant before any write.
    for candidate in (root.parent, root, *(root.joinpath(*parts[:i]) for i in range(1, len(parts) + 1))):
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        ) or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1):
            raise ValueError("managed model asset path contains a link")
    return target


def _model_asset(sibling) -> dict:
    name, size = sibling.rfilename, sibling.size
    if not isinstance(name, str) or type(size) is not int or size < 0:
        raise ValueError("model asset metadata needs a bounded size")
    lfs = getattr(sibling, "lfs", None)
    sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    blob = getattr(sibling, "blob_id", None)
    if isinstance(sha256, str) and re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
        algorithm, digest = "sha256", sha256.lower()
    elif isinstance(blob, str) and re.fullmatch(r"[a-fA-F0-9]{40}", blob):
        algorithm, digest = "git-sha1", blob.lower()
    else:
        raise ValueError("model asset metadata needs an immutable digest")
    return {"name": name, "size": size, "algorithm": algorithm, "digest": digest}


def _model_hasher(asset: dict):
    if asset["algorithm"] == "git-sha1":
        hasher = hashlib.sha1()  # Git object identity, not a security primitive.
        hasher.update(f"blob {asset['size']}\0".encode("ascii"))
        return hasher
    return hashlib.sha256()


def _valid_model_asset(path: Path, asset: dict) -> bool:
    if not path.is_file() or path.stat().st_size != asset["size"]:
        return False
    hasher = _model_hasher(asset)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == asset["digest"]


def _remote_model_chunks(client, url: str, asset: dict):
    """Reject excess response bytes before callers can put them on disk."""
    received = 0
    hasher = _model_hasher(asset)
    with client.stream("GET", url) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes(chunk_size=64 * 1024):
            received += len(chunk)
            if received > asset["size"]:
                raise ValueError("downloaded model asset size exceeds manifest")
            hasher.update(chunk)
            yield chunk
    if received != asset["size"]:
        raise ValueError("downloaded model asset size differs from manifest")
    if hasher.hexdigest() != asset["digest"]:
        raise ValueError("downloaded model asset digest differs from manifest")


def _write_model_bytes(path: Path, chunks, size: int) -> None:
    # The active cache mutation lock serializes writers. Reserve the entire new
    # file while retaining any old file until the atomic replacement succeeds.
    require_cache_space(path, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".partial", delete=False,
        ) as output:
            temporary = Path(output.name)
            received = 0
            for chunk in chunks:
                received += len(chunk)
                if received > size:
                    raise ValueError("managed model asset size exceeds reservation")
                output.write(chunk)
            if received != size:
                raise ValueError("managed model asset size differs from reservation")
            output.flush()
            os.fsync(output.fileno())
        replace_atomic(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def model_spec(model_name: str) -> tuple[str, str | None]:
    if Path(model_name).is_dir():
        return str(Path(model_name).resolve()), None
    if model_name == DEFAULT_MODEL:
        return model_name, DEFAULT_MODEL_REVISION
    identifier, separator, revision = model_name.rpartition("@")
    if separator and identifier and re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        return identifier, revision.lower()
    raise ValueError("hosted models require an immutable commit: use repository@40-character-SHA or a local model folder")


class LocalEmbedder:
    def __init__(
        self, model_name: str = DEFAULT_MODEL, *, cache_dir: Path,
        device: str = "cpu",
        progress: Callable[[str], None] | None = None,
    ):
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device):
            raise ValueError("embedding device must be cpu, cuda, or cuda:N")
        self.model_name = model_name
        self.device = device
        self._model_id, self.revision = model_spec(model_name)
        self.cache_dir = Path(cache_dir)
        self._progress = progress or (lambda _message: None)
        self._lock = threading.Lock()
        self.retrieval_gate = asyncio.Semaphore(1)
        self._model = None
        self._tokenizer = None

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        with self._lock:
            import torch
            # Query-time cache misses must fail visibly instead of downloading.
            self._load(local_files_only=True)
            values = [QUERY_INSTRUCTION + text if query else text for text in texts]
            batches = []
            with torch.inference_mode():
                for start in range(0, len(values), 16):
                    tokens = self._window_tokens(values[start:start + 16])
                    tokens = {key: value.to(self.device) for key, value in tokens.items()}
                    mapping = tokens.pop("overflow_to_sample_mapping")
                    pooled = None
                    for window_start in range(0, len(mapping), 16):
                        window_end = window_start + 16
                        window_tokens = {
                            key: value[window_start:window_end] for key, value in tokens.items()
                        }
                        output = self._model(**window_tokens).last_hidden_state[:, 0]
                        output = torch.nn.functional.normalize(output, p=2, dim=1)
                        if pooled is None:
                            pooled = torch.zeros(
                                (len(values[start:start + 16]), output.shape[1]), dtype=output.dtype,
                                device=self.device,
                            )
                        pooled.index_add_(0, mapping[window_start:window_end], output)
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    batches.append(pooled.cpu().numpy().astype(np.float32))
            return np.concatenate(batches, axis=0)

    def _window_tokens(self, values: list[str]):
        """Slice complete token sequences; backend overflow can silently lose tails."""
        import torch

        tokenizer = self._tokenizer
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if not callable(getattr(tokenizer, "pad", None)) or (
            backend is None and not callable(getattr(tokenizer, "prepare_for_model", None))
        ):
            raise ValueError("embedding tokenizer needs a fast backend or prepare_for_model and pad")
        capacity = 512 - tokenizer.num_special_tokens_to_add(pair=False)
        if capacity <= 64:
            raise ValueError("embedding tokenizer leaves insufficient space for token windows")
        features, mapping = [], []
        type_ids = "token_type_ids" in tokenizer.model_input_names
        if backend is None:
            sequences = tokenizer(
                values, add_special_tokens=False, truncation=False, padding=False,
                return_attention_mask=False, return_token_type_ids=False,
            )["input_ids"]
            for sample, ids in enumerate(sequences):
                for start in range(0, max(1, len(ids)), capacity - 64):
                    end = min(start + capacity, len(ids))
                    features.append(tokenizer.prepare_for_model(
                        ids[start:end], add_special_tokens=True, padding=False,
                        truncation=False, return_attention_mask=True,
                        return_token_type_ids=type_ids,
                    ))
                    mapping.append(sample)
                    if end == len(ids):
                        break
        else:
            truncation, padding = backend.truncation, backend.padding
            try:
                backend.no_truncation()
                backend.no_padding()
                for sample, sequence in enumerate(backend.encode_batch(values, add_special_tokens=False)):
                    size = len(sequence.ids)
                    for start in range(0, max(1, size), capacity - 64):
                        end = min(start + capacity, size)
                        window = copy.deepcopy(sequence)
                        if end < size:
                            window.truncate(end, direction="right")
                        if start:
                            window.truncate(end - start, direction="left")
                        prepared = backend.post_process(window, add_special_tokens=True)
                        item = {"input_ids": prepared.ids, "attention_mask": prepared.attention_mask}
                        if type_ids:
                            item["token_type_ids"] = prepared.type_ids
                        features.append(item)
                        mapping.append(sample)
                        if end == size:
                            break
            finally:
                if truncation is None:
                    backend.no_truncation()
                else:
                    backend.enable_truncation(**truncation)
                if padding is None:
                    backend.no_padding()
                else:
                    backend.enable_padding(**padding)
        if any(not item["input_ids"] or len(item["input_ids"]) > 512 for item in features):
            raise ValueError("embedding tokenizer produced an invalid token window")
        tokens = tokenizer.pad(features, padding=True, return_tensors="pt")
        tokens["overflow_to_sample_mapping"] = torch.tensor(mapping, dtype=torch.long)
        return tokens

    def prepare(self, *, progress: Callable[[str], None] | None = None) -> None:
        """Build-worker-only model preparation, including a possible first download."""
        with self._lock:
            previous = self._progress
            try:
                if progress is not None:
                    self._progress = progress
                if cache_available_bytes(self.cache_dir) is None:
                    self._load(local_files_only=False)
                else:
                    # The Hub/Xet downloader may create unbounded temporary
                    # files. Under a total quota, only our bounded writer may
                    # download; existing Hub snapshots remain readable.
                    try:
                        self._load(local_files_only=True)
                    except OSError:
                        if self.revision is None:
                            raise
                        self._prepare_managed_model()
                        self._load(local_files_only=True)
            finally:
                self._progress = previous

    def _managed_directory(self) -> Path:
        key = hashlib.sha256(f"{self._model_id}@{self.revision}".encode()).hexdigest()
        return self.cache_dir / f"managed-{key[:32]}"

    def _managed_snapshot(self) -> Path | None:
        if self.revision is None:
            return None
        root = self._managed_directory()
        marker = _plain_model_path(root, _MODEL_MARKER)
        if not marker.is_file() or marker.stat().st_size > _MAX_MODEL_METADATA:
            return None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if (
                payload["schema"] != 1 or payload["repository"] != self._model_id
                or payload["revision"] != self.revision or not isinstance(payload["assets"], list)
                or not 0 < len(payload["assets"]) <= 4096
            ):
                return None
            assets = payload["assets"]
            names = set()
            for asset in assets:
                name = asset["name"]
                if (
                    not isinstance(name, str) or name in names or type(asset["size"]) is not int
                    or asset["size"] < 0 or asset["algorithm"] not in {"sha256", "git-sha1"}
                    or not isinstance(asset["digest"], str)
                    or not re.fullmatch(
                        r"[a-f0-9]{64}" if asset["algorithm"] == "sha256" else r"[a-f0-9]{40}",
                        asset["digest"],
                    )
                ):
                    return None
                names.add(name)
                if not _valid_model_asset(_plain_model_path(root, name), asset):
                    return None
            if "config.json" not in names or not names & {"model.safetensors", "model.safetensors.index.json"}:
                return None
            return root
        except (OSError, KeyError, TypeError, ValueError, RecursionError):
            return None

    def _prepare_managed_model(self) -> None:
        """Fetch pinned safe assets with exact file, temporary, and digest bounds.

        Model files are separate from source materials. This intentionally uses
        plain HTTPS rather than a Hub downloader so Xet or global Hub caches
        cannot grow outside the active cache budget.
        """
        import httpx
        from huggingface_hub import HfApi, get_token, hf_hub_url

        root = self._managed_directory()
        _plain_model_path(root, _MODEL_MARKER)
        self._progress("准备下载固定版本本地语义模型；先核对文件大小与缓存总上限。")
        info = HfApi().model_info(self._model_id, revision=self.revision, files_metadata=True)
        if not isinstance(info.sha, str) or info.sha.lower() != self.revision:
            raise ValueError("hosted model manifest revision differs from pinned revision")
        siblings = {}
        for sibling in info.siblings or ():
            name = sibling.rfilename
            if not isinstance(name, str) or name in siblings:
                raise ValueError("invalid duplicate model asset metadata")
            siblings[name] = sibling
        if "config.json" not in siblings:
            raise ValueError("hosted model has no config.json")
        selected = set(siblings) & _TOKENIZER_ASSETS
        prefetched = {}
        token = get_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60.0), headers=headers) as client:
            if "model.safetensors" in siblings:
                selected.add("model.safetensors")
            elif "model.safetensors.index.json" in siblings:
                name = "model.safetensors.index.json"
                index_asset = _model_asset(siblings[name])
                if index_asset["size"] > _MAX_MODEL_METADATA:
                    raise ValueError("model shard index metadata size exceeds limit")
                # Read only this small manifest in memory to discover the exact
                # referenced shards before reserving or creating disk files.
                raw = b"".join(_remote_model_chunks(
                    client, hf_hub_url(self._model_id, name, revision=self.revision), index_asset,
                ))
                index = json.loads(raw)
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError("model shard index needs a nonempty weight map")
                if any(not isinstance(value, str) for value in weight_map.values()):
                    raise ValueError("invalid model shard path")
                shards = set(weight_map.values())
                if len(shards) > 4090:
                    raise ValueError("model shard index exceeds supported asset count")
                for shard in shards:
                    if not isinstance(shard, str) or not shard.endswith(".safetensors"):
                        raise ValueError("invalid model shard path")
                    _plain_model_path(root, shard)
                    if shard not in siblings:
                        raise ValueError("model shard path is absent from pinned manifest")
                selected.update(shards)
                selected.add(name)
                prefetched[name] = raw
            else:
                raise ValueError("hosted embedding models require safetensors weights")
            assets = [_model_asset(siblings[name]) for name in sorted(selected)]
            marker_payload = {
                "schema": 1, "repository": self._model_id, "revision": self.revision, "assets": assets,
            }
            marker_bytes = json.dumps(marker_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(marker_bytes) > _MAX_MODEL_METADATA:
                raise ValueError("managed model metadata size exceeds limit")
            missing = []
            for asset in assets:
                target = _plain_model_path(root, asset["name"])
                if not _valid_model_asset(target, asset):
                    missing.append(asset)
            # Every asset is complete or absent. A budget failure preserves any
            # previous valid model and prevents publishing a partial snapshot.
            require_cache_space(root, sum(asset["size"] for asset in missing) + len(marker_bytes))
            for asset in missing:
                name = asset["name"]
                self._progress(f"下载本地语义模型：{name}（{asset['size']} 字节）。")
                chunks = (
                    (prefetched[name],) if name in prefetched else _remote_model_chunks(
                        client, hf_hub_url(self._model_id, name, revision=self.revision), asset,
                    )
                )
                _write_model_bytes(_plain_model_path(root, name), chunks, asset["size"])
            _write_model_bytes(_plain_model_path(root, _MODEL_MARKER), (marker_bytes,), len(marker_bytes))

    def _load(self, *, local_files_only: bool) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        self._progress(
            f"加载本地语义模型 {self.model_name}；"
            + ("首次使用可能下载权重。" if not local_files_only else "仅使用已下载的本地权重。")
            + "材料与向量计算保留在本机。"
        )
        managed = self._managed_snapshot()
        model_id = str(managed) if managed is not None else self._model_id
        revision = None if managed is not None else self.revision
        local_files_only = local_files_only or managed is not None
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=str(self.cache_dir), trust_remote_code=False,
            local_files_only=local_files_only, revision=revision,
        )
        model = AutoModel.from_pretrained(
            model_id, cache_dir=str(self.cache_dir), trust_remote_code=False,
            use_safetensors=True, local_files_only=local_files_only, revision=revision,
        ).to(self.device).eval()
        self._tokenizer, self._model = tokenizer, model
