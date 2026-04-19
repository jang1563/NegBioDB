"""Helpers for resolving and caching merged LoRA adapters."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MERGE_METADATA_FILENAME = ".adapter_merge_meta.json"
MERGE_FORMAT_VERSION = 2
_TRACKED_SUFFIXES = {".json", ".safetensors", ".bin"}
_SPECIAL_TOKEN_ID_FIELDS = (
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
    "unk_token_id",
    "sep_token_id",
    "decoder_start_token_id",
)


def resolve_adapter_dir(adapter_path: Path) -> Path:
    """Resolve a run directory to the concrete adapter artifact directory."""

    adapter_path = Path(adapter_path)
    final_dir = adapter_path / "final"
    if (final_dir / "adapter_config.json").exists():
        return final_dir
    return adapter_path


def _file_signature(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def load_tokenizer(model_id: str):
    """Load a tokenizer with safe defaults across supported model families."""

    from transformers import AutoTokenizer

    tokenizer_kwargs = {"trust_remote_code": True}
    try:
        return AutoTokenizer.from_pretrained(
            model_id,
            fix_mistral_regex=True,
            **tokenizer_kwargs,
        )
    except TypeError:
        pass
    try:
        return AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
    except TypeError:
        # _patch_mistral_regex in some transformers versions fails on local paths;
        # slow tokenizer avoids that code path entirely.
        return AutoTokenizer.from_pretrained(model_id, use_fast=False, **tokenizer_kwargs)


def _align_special_token_ids(model, tokenizer) -> None:
    """Persist tokenizer special-token ids into saved model configs."""

    token_id_values = {
        field: getattr(tokenizer, field, None) for field in _SPECIAL_TOKEN_ID_FIELDS
    }
    for config_name in ("config", "generation_config"):
        config = getattr(model, config_name, None)
        if config is None:
            continue
        for field, value in token_id_values.items():
            if hasattr(config, field):
                setattr(config, field, value)


def build_merge_metadata(adapter_path: Path) -> dict[str, object]:
    """Build a stable fingerprint for a merged-adapter cache entry."""

    adapter_dir = resolve_adapter_dir(adapter_path)
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

    with open(config_path) as f:
        adapter_config = json.load(f)

    tracked_files = [
        _file_signature(path)
        for path in sorted(adapter_dir.iterdir())
        if path.is_file() and path.suffix in _TRACKED_SUFFIXES
    ]
    if not tracked_files:
        raise FileNotFoundError(f"No adapter artifacts found in {adapter_dir}")

    return {
        "merge_format_version": MERGE_FORMAT_VERSION,
        "adapter_dir": str(adapter_dir.resolve()),
        "base_model_name_or_path": adapter_config["base_model_name_or_path"],
        "tracked_files": tracked_files,
    }


def get_merged_dir(adapter_path: Path) -> Path:
    """Return the cache directory used for the merged full model."""

    adapter_dir = resolve_adapter_dir(adapter_path)
    return adapter_dir.parent.parent / f"merged_{adapter_dir.parent.name}"


def merged_cache_is_fresh(merged_dir: Path, metadata: dict[str, object]) -> bool:
    """Whether a merged cache entry still matches the current adapter artifacts."""

    meta_path = merged_dir / MERGE_METADATA_FILENAME
    model_config = merged_dir / "config.json"
    if not meta_path.exists() or not model_config.exists():
        return False

    try:
        with open(meta_path) as f:
            existing = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return existing == metadata


def write_merge_metadata(merged_dir: Path, metadata: dict[str, object]) -> None:
    with open(merged_dir / MERGE_METADATA_FILENAME, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def prepare_merged_adapter(
    adapter_path: Path,
    requested_base_model: str | None = None,
) -> tuple[str, Path]:
    """Merge a LoRA adapter into its base model, refreshing stale caches.

    Returns:
        Tuple of (resolved_base_model_id, merged_model_dir)
    """

    adapter_dir = resolve_adapter_dir(adapter_path)
    metadata = build_merge_metadata(adapter_dir)
    base_model_id = str(metadata["base_model_name_or_path"])
    merged_dir = get_merged_dir(adapter_dir)

    if merged_cache_is_fresh(merged_dir, metadata):
        return base_model_id, merged_dir

    if requested_base_model and requested_base_model != base_model_id:
        print(
            f"Adapter base model {base_model_id} overrides requested --base-model "
            f"{requested_base_model}"
        )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"  Loading base model {base_model_id} for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()
    tokenizer = load_tokenizer(base_model_id)
    _align_special_token_ids(merged_model, tokenizer)

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))
    write_merge_metadata(merged_dir, metadata)

    del merged_model, peft_model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return base_model_id, merged_dir
