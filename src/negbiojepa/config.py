"""JEPAConfig: YAML-backed configuration dataclass for Negative-JEPA training."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class JEPAConfig:
    """Full configuration for Negative-JEPA pretraining and fine-tuning.

    Defaults correspond to the recommended Option A (SIGReg, no EMA) on 3 domains.
    For Option B (VICReg + EMA), set reg_type="vicreg" and use_ema=True.
    """

    # ── Encoder ───────────────────────────────────────────────────────────────
    embed_dim: int = 256
    """Shared latent dimension D for all sub-encoders and predictor."""

    tabular_max_features: int = 300
    """Maximum tabular feature count; inputs zero-padded to this size.
    MD domain has 2068 features — apply PCA/feature-selection before passing."""

    tabular_depth: int = 2
    """Number of Transformer layers in TabularEncoder."""

    tabular_n_heads: int = 8
    """Number of attention heads in TabularEncoder Transformer."""

    perceiver_n_latents: int = 16
    """Number of fixed-size latent queries in PerceiverFusion."""

    perceiver_depth: int = 2
    """Number of cross-attention + self-attention rounds in PerceiverFusion."""

    n_domains: int = 8
    """Total number of domains (size of domain embedding table)."""

    # ── Predictor ─────────────────────────────────────────────────────────────
    predictor_depth: int = 4
    """Transformer depth for JEPAPredictor (intentionally smaller than encoder)."""

    predictor_n_heads: int = 8
    """Number of attention heads in JEPAPredictor."""

    # ── Training objective ────────────────────────────────────────────────────
    reg_type: str = "sigreg"
    """Regularization type: 'sigreg' (LeJEPA, Option A) or 'vicreg' (V-JEPA, Option B)."""

    use_ema: bool = False
    """If True, use EMA target encoder (Option B). If False, gradients flow through
    both encoder branches (required for LeJEPA/SIGReg correctness)."""

    ema_base_decay: float = 0.996
    """EMA decay at the start of training (Option B only)."""

    ema_final_decay: float = 0.999
    """EMA decay at the end of training, annealed via cosine schedule (Option B only)."""

    sigreg_lambda: float = 1.0
    """SIGReg regularization weight (Option A)."""

    sigreg_sketch_dim: int = 64
    """Sketch dimension for SIGReg random projection (Option A)."""

    vicreg_lambda_var: float = 1.0
    """VICReg variance loss weight (Option B)."""

    vicreg_lambda_cov: float = 0.04
    """VICReg covariance loss weight (Option B)."""

    # ── Optimizer ─────────────────────────────────────────────────────────────
    lr: float = 3e-4
    """Peak learning rate for AdamW."""

    weight_decay: float = 0.05
    """AdamW weight decay."""

    warmup_epochs: int = 5
    """Linear warmup epochs before cosine decay begins."""

    epochs: int = 100
    """Total pretraining epochs."""

    batch_size: int = 256
    """Per-domain batch size (before multi-domain collation)."""

    grad_clip: float = 1.0
    """Gradient norm clipping threshold; 0.0 = disabled."""

    # ── Masking ───────────────────────────────────────────────────────────────
    entity_mask_ratio: float = 0.5
    """Fraction of samples where entity A or entity B is fully masked."""

    feature_mask_ratio: float = 0.5
    """Per-feature masking probability for tabular feature masking (T-JEPA style)."""

    subgraph_mask_ratio: float = 0.2
    """Fraction of graph nodes masked in subgraph masking (Graph-JEPA style)."""

    # ── Data ──────────────────────────────────────────────────────────────────
    domains: List[str] = field(default_factory=lambda: ["dti", "ppi", "ge"])
    """Domains to include in multi-domain pretraining.
    Phase 3 initial trio: dti + ppi + ge (covers all 3 modality encoders)."""

    domain_temp: float = 0.5
    """Temperature τ for domain sampling: P(d) ∝ |D_d|^τ.
    τ=0.5 gives sqrt-scaled sampling; τ=1.0 is proportional; τ=0.0 is uniform."""

    data_root: str = "/athena/masonlab/scratch/users/jak4013/negbiodb_exports"
    """Root directory containing per-domain parquet export subdirectories.
    For local smoke tests, set to 'synthetic' to generate data in-memory."""

    max_samples_per_domain: int = 0
    """Cap samples per domain (0=no cap). Useful when full DTI (25M) + GE (22M) exceed RAM.
    Set to e.g. 2_000_000 to fit in 64GB with lazy loading."""

    num_workers: int = 4
    """DataLoader worker processes."""

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir: str = "./outputs/negjepa"
    """Directory for checkpoints and logs."""

    log_every: int = 100
    """Log metrics every N training steps."""

    save_every: int = 10
    """Save checkpoint every N epochs."""

    # ── Fine-tuning (used by NegJEPAFinetuner) ────────────────────────────────
    freeze_encoder: bool = True
    """If True, freeze encoder weights during fine-tuning (linear probing mode)."""

    ft_lr: float = 1e-4
    """Fine-tuning learning rate (encoder unfrozen) or linear head lr (frozen)."""

    ft_epochs: int = 50
    """Maximum fine-tuning epochs."""

    ft_warmup_epochs: int = 5
    """LR warmup epochs during fine-tuning (cosine schedule same as pretraining)."""

    ft_patience: int = 10
    """Early stopping patience (measured on val MCC)."""

    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "JEPAConfig":
        """Load config from YAML file, merging with dataclass defaults."""
        import yaml  # lazy import — not required for config creation
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        # Convert list fields that may be loaded as non-list
        if "domains" in data and isinstance(data["domains"], str):
            data["domains"] = [data["domains"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Serialize config to YAML file."""
        import yaml
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, default_flow_style=False, sort_keys=True)

    def replace(self, **kwargs) -> "JEPAConfig":
        """Return a new JEPAConfig with specified fields overridden."""
        d = asdict(self)
        d.update(kwargs)
        return JEPAConfig(**d)

    def __post_init__(self) -> None:
        if self.reg_type not in ("sigreg", "vicreg"):
            raise ValueError(f"reg_type must be 'sigreg' or 'vicreg', got {self.reg_type!r}")
        if self.use_ema and self.reg_type == "sigreg":
            raise ValueError(
                "use_ema=True requires reg_type='vicreg'. "
                "LeJEPA/SIGReg must NOT use EMA (gradients must flow through target encoder)."
            )
        if self.embed_dim % self.tabular_n_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by tabular_n_heads ({self.tabular_n_heads})"
            )
        if self.embed_dim % self.predictor_n_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by predictor_n_heads ({self.predictor_n_heads})"
            )
