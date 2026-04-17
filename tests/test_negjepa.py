"""Tests for Negative-JEPA package (src/negbiojepa/).

Conventions follow test_models.py and test_ppi_models.py:
  - pytest.importorskip for torch
  - HAS_TORCH_GEOMETRIC conditional with @requires_pyg decorator
  - Class-based tests, shape assertions, gradient flow tests
  - Fixtures for tiny synthetic data (B=4 samples)
  - @pytest.mark.integration for tests requiring HPC parquet files

Run non-integration tests locally:
  PYTHONPATH=src python -m pytest tests/test_negjepa.py -v -k "not integration"
"""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires torch")
import torch.nn as nn

try:
    from torch_geometric.data import Batch, Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

requires_pyg = pytest.mark.skipif(
    not HAS_TORCH_GEOMETRIC, reason="requires torch_geometric"
)

# ─── Imports under test ───────────────────────────────────────────────────────

from negbiojepa.config import JEPAConfig
from negbiojepa.encoders import (
    TabularEncoder,
    PerceiverFusion,
    SequenceEncoder,
    UnifiedEncoder,
)
from negbiojepa.predictor import JEPAPredictor, EMAUpdater
from negbiojepa.losses import SIGReg, VICRegLoss, NegJEPALoss, check_collapse
from negbiojepa.masking import MultiLevelMasker
from negbiojepa.dataset import (
    NegJEPADataset,
    NegJEPASample,
    DOMAIN_ID,
    jepa_collate_fn,
    _ge_adapter,
    _ppi_adapter,
)

if HAS_TORCH_GEOMETRIC:
    from negbiojepa.encoders import MolGraphEncoder


# ─── Shared fixtures ──────────────────────────────────────────────────────────

B = 4        # batch size for all tests
D = 64       # embed_dim (small for fast tests)
F = 20       # tabular feature count
N_LATENTS = 8


@pytest.fixture
def cfg() -> JEPAConfig:
    return JEPAConfig(
        embed_dim=D,
        tabular_max_features=F,
        tabular_depth=1,
        tabular_n_heads=4,
        perceiver_n_latents=N_LATENTS,
        perceiver_depth=1,
        predictor_depth=2,
        predictor_n_heads=4,
        n_domains=8,
        reg_type="sigreg",
        use_ema=False,
        batch_size=B,
        epochs=1,
        data_root="synthetic",
    )


@pytest.fixture
def tiny_tabular_batch() -> dict:
    """B=4 tabular-only batch (GE-style)."""
    return {
        "tabular_A": torch.randn(B, F),
        "tabular_B": torch.randn(B, F),
        "domain_id": torch.zeros(B, dtype=torch.long),
        "label": torch.randint(0, 2, (B,)),
    }


@pytest.fixture
def tiny_seq_batch() -> dict:
    """B=4 batch with sequences (PPI-style)."""
    return {
        "tabular_A": torch.randn(B, F),
        "tabular_B": torch.randn(B, F),
        "seq_A": torch.randint(1, 22, (B, 30)),
        "seq_B": torch.randint(1, 22, (B, 30)),
        "domain_id": torch.ones(B, dtype=torch.long),
        "label": torch.randint(0, 2, (B,)),
    }


@pytest.fixture
def tiny_graph_batch():
    """B=4 batch with molecular graphs (DTI-style). Skipped if no PyG."""
    if not HAS_TORCH_GEOMETRIC:
        pytest.skip("requires torch_geometric")
    graphs = []
    for _ in range(B):
        n_atoms = 6
        x = torch.rand(n_atoms, 82)
        src = torch.tensor([0, 1, 2, 3, 4])
        dst = torch.tensor([1, 2, 3, 4, 5])
        edge_index = torch.stack([
            torch.cat([src, dst]), torch.cat([dst, src])
        ], dim=0)
        graphs.append(Data(x=x, edge_index=edge_index))
    return {
        "tabular_A": torch.randn(B, F),
        "tabular_B": torch.randn(B, F),
        "graph_A": Batch.from_data_list(graphs),
        "seq_B": torch.randint(1, 22, (B, 30)),
        "domain_id": torch.zeros(B, dtype=torch.long),
        "label": torch.randint(0, 2, (B,)),
    }


# ─── TestJEPAConfig ───────────────────────────────────────────────────────────

class TestJEPAConfig:
    def test_defaults(self):
        cfg = JEPAConfig()
        assert cfg.embed_dim == 256
        assert cfg.reg_type == "sigreg"
        assert cfg.use_ema is False

    def test_invalid_reg_type(self):
        with pytest.raises(ValueError, match="reg_type"):
            JEPAConfig(reg_type="invalid")

    def test_ema_sigreg_conflict(self):
        with pytest.raises(ValueError, match="use_ema=True requires"):
            JEPAConfig(reg_type="sigreg", use_ema=True)

    def test_head_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            JEPAConfig(embed_dim=100, tabular_n_heads=8)

    def test_replace(self):
        cfg = JEPAConfig()
        cfg2 = cfg.replace(embed_dim=128)
        assert cfg2.embed_dim == 128
        assert cfg.embed_dim == 256  # original unchanged

    def test_vicreg_with_ema(self):
        cfg = JEPAConfig(reg_type="vicreg", use_ema=True)
        assert cfg.use_ema is True


# ─── TestTabularEncoder ───────────────────────────────────────────────────────

class TestTabularEncoder:
    def test_output_shape(self, tiny_tabular_batch):
        enc = TabularEncoder(max_features=F, embed_dim=D, depth=1, n_heads=4)
        out = enc(tiny_tabular_batch["tabular_A"])
        assert out.shape == (B, F, D), f"Expected ({B}, {F}, {D}), got {out.shape}"

    def test_masked_tokens_differ(self, tiny_tabular_batch):
        enc = TabularEncoder(max_features=F, embed_dim=D, depth=1, n_heads=4)
        x = tiny_tabular_batch["tabular_A"]
        mask = torch.zeros(B, F, dtype=torch.bool)
        mask[:, :F // 2] = True
        out_masked = enc(x, mask)
        out_unmasked = enc(x)
        # At masked positions, output should differ (reg tokens replace values)
        assert not torch.allclose(out_masked[:, :F // 2], out_unmasked[:, :F // 2])

    def test_batch_mask_independence(self, tiny_tabular_batch):
        """Each sample in the batch must use its own mask, not sample 0's mask."""
        enc = TabularEncoder(max_features=F, embed_dim=D, depth=1, n_heads=4)
        x = tiny_tabular_batch["tabular_A"]
        # Give sample 0 a full mask, others no mask
        mask = torch.zeros(B, F, dtype=torch.bool)
        mask[0, :] = True  # only sample 0 fully masked
        out = enc(x, mask)
        out_no_mask = enc(x)
        # Sample 0 should differ (masked), sample 1 should not
        assert not torch.allclose(out[0], out_no_mask[0])
        assert torch.allclose(out[1], out_no_mask[1], atol=1e-5)

    def test_gradient_flows(self, tiny_tabular_batch):
        enc = TabularEncoder(max_features=F, embed_dim=D, depth=1, n_heads=4)
        out = enc(tiny_tabular_batch["tabular_A"])
        out.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert len(grads) > 0


# ─── TestMolGraphEncoder ──────────────────────────────────────────────────────

class TestMolGraphEncoder:
    @requires_pyg
    def test_output_shape(self, tiny_graph_batch):
        enc = MolGraphEncoder(in_features=82, hidden_dim=32, embed_dim=D, n_layers=2)
        out = enc(tiny_graph_batch["graph_A"])
        assert out.shape == (B, D), f"Expected ({B}, {D}), got {out.shape}"

    @requires_pyg
    def test_node_mask_changes_output(self, tiny_graph_batch):
        enc = MolGraphEncoder(in_features=82, hidden_dim=32, embed_dim=D, n_layers=2)
        graph = tiny_graph_batch["graph_A"]
        n_nodes = graph.x.shape[0]
        node_mask = torch.zeros(n_nodes, dtype=torch.bool)
        node_mask[:n_nodes // 2] = True
        out_masked = enc(graph, node_mask)
        out_clean = enc(graph)
        assert not torch.allclose(out_masked, out_clean)

    @requires_pyg
    def test_gradient_flows(self, tiny_graph_batch):
        enc = MolGraphEncoder(in_features=82, hidden_dim=32, embed_dim=D, n_layers=2)
        out = enc(tiny_graph_batch["graph_A"])
        out.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert len(grads) > 0


# ─── TestSequenceEncoder ──────────────────────────────────────────────────────

class TestSequenceEncoder:
    def test_seq_path_shape(self, tiny_seq_batch):
        enc = SequenceEncoder(vocab_size=26, embed_dim=D, n_filters=32)
        out = enc(seq=tiny_seq_batch["seq_A"])
        assert out.shape == (B, D)

    def test_esm2_path_shape(self):
        enc = SequenceEncoder(embed_dim=D, n_filters=32)
        esm2 = torch.randn(B, 1280)
        out = enc(esm2_emb=esm2)
        assert out.shape == (B, D)

    def test_requires_one_input(self):
        enc = SequenceEncoder(embed_dim=D, n_filters=32)
        with pytest.raises(ValueError, match="requires either"):
            enc()

    def test_gradient_seq(self, tiny_seq_batch):
        enc = SequenceEncoder(vocab_size=26, embed_dim=D, n_filters=32)
        out = enc(seq=tiny_seq_batch["seq_A"])
        out.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert len(grads) > 0


# ─── TestPerceiverFusion ──────────────────────────────────────────────────────

class TestPerceiverFusion:
    def test_fixed_output_size(self, tiny_tabular_batch):
        fusion = PerceiverFusion(embed_dim=D, n_latents=N_LATENTS, n_domains=8, depth=1, n_heads=4)
        tokens = torch.randn(B, 3, D)  # 3 modality tokens
        out = fusion(tokens, tiny_tabular_batch["domain_id"])
        assert out.shape == (B, N_LATENTS, D)

    def test_variable_modality_count(self, tiny_tabular_batch):
        fusion = PerceiverFusion(embed_dim=D, n_latents=N_LATENTS, n_domains=8, depth=1, n_heads=4)
        domain_id = tiny_tabular_batch["domain_id"]
        out1 = fusion(torch.randn(B, 1, D), domain_id)
        out5 = fusion(torch.randn(B, 5, D), domain_id)
        # Both should produce (B, N_LATENTS, D) regardless of modality count
        assert out1.shape == out5.shape == (B, N_LATENTS, D)

    def test_domain_tokens_affect_output(self):
        fusion = PerceiverFusion(embed_dim=D, n_latents=N_LATENTS, n_domains=8, depth=1, n_heads=4)
        tokens = torch.randn(B, 2, D)
        domain_0 = torch.zeros(B, dtype=torch.long)
        domain_3 = torch.full((B,), 3, dtype=torch.long)
        out0 = fusion(tokens, domain_0)
        out3 = fusion(tokens, domain_3)
        assert not torch.allclose(out0, out3)


# ─── TestUnifiedEncoder ───────────────────────────────────────────────────────

class TestUnifiedEncoder:
    def test_tabular_only(self, tiny_tabular_batch, cfg):
        enc = UnifiedEncoder(cfg)
        out = enc(tiny_tabular_batch)
        assert out.shape == (B, N_LATENTS, D)

    def test_seq_batch(self, tiny_seq_batch, cfg):
        enc = UnifiedEncoder(cfg)
        out = enc(tiny_seq_batch)
        assert out.shape == (B, N_LATENTS, D)

    @requires_pyg
    def test_graph_batch(self, tiny_graph_batch, cfg):
        enc = UnifiedEncoder(cfg)
        out = enc(tiny_graph_batch)
        assert out.shape == (B, N_LATENTS, D)

    def test_gradient_flows(self, tiny_tabular_batch, cfg):
        enc = UnifiedEncoder(cfg)
        out = enc(tiny_tabular_batch)
        out.sum().backward()
        grads = [p.grad for p in enc.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_ema_target_no_grad(self, tiny_tabular_batch, cfg):
        """With use_ema=True, target encoder params should not accumulate gradients."""
        cfg_ema = cfg.replace(reg_type="vicreg", use_ema=True)
        from negbiojepa.encoders import build_encoder_pair
        ctx_enc, tgt_enc = build_encoder_pair(cfg_ema)
        assert not any(p.requires_grad for p in tgt_enc.parameters())


# ─── TestJEPAPredictor ────────────────────────────────────────────────────────

class TestJEPAPredictor:
    def test_output_shape(self, cfg):
        predictor = JEPAPredictor.from_config(cfg)
        context = torch.randn(B, N_LATENTS, D)
        mask_positions = list(range(N_LATENTS // 2, N_LATENTS))
        out = predictor(context, mask_positions)
        assert out.shape == (B, len(mask_positions), D)

    def test_empty_mask_positions(self, cfg):
        predictor = JEPAPredictor.from_config(cfg)
        context = torch.randn(B, N_LATENTS, D)
        out = predictor(context, [])
        assert out.shape == (B, 0, D)

    def test_mask_token_has_grad(self, cfg):
        predictor = JEPAPredictor.from_config(cfg)
        context = torch.randn(B, N_LATENTS, D)
        out = predictor(context, [0, 1])
        out.sum().backward()
        assert predictor.mask_token.grad is not None


# ─── TestEMAUpdater ───────────────────────────────────────────────────────────

class TestEMAUpdater:
    def test_ema_weights_change(self, cfg):
        cfg_ema = cfg.replace(reg_type="vicreg", use_ema=True)
        from negbiojepa.encoders import build_encoder_pair
        online, target = build_encoder_pair(cfg_ema)
        # Perturb online params
        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p))
        # Store target before update
        before = {k: v.clone() for k, v in target.state_dict().items()}
        updater = EMAUpdater(base_decay=0.9, final_decay=0.9)
        updater.update(online, target, step=0, total_steps=100)
        # At least some target params should have changed
        changed = any(
            not torch.allclose(target.state_dict()[k], before[k])
            for k in before
        )
        assert changed

    def test_ema_decay_schedule(self):
        updater = EMAUpdater(base_decay=0.996, final_decay=0.999)
        # Cosine schedule: at step=0, decay = base_decay (0.996)
        # At step=total_steps, decay = final_decay (0.999)
        decay_start = updater.get_decay(0, 1000)
        decay_end = updater.get_decay(999, 1000)
        assert 0.996 <= decay_start <= 0.999
        assert 0.996 <= decay_end <= 0.999
        # Decay should increase over time (higher near end → more stable targets)
        assert decay_start < decay_end
        assert abs(decay_start - 0.996) < 0.002  # starts near base_decay
        assert abs(decay_end - 0.999) < 0.002    # ends near final_decay


# ─── TestSIGReg ───────────────────────────────────────────────────────────────

class TestSIGReg:
    def test_returns_scalar(self):
        reg = SIGReg(lmbda=1.0, sketch_dim=16)
        z = torch.randn(B, N_LATENTS, D)
        loss = reg(z)
        assert loss.shape == ()

    def test_not_nan(self):
        reg = SIGReg(lmbda=1.0, sketch_dim=16)
        z = torch.randn(B, N_LATENTS, D)
        loss = reg(z)
        assert not torch.isnan(loss)

    def test_collapsed_repr_high_loss(self):
        """SIGReg penalizes collapsed repr relative to well-scaled diverse repr.

        SIGReg minimizes MSE(cov(proj(z)), I) where proj scales by 1/sqrt(sketch_dim).
        For z ~ N(0, σ²I): projected cov ≈ σ² * (D/sketch_dim) * I.
        Minimum at σ² * (D/sketch_dim) = 1 → σ = sqrt(sketch_dim/D) = sqrt(16/64) = 0.5.
        Collapsed (all-ones → cov=0): MSE(0, I) = 16/256 = 0.0625 > ~0 for optimal σ.
        """
        import math
        torch.manual_seed(0)
        sketch_dim = 16
        optimal_scale = math.sqrt(sketch_dim / D)   # ≈ 0.5 for D=64, sketch_dim=16
        reg = SIGReg(lmbda=1.0, sketch_dim=sketch_dim)
        z_diverse = torch.randn(500, N_LATENTS, D) * optimal_scale  # cov_proj ≈ I
        z_collapsed = torch.ones(500, N_LATENTS, D)                 # cov_proj = 0
        assert reg(z_diverse) < reg(z_collapsed)


# ─── TestVICRegLoss ───────────────────────────────────────────────────────────

class TestVICRegLoss:
    def test_returns_scalar(self):
        reg = VICRegLoss()
        z = torch.randn(B, N_LATENTS, D)
        loss = reg(z)
        assert loss.shape == ()

    def test_not_nan(self):
        reg = VICRegLoss()
        z = torch.randn(B, N_LATENTS, D)
        assert not torch.isnan(reg(z))

    def test_collapsed_repr_high_loss(self):
        reg = VICRegLoss(lambda_var=1.0, lambda_cov=0.04, var_threshold=1.0)
        z_diverse = torch.randn(B * 4, N_LATENTS, D)  # larger batch for stable std
        z_collapsed = torch.ones(B * 4, N_LATENTS, D)
        assert reg(z_collapsed) > reg(z_diverse)


# ─── TestNegJEPALoss ──────────────────────────────────────────────────────────

class TestNegJEPALoss:
    def test_sigreg_mode_no_detach(self, cfg):
        """use_ema=False: gradients should flow through target."""
        loss_fn = NegJEPALoss(cfg)
        predicted = torch.randn(B, N_LATENTS // 2, D, requires_grad=True)
        target = torch.randn(B, N_LATENTS // 2, D, requires_grad=True)
        context_repr = torch.randn(B, N_LATENTS, D)
        loss, _ = loss_fn(predicted, target, context_repr, use_ema=False)
        loss.backward()
        # Both predicted and target should have gradients
        assert predicted.grad is not None
        assert target.grad is not None

    def test_ema_mode_detaches_target(self, cfg):
        """use_ema=True: target should be detached (no gradient)."""
        cfg_ema = cfg.replace(reg_type="vicreg", use_ema=True)
        loss_fn = NegJEPALoss(cfg_ema)
        predicted = torch.randn(B, N_LATENTS // 2, D, requires_grad=True)
        target = torch.randn(B, N_LATENTS // 2, D, requires_grad=True)
        context_repr = torch.randn(B, N_LATENTS, D)
        loss, _ = loss_fn(predicted, target, context_repr, use_ema=True)
        loss.backward()
        assert predicted.grad is not None
        assert target.grad is None  # detached — no gradient

    def test_metrics_dict(self, cfg):
        loss_fn = NegJEPALoss(cfg)
        predicted = torch.randn(B, 2, D)
        target = torch.randn(B, 2, D)
        context_repr = torch.randn(B, N_LATENTS, D)
        _, metrics = loss_fn(predicted, target, context_repr)
        assert "pred" in metrics
        assert "reg" in metrics


# ─── TestCheckCollapse ────────────────────────────────────────────────────────

class TestCheckCollapse:
    def test_healthy_repr_not_collapsed(self):
        z = torch.randn(32, N_LATENTS, D)
        collapsed, msg = check_collapse(z)
        assert not collapsed

    def test_point_collapse_detected(self):
        z = torch.ones(32, N_LATENTS, D) * 0.001
        collapsed, msg = check_collapse(z, min_std=0.01)
        assert collapsed
        assert "Point collapse" in msg

    def test_dimensional_collapse_detected(self):
        # All samples lie in a 1D subspace
        v = torch.randn(D)
        v = v / v.norm()
        z = torch.outer(torch.randn(32 * N_LATENTS), v).reshape(32, N_LATENTS, D)
        collapsed, msg = check_collapse(z, min_rank_ratio=0.1)
        assert collapsed


# ─── TestMultiLevelMasker ─────────────────────────────────────────────────────

class TestMultiLevelMasker:
    def test_entity_mask_shape(self, tiny_tabular_batch):
        masker = MultiLevelMasker()
        masks = masker.generate_masks(tiny_tabular_batch)
        assert masks["entity_mask"].shape == (B,)
        assert masks["entity_mask"].dtype == torch.bool

    def test_tab_mask_shape(self, tiny_tabular_batch):
        masker = MultiLevelMasker()
        masks = masker.generate_masks(tiny_tabular_batch)
        assert masks["tab_mask_A"].shape == (B, F)

    def test_masks_are_per_sample(self, tiny_tabular_batch):
        """Masks for different samples should differ (probabilistically)."""
        masker = MultiLevelMasker(feature_ratio=0.5)
        masks = masker.generate_masks(tiny_tabular_batch)
        tab_mask = masks["tab_mask_A"]
        # Very unlikely all rows are identical with 50% feature masking
        all_same = all(torch.equal(tab_mask[0], tab_mask[i]) for i in range(1, B))
        assert not all_same

    def test_apply_entity_mask_zeros_entity_b(self, tiny_tabular_batch):
        masker = MultiLevelMasker(entity_ratio=1.0)
        masks = masker.generate_masks(tiny_tabular_batch)
        # Force mask_side=1 (always mask B) for testing
        masks["entity_mask"] = torch.ones(B, dtype=torch.bool)
        masks["mask_side"] = torch.ones(B, dtype=torch.long)
        masked_batch = masker.apply_entity_mask_to_batch(tiny_tabular_batch, masks)
        assert torch.all(masked_batch["tabular_B"] == 0.0)
        # tabular_A should be unchanged
        assert torch.allclose(masked_batch["tabular_A"], tiny_tabular_batch["tabular_A"])


# ─── TestNegJEPADataset ───────────────────────────────────────────────────────

class TestNegJEPADataset:
    def test_synthetic_ge_dataset(self):
        ds = NegJEPADataset._make_synthetic("ge", max_features=F, n=10)
        assert len(ds) == 10
        sample = ds[0]
        assert isinstance(sample, NegJEPASample)
        assert sample.graph_A is None
        assert sample.seq_A is None
        assert sample.tabular_A.shape == (F,)
        assert sample.domain_id == DOMAIN_ID["ge"]

    @requires_pyg
    def test_synthetic_dti_has_graph(self):
        ds = NegJEPADataset._make_synthetic("dti", max_features=F, n=10)
        sample = ds[0]
        assert sample.graph_A is not None
        assert sample.graph_A.x.shape[1] == 82

    def test_labels_alternate(self):
        ds = NegJEPADataset._make_synthetic("ge", max_features=F, n=4)
        labels = [ds[i].label for i in range(4)]
        assert set(labels) == {0, 1}  # both classes present


# ─── TestCollate ──────────────────────────────────────────────────────────────

class TestCollate:
    def test_tabular_batch_shape(self):
        ds = NegJEPADataset._make_synthetic("ge", max_features=F, n=B)
        samples = [ds[i] for i in range(B)]
        batch = jepa_collate_fn(samples)
        assert batch["tabular_A"].shape == (B, F)
        assert batch["domain_id"].shape == (B,)
        assert batch["label"].shape == (B,)

    def test_seq_batch_present(self):
        ds = NegJEPADataset._make_synthetic("ppi", max_features=F, n=B)
        # Manually add sequences to synthetic samples
        samples = []
        for i in range(B):
            s = ds[i]
            s.seq_A = torch.randint(1, 22, (20,))
            s.seq_B = torch.randint(1, 22, (15,))
            samples.append(s)
        batch = jepa_collate_fn(samples)
        assert "seq_A" in batch
        assert batch["seq_A"].shape == (B, 20)  # padded to max length

    @requires_pyg
    def test_graph_batch_present(self):
        ds = NegJEPADataset._make_synthetic("dti", max_features=F, n=B)
        samples = [ds[i] for i in range(B)]
        batch = jepa_collate_fn(samples)
        assert "graph_A" in batch
        assert isinstance(batch["graph_A"], Batch)


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestIntegration:
    @pytest.mark.integration
    def test_pretrain_smoke(self, cfg):
        """End-to-end: 3 training steps on synthetic data; loss should decrease."""
        from negbiojepa.encoders import build_encoder_pair
        from negbiojepa.predictor import JEPAPredictor
        from negbiojepa.trainer import NegJEPATrainer
        from negbiojepa.dataset import NegJEPADataset, MultiDomainDataset, jepa_collate_fn

        device = torch.device("cpu")
        cfg_smoke = cfg.replace(domains=["ge"], data_root="synthetic", batch_size=8, epochs=1)

        ctx_enc, tgt_enc = build_encoder_pair(cfg_smoke)
        predictor = JEPAPredictor.from_config(cfg_smoke)
        trainer = NegJEPATrainer(cfg_smoke, ctx_enc, tgt_enc, predictor, device)

        ds_ge = NegJEPADataset._make_synthetic("ge", max_features=F, n=32)
        datasets = {"ge": ds_ge}
        combined = MultiDomainDataset(datasets)
        loader = torch.utils.data.DataLoader(
            combined, batch_size=8, shuffle=True, collate_fn=jepa_collate_fn, drop_last=True
        )

        losses = []
        total_steps = 3 * len(loader)
        for batch in loader:
            metrics = trainer._train_step(batch, total_steps)
            losses.append(metrics["loss"])
            if len(losses) >= 3:
                break

        assert len(losses) == 3
        assert all(not math.isnan(l) for l in losses), "Loss is NaN"
        # No collapse at end
        with torch.no_grad():
            for batch in loader:
                z = ctx_enc(batch)
                collapsed, _ = check_collapse(z)
                assert not collapsed, "Representation collapsed after smoke test"
                break

    @pytest.mark.integration
    def test_finetune_smoke(self, cfg):
        """Fine-tune a random encoder on tiny data; AUROC should be >= 0.4."""
        from negbiojepa.encoders import UnifiedEncoder
        from negbiojepa.trainer import NegJEPAFinetuner
        from negbiojepa.dataset import NegJEPADataset, jepa_collate_fn

        device = torch.device("cpu")
        cfg_ft = cfg.replace(
            domains=["ge"], data_root="synthetic",
            freeze_encoder=True, ft_epochs=3, ft_patience=3
        )
        encoder = UnifiedEncoder(cfg_ft)
        finetuner = NegJEPAFinetuner(cfg_ft, encoder, domain="ge", device=device)

        def make_loader(n):
            ds = NegJEPADataset._make_synthetic("ge", max_features=F, n=n)
            return torch.utils.data.DataLoader(
                ds, batch_size=8, shuffle=False, collate_fn=jepa_collate_fn
            )

        train_loader = make_loader(40)
        val_loader = make_loader(10)
        test_loader = make_loader(10)

        metrics = finetuner.fit(train_loader, val_loader, test_loader)
        assert "auroc" in metrics
        # With a random encoder, AUROC should be at least 0.3
        assert metrics["auroc"] >= 0.3, f"AUROC too low: {metrics['auroc']}"
