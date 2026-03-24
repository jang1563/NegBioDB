#!/usr/bin/env python3
"""Generate all 3 paper figures as PDF files.

Figure 1: NegBioDB Architecture + Scale (architecture diagram + bar chart)
Figure 2: ML Cold-Split Catastrophe Heatmap (cross-domain AUROC heatmap)
Figure 3: L4 Opacity Gradient + Contamination (MCC bars + contamination panel)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

# NeurIPS style settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

OUTDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)


# ============================================================
# Figure 1: Architecture + Scale
# ============================================================
def fig1_overview():
    """Architecture diagram (Panel A) + stacked bar chart (Panel B)."""
    fig, (ax_arch, ax_bar) = plt.subplots(
        1, 2, figsize=(7, 2.8), gridspec_kw={"width_ratios": [1.3, 1]}
    )

    # --- Panel A: Architecture diagram ---
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 7)
    ax_arch.axis("off")
    ax_arch.set_title("(a) NegBioDB Architecture", fontsize=9, fontweight="bold", pad=4)

    # Common layer box
    common = FancyBboxPatch(
        (1, 5.5), 8, 1.2, boxstyle="round,pad=0.1",
        facecolor="#E8E8E8", edgecolor="black", linewidth=1.0
    )
    ax_arch.add_patch(common)
    ax_arch.text(5, 6.1, "Common Layer", ha="center", va="center",
                fontsize=8, fontweight="bold")
    ax_arch.text(5, 5.7, "Hypothesis | Evidence | Outcome | Confidence Tier",
                ha="center", va="center", fontsize=6, style="italic")

    # Domain boxes
    domains = [
        ("DTI", "#4C72B0", 1.0, [
            "ChEMBL (30.5M)", "PubChem", "BindingDB", "DAVIS"
        ]),
        ("CT", "#DD8452", 4.0, [
            "AACT (133K)", "Open Targets", "CTO", "Shi & Du"
        ]),
        ("PPI", "#55A868", 7.0, [
            "IntAct (2.2M)", "HuRI", "hu.MAP", "STRING"
        ]),
    ]
    for name, color, x, sources in domains:
        box = FancyBboxPatch(
            (x, 1.0), 2.0, 3.8, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.25
        )
        ax_arch.add_patch(box)
        ax_arch.text(x + 1.0, 4.4, name, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=color)
        for i, src in enumerate(sources):
            ax_arch.text(x + 1.0, 3.6 - i * 0.65, src, ha="center",
                        va="center", fontsize=5.5)

        # Arrow from common to domain
        ax_arch.annotate(
            "", xy=(x + 1.0, 4.8), xytext=(x + 1.0, 5.5),
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8)
        )

    # --- Panel B: Stacked bar chart ---
    ax_bar.set_title("(b) Scale by Confidence Tier", fontsize=9, fontweight="bold", pad=4)

    # Tier data (verified from database queries)
    tier_colors = {"Gold": "#FFD700", "Silver": "#C0C0C0", "Bronze": "#CD7F32", "Copper": "#B87333"}
    domains_data = {
        "DTI": {"Gold": 818611, "Silver": 774875, "Bronze": 28866097, "Copper": 0},
        "PPI": {"Gold": 500069, "Silver": 1229601, "Bronze": 500000, "Copper": 0},
        "CT": {"Gold": 23570, "Silver": 28505, "Bronze": 60223, "Copper": 20627},
    }

    x_pos = np.arange(3)
    labels = ["DTI", "PPI", "CT"]
    bottom = np.zeros(3)

    for tier, color in tier_colors.items():
        vals = [domains_data[d][tier] for d in labels]
        ax_bar.bar(x_pos, vals, 0.6, bottom=bottom, color=color, label=tier,
                  edgecolor="white", linewidth=0.5)
        bottom += vals

    ax_bar.set_yscale("log")
    ax_bar.set_ylabel("Negative Results")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(1e4, 5e7)
    ax_bar.legend(loc="upper right", framealpha=0.9, ncol=2)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Totals on top
    totals = [30.5e6, 2.23e6, 132925]
    for i, t in enumerate(totals):
        if t >= 1e6:
            label = f"{t/1e6:.1f}M"
        else:
            label = f"{t/1e3:.0f}K"
        ax_bar.text(i, bottom[i] * 1.15, label, ha="center", va="bottom", fontsize=7,
                   fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig1_overview.pdf")
    plt.close(fig)
    print("  -> fig1_overview.pdf")


# ============================================================
# Figure 2: ML Cold-Split Catastrophe Heatmap
# ============================================================
def fig2_ml_heatmap():
    """Cross-domain ML AUROC heatmap showing cold-split catastrophe."""
    # Data: AUROC values (negbiodb, best seed or 3-seed avg)
    # Rows: (Domain, Model)
    # Columns: split strategies

    row_labels = [
        "DTI / DeepDTA",
        "DTI / GraphDTA",
        "DTI / DrugBAN",
        "CT / XGBoost",
        "CT / MLP",
        "CT / GNN",
        "PPI / SiameseCNN",
        "PPI / PIPR",
        "PPI / MLPFeatures",
    ]

    # Column labels: Random, Cold-X, Cold-Y, DDB
    col_labels = ["Random", "Cold-X", "Cold-Y", "DDB"]

    # AUROC data matrix
    # DTI: seed 42, negbiodb negatives
    # Cold-X = cold_compound (DTI), cold_drug (CT), cold_protein (PPI)
    # Cold-Y = cold_target (DTI), cold_condition (CT), cold_both (PPI)
    data = np.array([
        # DTI (seed 42)
        [0.997, 0.996, 0.887, 0.997],  # DeepDTA
        [0.997, 0.997, 0.863, 0.997],  # GraphDTA
        [0.997, 0.997, 0.760, 0.997],  # DrugBAN
        # CT (seed 42, mean of 3 seeds where available)
        [1.000, 1.000, 1.000, np.nan],  # XGBoost (no DDB)
        [1.000, 1.000, 1.000, np.nan],  # MLP
        [1.000, 1.000, 1.000, np.nan],  # GNN
        # PPI (3-seed average)
        [0.963, 0.873, 0.585, 0.962],  # SiameseCNN
        [0.964, 0.859, 0.409, 0.964],  # PIPR
        [0.962, 0.931, 0.950, 0.961],  # MLPFeatures
    ])

    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    # Create masked array for NaN
    masked = np.ma.masked_invalid(data)

    # Custom colormap: red for catastrophe, green for good
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ["#d62728", "#ff7f0e", "#ffdd57", "#98df8a", "#2ca02c"]
    cmap = LinearSegmentedColormap.from_list("catastrophe", colors_list, N=256)
    cmap.set_bad(color="#f0f0f0")

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0.3, vmax=1.0)

    # Annotate cells
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                       fontsize=6.5, color="gray")
            else:
                color = "white" if val < 0.6 else "black"
                weight = "bold" if val < 0.7 else "normal"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                       fontsize=6.5, color=color, fontweight=weight)

    # Domain separators
    ax.axhline(2.5, color="black", linewidth=1.5)
    ax.axhline(5.5, color="black", linewidth=1.5)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Domain labels on right
    for y, label in [(1, "DTI"), (4, "CT"), (7, "PPI")]:
        ax.text(len(col_labels) - 0.3, y, label, ha="left", va="center",
               fontsize=8, fontweight="bold", color="gray",
               transform=ax.get_yaxis_transform())

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.08)
    cbar.set_label("AUROC", fontsize=8)

    ax.set_title("ML Cold-Split Performance (AUROC)", fontsize=9,
                fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig2_ml_heatmap.pdf")
    plt.close(fig)
    print("  -> fig2_ml_heatmap.pdf")


# ============================================================
# Figure 3: L4 Opacity Gradient + Contamination
# ============================================================
def fig3_l4_gradient():
    """Panel A: L4 MCC bars across domains. Panel B: PPI contamination."""
    fig, (ax_mcc, ax_contam) = plt.subplots(
        1, 2, figsize=(7, 2.5), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # --- Panel A: L4 MCC across domains ---
    # Use best config (3-shot) for each model, common models only
    # 4 common models: Gemini, GPT-4o-mini, Llama, Qwen
    # + Haiku for CT/PPI (DTI N/A)
    models = ["Gemini", "GPT-4o", "Llama", "Qwen", "Haiku"]
    model_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    # MCC values (best config per model — may be zero-shot or 3-shot)
    # DTI: Gemini 3s, GPT 3s, Llama 3s, Qwen 3s, Haiku N/A
    dti_mcc = [-0.102, 0.047, 0.184, 0.113, np.nan]
    # PPI: Gemini 3s, GPT 0s, Llama 0s, Qwen 3s, Haiku 3s
    ppi_mcc = [0.382, 0.430, 0.441, 0.369, 0.390]
    # CT: Gemini 3s, GPT 0s, Llama 3s, Qwen 0s, Haiku 0s
    ct_mcc = [0.563, 0.491, 0.504, 0.519, 0.514]

    x = np.arange(3)  # 3 domains
    n_models = len(models)
    width = 0.15
    offsets = np.arange(n_models) - (n_models - 1) / 2

    for i, (model, color) in enumerate(zip(models, model_colors)):
        vals = [dti_mcc[i], ppi_mcc[i], ct_mcc[i]]
        positions = x + offsets[i] * width
        bars = ax_mcc.bar(positions, vals, width * 0.9, color=color, label=model,
                         edgecolor="white", linewidth=0.3)
        # Mark NaN bars
        for j, v in enumerate(vals):
            if np.isnan(v):
                ax_mcc.text(positions[j], 0.02, "N/A", ha="center", va="bottom",
                           fontsize=5, color="gray", rotation=90)

    ax_mcc.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_mcc.set_xticks(x)
    ax_mcc.set_xticklabels(["DTI\n(opaque)", "PPI\n(crawlable)", "CT\n(public)"])
    ax_mcc.set_ylabel("MCC")
    ax_mcc.set_ylim(-0.15, 0.65)
    ax_mcc.set_title("(a) L4 Discrimination: The Opacity Gradient",
                    fontsize=9, fontweight="bold", pad=4)
    ax_mcc.legend(loc="upper left", ncol=3, framealpha=0.9, fontsize=6)
    ax_mcc.spines["top"].set_visible(False)
    ax_mcc.spines["right"].set_visible(False)

    # Trend arrow
    ax_mcc.annotate(
        "", xy=(2.35, 0.55), xytext=(-0.15, 0.0),
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5,
                       connectionstyle="arc3,rad=0.15", alpha=0.4)
    )

    # --- Panel B: PPI Contamination ---
    # Pre-2015 vs Post-2020 accuracy per model (best 3-shot run)
    contam_models = ["Gemini", "GPT-4o", "Llama", "Qwen", "Haiku"]
    pre_2015 = [0.765, 0.569, 0.745, 0.588, 0.618]
    post_2020 = [0.184, 0.112, 0.133, 0.112, 0.051]

    x_c = np.arange(len(contam_models))
    w = 0.35

    ax_contam.bar(x_c - w/2, pre_2015, w, color="#4C72B0", label="Pre-2015",
                 edgecolor="white", linewidth=0.3)
    ax_contam.bar(x_c + w/2, post_2020, w, color="#DD8452", label="Post-2020",
                 edgecolor="white", linewidth=0.3)

    # Gap annotations
    for i in range(len(contam_models)):
        gap = pre_2015[i] - post_2020[i]
        mid = (pre_2015[i] + post_2020[i]) / 2
        ax_contam.annotate(
            f"\u0394={gap:.2f}", xy=(i, mid), fontsize=5.5,
            ha="center", va="center", color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                     edgecolor="none", alpha=0.8)
        )

    ax_contam.axhline(0.5, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
    ax_contam.set_xticks(x_c)
    ax_contam.set_xticklabels(contam_models, fontsize=6.5)
    ax_contam.set_ylabel("Accuracy")
    ax_contam.set_ylim(0, 0.9)
    ax_contam.set_title("(b) PPI Contamination (L4)",
                       fontsize=9, fontweight="bold", pad=4)
    ax_contam.legend(loc="upper right", framealpha=0.9, fontsize=6)
    ax_contam.spines["top"].set_visible(False)
    ax_contam.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(OUTDIR / "fig3_l4_gradient.pdf")
    plt.close(fig)
    print("  -> fig3_l4_gradient.pdf")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_overview()
    fig2_ml_heatmap()
    fig3_l4_gradient()
    print("Done. Figures saved to:", OUTDIR)
