"""
Regenerates the two-panel results figure for the README
(AUC forest plot + GPT-4o few-shot learning curves).

Values are taken from Ahn et al., JMIR AI (2026). This script exists so the
figure can be reproduced; replace figures/model_results.png with the original
publication figure if you prefer the exact typeset version.
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# ---------------------------------------------------------------- Panel A: AUC
ax = axes[0]
models = ["LogReg", "RandomForest", "XG Boost", "tabnet", "tabicl", "tabpfn"]
auc    = [0.7748, 0.7889, 0.7267, 0.6748, 0.7067, 0.7859]
lo     = [0.73,   0.66,   0.53,   0.44,   0.60,   0.63]
hi     = [0.81,   0.91,   0.92,   0.91,   0.81,   0.94]

y = np.arange(len(models))[::-1]  # top-to-bottom order matching the paper
navy = "#1f2d4d"
for yi, m, a, l, h in zip(y, models, auc, lo, hi):
    ax.errorbar(a, yi, xerr=[[a - l], [h - a]], fmt="o", color=navy,
                ecolor=navy, elinewidth=1.6, capsize=6, capthick=1.6,
                markersize=9)
    ax.text(a, yi + 0.22, f"{a:.4f}", ha="center", va="bottom",
            fontsize=11, color="#222")

ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=11)
ax.set_xlim(0, 1.0)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("Confidence Interval", fontsize=11)
ax.set_ylabel("Models", fontsize=11)
ax.set_title("AUC", fontsize=15, fontweight="bold", color="white",
             backgroundcolor="#1f2d4d", pad=12)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.grid(axis="x", color="#cccccc", linewidth=0.8)
ax.set_axisbelow(True)

# ------------------------------------------------ Panel B: few-shot GPT-4o
ax = axes[1]
shots = [20, 30, 40, 50, 60, 70]
baseline = [0.7396, 0.7220, 0.7152, 0.7565, 0.7538, 0.7333]  # 30-shot interpolated
vars7    = [0.7245, 0.7256, 0.7030, 0.7652, 0.7846, 0.8667]
vars100  = [0.6038, 0.5953, 0.6121, 0.6870, 0.6615, 0.8000]

blue, red, yellow = "#3b7ddd", "#e8322a", "#f5b800"
ax.plot(shots, baseline, "-", color=blue,   lw=2.5, label="Baseline (RF)")
ax.plot(shots, vars7,    "-", color=red,    lw=2.5, label="7 vars token")
ax.plot(shots, vars100,  "-", color=yellow, lw=2.5, label="100 vars token")

# annotate the labeled points from the paper
def annotate(xs, ys, series, color, skip=()):
    for x, v in zip(xs, ys):
        if x in skip:
            continue
        ax.text(x, v + 0.012, f"{v:.4g}", ha="center", va="bottom",
                fontsize=8.5, color=color)
annotate(shots, baseline, "b", blue, skip=(30,))
annotate(shots, vars7, "r", red)
annotate(shots, vars100, "y", "#c99400")

ax.set_ylim(0.5, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticks(shots)
ax.set_xticklabels([f"{s}-shot" for s in shots], fontsize=10)
ax.set_ylabel("Accuracy", fontsize=13)
ax.set_title("Few-shot learning (GPT-4o)", fontsize=15, fontweight="bold",
             color="white", backgroundcolor="#3b7ddd", pad=12)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3,
          frameon=False, fontsize=9)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.grid(axis="y", color="#dddddd", linewidth=0.8)
ax.set_axisbelow(True)

fig.text(0.99, 0.01, "(Ahn et al., JMIR AI 2026)", ha="right", fontsize=10,
         style="italic", color="#333")
plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("figures/model_results.png", dpi=200, bbox_inches="tight")
print("wrote figures/model_results.png")
