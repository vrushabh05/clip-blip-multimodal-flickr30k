"""Draw assets/clip_blip_results.png from results/metrics.csv."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
rows = list(csv.DictReader((root / "results" / "metrics.csv").open()))
order = ["zero-shot", "linear probe", "partial fine-tune", "LoRA", "full fine-tune"]


def pick(model, metric):
    d = {r["method"]: float(r["value"]) for r in rows if r["model"] == model and r["metric"] == metric}
    return [d[m] for m in order]


BG, INK, MUT, BLUE, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#a9a8a3"
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200, facecolor=BG)
for ax, (title, vals, ylabel) in zip(axes, [("CLIP text-to-image retrieval", pick("CLIP", "Recall@1"), "Recall@1"),
                                           ("BLIP captioning", pick("BLIP", "BLEU-4"), "BLEU-4")]):
    ax.set_facecolor(BG)
    bars = ax.bar(range(5), vals, color=[GREY] + [BLUE] * 4, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", color=INK, fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels(["zero-shot", "linear\nprobe", "partial\nFT", "LoRA", "full\nFT"], fontsize=8, color=MUT)
    ax.set_ylabel(ylabel, color=MUT); ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUT)
fig.tight_layout()
out = root / "assets" / "clip_blip_results.png"
fig.savefig(out, facecolor=BG)
print("wrote", out)
