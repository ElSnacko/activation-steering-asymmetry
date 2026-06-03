"""Cross-model steering figure: genuine compliance vs hedge vs degeneration (comply
direction), and refusal-deepening degeneration vs coherent no-op (refusal direction).
Mean KL (neutral prompts) is overlaid on every panel, right axis. Reads recovered
optimize_alpha summaries. CPU-only, no model load."""
import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = {
    "Qwen3.5-9B": "outputs/qwen3-5-9b/20260526-beavertails-mlp",
    "Mistral-7B": "outputs/mistral-7b-instruct-v0-2/20260527-beavertails-mlp",
}

def load(model, run):
    f = os.path.join(BASE[model], run, "optimization_summary.json")
    d = json.load(open(f))
    rows = []
    for t in d["all_results"]:
        m = t["metrics"]
        rows.append((t["alpha"], m["compliance_rate"], m["refusal_rate"],
                     m["degenerate_rate"], m["strong_comply_rate"],
                     m["weak_comply_rate"], t.get("kl_divergence", {}).get("mean_kl", np.nan)))
    rows.sort()
    return np.array(rows)

def align_twin(ax_left, ax_right, kl_max):
    """Make the left (% of responses) and right (mean KL) axes share gridlines:
    both get 6 evenly-spaced ticks over a fixed range, so the marks line up."""
    top = max(math.ceil(kl_max / 0.25) * 0.25, 0.5)  # nice round KL ceiling
    ax_left.set_ylim(0, 100)
    ax_right.set_ylim(0, top)
    ax_left.yaxis.set_major_locator(mticker.LinearLocator(6))   # 0,20,40,60,80,100
    ax_right.yaxis.set_major_locator(mticker.LinearLocator(6))  # 0 .. top in 5 steps
    ax_left.grid(True, alpha=0.25)
    ax_right.grid(False)

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Steering toward non-default behaviour is absorbed, not converted\n"
             "BeaverTails · 4-layer MLP steering · α chosen by KL-weighted optimisation",
             fontsize=12, fontweight="bold")

# Row 0: comply direction (global balanced run) — strong vs hedge vs degen (+ KL twin axis)
for j, model in enumerate(["Qwen3.5-9B", "Mistral-7B"]):
    r = load(model, "optimize_alpha_global_validation")
    a = -r[:, 0]
    o = np.argsort(a)
    a = a[o]
    ax[0, j].plot(a, r[o, 4]*100, "o-", color="#dc2626", lw=2, label="genuine comply (strong, −1.0)")
    ax[0, j].plot(a, r[o, 5]*100, "s--", color="#f59e0b", lw=1.8, label="hedge (weak, −0.5)")
    ax[0, j].plot(a, r[o, 3]*100, "^:", color="#6b7280", lw=1.8, label="degenerate")
    ax[0, j].set_title(f"{model} — steering toward COMPLIANCE  (n={len(r)} trials)",
                       fontsize=10, fontweight="bold")
    ax[0, j].set_xlabel("|α|  (compliance push)")
    ax[0, j].set_ylabel("% of responses")
    twin = ax[0, j].twinx()
    twin.plot(a, r[o, 6], "D--", color="#16a34a", lw=1.4, ms=4, label="mean KL (harmless)")
    twin.set_ylabel("mean KL", color="#16a34a")
    twin.tick_params(axis="y", labelcolor="#16a34a")
    align_twin(ax[0, j], twin, np.nanmax(r[:, 6]))
    lines = ax[0, j].get_lines() + twin.get_lines()
    ax[0, j].legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

# Row 1: refusal direction — refuse vs degen (+ KL twin axis)
for j, model in enumerate(["Qwen3.5-9B", "Mistral-7B"]):
    r = load(model, "optimize_alpha_refuse")
    a = r[:, 0]
    o = np.argsort(a)
    a = a[o]
    ax[1, j].plot(a, r[o, 2]*100, "o-", color="#2563eb", lw=2, label="refusal (over-refusal on benign)")
    ax[1, j].plot(a, r[o, 3]*100, "^:", color="#6b7280", lw=1.8, label="degenerate")
    ax[1, j].set_title(f"{model} — steering toward REFUSAL  (n={len(r)} trials)",
                       fontsize=10, fontweight="bold")
    ax[1, j].set_xlabel("α  (refusal push)")
    ax[1, j].set_ylabel("% of responses")
    twin = ax[1, j].twinx()
    twin.plot(a, r[o, 6], "D--", color="#16a34a", lw=1.4, ms=4, label="mean KL (harmless)")
    twin.set_ylabel("mean KL", color="#16a34a")
    twin.tick_params(axis="y", labelcolor="#16a34a")
    align_twin(ax[1, j], twin, np.nanmax(r[:, 6]))
    lines = ax[1, j].get_lines() + twin.get_lines()
    ax[1, j].legend(lines, [l.get_label() for l in lines], fontsize=8, loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = "fig_steering_asymmetry.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)

# Note: trial counts differ by run (7–10 α points). Later Bayesian trials that added no
# uplift were dropped to save compute; see the limitations section.

summary = {}
for model in BASE:
    summary[model] = {}
    for run in ["optimize_alpha_global_validation", "optimize_alpha_refuse"]:
        r = load(model, run)
        summary[model][run] = {
            "n_trials": len(r),
            "points": [
                {"alpha": round(x[0], 3), "comply": x[1], "refuse": x[2], "degen": x[3],
                 "strong": x[4], "weak": x[5], "mean_kl": x[6]} for x in r],
        }
json.dump(summary, open("fig_steering_asymmetry_data.json", "w"), indent=1)
print("wrote fig_steering_asymmetry_data.json")
