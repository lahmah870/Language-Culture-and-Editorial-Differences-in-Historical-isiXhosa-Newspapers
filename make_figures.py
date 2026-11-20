import numpy as np
import pandas as pd

from config import OUTPUT_DIR, FIGURE_DIR, plt, finish_plot

plt.rcParams.update({"font.size": 7})

TOPIC_NAMES = [
    "1 Generic isiXhosa",
    "2 English",
    "3 Retail (goods)",
    "4 Retail (prices)",
    "5 Retail (soap, shops)",
    "6 Clothing",
    "7 Politics/institutions",
    "8 Medicine ads",
]

# ---------------- Figure 1: topic weights --------------------------------
summary = pd.read_csv(
    OUTPUT_DIR / "nmf_year_matched_topic_summary.csv", index_col=0
)

y = np.arange(len(TOPIC_NAMES))[::-1]
h = 0.38

fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.barh(y + h / 2, summary.loc["IMVO"].values, h, label="IMVO", color="#1b7f79")
ax.barh(y - h / 2, summary.loc["ISIG"].values, h, label="ISIG", color="#a04a8f")

for yy, a, b in zip(y, summary.loc["IMVO"].values, summary.loc["ISIG"].values):
    ax.text(a + 0.01, yy + h / 2, f"{a:.2f}", va="center", fontsize=6)
    ax.text(b + 0.01, yy - h / 2, f"{b:.2f}", va="center", fontsize=6)

ax.set_yticks(y)
ax.set_yticklabels(TOPIC_NAMES)
ax.set_xlim(0, 0.7)
ax.set_xlabel("Mean relative topic weight")
ax.legend(frameon=False, loc="lower right")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
finish_plot("paper_fig1_topics")

# ---------------- Figure 2: shortcut baselines ---------------------------
data = pd.read_csv(OUTPUT_DIR / "shortcut_baselines.csv")
colours = {
    "Shortcut baseline": "#9a9a9a",
    "TF-IDF classifier": "#2a6fb0",
    "Post-hoc robustness": "#e08a2c",
}
data = data.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
bars = ax.barh(
    data["model"], data["macro_f1"],
    color=[colours[g] for g in data["group"]]
)
for bar, value in zip(bars, data["macro_f1"]):
    ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}", va="center", fontsize=6)

ax.set_xlim(0, 1.15)
ax.set_xlabel("Test macro-F1")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours.values()]
ax.legend(handles, colours.keys(), loc="upper center",
          bbox_to_anchor=(0.4, -0.22), ncol=3, fontsize=5.5, frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.tight_layout()
finish_plot("paper_fig2_shortcuts")

print("Saved figures to", FIGURE_DIR)
