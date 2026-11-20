import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

RANDOM_STATE = 42
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")  # save figures only, no pop-up windows

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 60)
pd.set_option("display.max_columns", 30)


def find_data_zip():
    """Locate the dataset zip (data/ folder first, then the project folder)."""
    for folder in (DATA_DIR, PROJECT_DIR):
        if folder.exists():
            matches = sorted(folder.glob("*TxtFiles1870-1894*.zip"))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        "Cannot find the dataset zip. Put 'TxtFiles1870-1894_TB.zip' "
        f"(any name containing 'TxtFiles1870-1894') in {DATA_DIR}"
    )


def show(obj):
    """Print a DataFrame/Series/object (replaces the notebook's display())."""
    if hasattr(obj, "to_string"):
        print(obj.to_string())
    else:
        print(obj)


def finish_plot(name=None):
    """Finish the current figure: save it (if a name is given), optionally
    show it on screen, then close it."""
    if name is not None:
        plt.savefig(FIGURE_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")
