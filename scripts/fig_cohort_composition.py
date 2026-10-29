"""Plot site-level psychosis and no-psychosis sample sizes from T_demographics.csv."""
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from psikoz.report import CLASS_COL, save


def main():
    table = pd.read_csv(ROOT / "results" / "T_demographics.csv")
    table = table[table["Site"] != "All sites"].copy()
    if table.empty:
        raise ValueError("T_demographics.csv contains no site-level rows")

    sites = table["Site"].tolist()
    no_psychosis = pd.to_numeric(table["n (No psychosis)"], errors="raise").to_numpy()
    psychosis = pd.to_numeric(table["n (Psychosis)"], errors="raise").to_numpy()
    y = np.arange(len(sites))
    bar_height = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    bars_no = ax.barh(y - bar_height / 2, no_psychosis, height=bar_height,
                      color=CLASS_COL[0], label="No psychosis")
    bars_yes = ax.barh(y + bar_height / 2, psychosis, height=bar_height,
                       color=CLASS_COL[1], label="Psychosis")
    ax.set_yticks(y, sites)
    ax.invert_yaxis()
    ax.set_xlabel("Participants (n)")
    ax.set_title("Sample composition by site")
    ax.set_xlim(0, max(no_psychosis + psychosis) * 1.16)
    ax.axhline(len(sites) - 1.5, color="#777777", linewidth=0.7, linestyle=":")

    for bars in (bars_no, bars_yes):
        for bar in bars:
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width())}", va="center", ha="left", fontsize=6)

    ax.legend(loc="lower right")
    ax.text(0.99, 0.98, "TCP is the external test cohort", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.3, color="#444444")
    fig.tight_layout()
    save(fig, ROOT / "results", "F_cohort_composition")


if __name__ == "__main__":
    main()
