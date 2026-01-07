import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main(csv_path: Path, cutoff: float, out_png: Path, show: bool):
    df = pd.read_csv(csv_path)

    # Filter by threshold cutoff (features with threshold <= cutoff are "assigned")
    df_cut = df[df["threshold"] >= cutoff]

    # Count unique dimensions per family (how many CC features map to each family)
    counts = df_cut.groupby("class")["dim"].nunique()

    # Prepare plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.histplot(counts.values, bins=range(1, counts.max() + 2), discrete=True)
    plt.xlabel("Number of CC features per family")
    plt.ylabel("Number of families")
    plt.title(f"Histogram of CC features per family (cutoff={cutoff})")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)

    if show:
        plt.show()

    # Print light summary
    print(f"Saved histogram to {out_png}")
    print(f"Families considered: {len(counts)}")
    print(f"Mean CC features per family: {counts.mean():.3f}")
    print(f"Median CC features per family: {counts.median():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=(Path(__file__).parent.parent / "family_specific_features.csv"), help="Path to family_specific_features.csv")
    parser.add_argument("--cutoff", type=float, default=0.5, help="Threshold cutoff to consider a CC feature assigned")
    parser.add_argument("--out", type=Path, default=(Path(__file__).parent.parent / "plots" / "family_feature_hist.png"), help="Output PNG path")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")
    args = parser.parse_args()

    main(args.csv, args.cutoff, args.out, args.show)
