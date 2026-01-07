import pandas as pd
from pathlib import Path

def main(csv_path: Path = None):
    csv_path = csv_path or (Path(__file__).parent.parent / "family_specific_features.csv")
    df = pd.read_csv(csv_path)

    # rows that would be triggered by a fixed cutoff of 0.5
    df_cut = df[df["threshold"] >= 0.5]

    n_cc_features_assigned = df_cut["dim"].nunique()
    n_families_assigned = df_cut["class"].nunique()
    families_multi = df_cut.groupby("class")["dim"].nunique()
    n_families_with_multiple_cc = (families_multi > 1).sum()

    print(f"cc_features_assigned: {n_cc_features_assigned}")
    print(f"families_assigned: {n_families_assigned}")
    print(f"families_with_multiple_cc_features: {n_families_with_multiple_cc}")

if __name__ == "__main__":
    main()