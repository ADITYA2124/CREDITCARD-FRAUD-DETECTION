import pandas as pd
from pathlib import Path
from collections import Counter

DATA_PATH = Path("data/raw/creditcard.csv")

EXPECTED_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Shape: {df.shape}")

    # column check
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        print(f"[ERROR] Missing: {missing}")
    else:
        print("All expected columns present ✅")

    # null check
    if df.isna().sum().sum() == 0:
        print("No missing values ✅")
    else:
        print("Missing values detected!")

    # stats
    print("\n[Time] describe():")
    print(df["Time"].describe())
    print("\n[Amount] describe():")
    print(df["Amount"].describe())

    # imbalance check
    counts = Counter(df["Class"])
    total = len(df)
    print(f"\nClass distribution: Legitimate={counts[0]}, Fraud={counts[1]} "
          f"({100*counts[1]/total:.4f}%)")

if __name__ == "__main__":
    main()
