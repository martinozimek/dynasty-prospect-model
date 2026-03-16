
import sys
from pathlib import Path
import pandas as pd

data_dir = Path(r'C:\Users\Ozimek\Documents\Claude\FF\dynasty-prospect-model\data')
new_cols = [
    "best_deep_yprr", "best_deep_target_rate", "best_behind_los_rate",
    "best_slot_yprr", "best_slot_target_rate", "best_screen_rate",
    "best_man_yprr", "best_zone_yprr", "best_man_zone_delta",
]

for pos in ["WR", "RB", "TE"]:
    df = pd.read_csv(data_dir / f"training_{pos}.csv")
    print(f"\n=== {pos} (n={len(df)}) ===")
    for col in new_cols:
        if col in df.columns:
            n_filled = df[col].notna().sum()
            print(f"  {col}: {n_filled}/{len(df)} ({100*n_filled/len(df):.0f}%)")
        else:
            print(f"  {col}: MISSING from CSV")
