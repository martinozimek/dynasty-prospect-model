
"""Quick check: how many training rows remain if we restrict to 2014+ draft classes."""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(r'C:\Users\Ozimek\Documents\Claude\FF\dynasty-prospect-model')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze import load_training
from config import get_data_dir

data_dir = get_data_dir()

for pos in ['WR', 'RB', 'TE']:
    df = load_training(pos, data_dir)
    df_all = df[df['b2s_score'].notna()]
    df_2014 = df_all[df_all['draft_year'] >= 2014]
    print(f"{pos}: full={len(df_all)}  2014+={len(df_2014)}  lost={len(df_all)-len(df_2014)}")
    # Per-year counts for 2014+
    vc = df_2014['draft_year'].value_counts().sort_index()
    print(f"  Per-year (2014+): {dict(vc)}")
    print()
