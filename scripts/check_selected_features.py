
import json
from pathlib import Path

meta = json.loads((Path(r'C:\Users\Ozimek\Documents\Claude\FF\dynasty-prospect-model\models\metadata.json')).read_text())
new_splits = {
    "best_deep_yprr", "best_deep_target_rate", "best_behind_los_rate",
    "best_slot_yprr", "best_slot_target_rate", "best_screen_rate",
    "best_man_yprr", "best_zone_yprr", "best_man_zone_delta",
    "deep_target_x_capital", "deep_yprr_x_capital",
    "slot_rate_x_capital", "man_delta_x_capital",
}
for pos in ['WR', 'RB', 'TE']:
    feats = meta[pos]['features']
    new_selected = [f for f in feats if f in new_splits]
    print(f"{pos} ({len(feats)} feats): {feats}")
    print(f"  NEW split features selected: {new_selected}")
    print()
