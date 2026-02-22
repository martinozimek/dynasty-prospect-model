"""
Apply a fitted dynasty prospect model to a draft class.

Produces a ranked prospect sheet for a given draft year with projected B2S scores.
Requires fit_model.py to have been run first to produce model/ artifacts.

Usage:
    python scripts/score_class.py --year 2026
    python scripts/score_class.py --year 2026 --position WR
    python scripts/score_class.py --year 2026 --output zap_2026.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_data_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a draft class with the fitted dynasty prospect model."
    )
    parser.add_argument("--year", type=int, required=True, help="Draft year to score.")
    parser.add_argument("--position", type=str, choices=["WR", "RB", "TE"])
    parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
    args = parser.parse_args()

    model_dir = Path(__file__).parent.parent / "model"
    data_dir = get_data_dir()

    positions = ["WR", "RB", "TE"] if args.position is None else [args.position]
    out_path = args.output or str(data_dir / f"scores_{args.year}.csv")

    # TODO: implement after fit_model.py is built
    # Steps:
    #   1. Build feature rows for --year draft class (same as build_training_set.py
    #      but without requiring B2S labels — these haven't played yet)
    #   2. Load fitted model from model/{position}_model.pkl
    #   3. Apply model to get projected B2S score
    #   4. Calibrate to 0-100 scale
    #   5. Output ranked CSV: player_name, position, draft_year, projected_b2s,
    #      percentile_rank, consensus_rank, draft_capital_score

    logger.info(
        "TODO: score_class.py not yet implemented — build fit_model.py first.\n"
        "  Target: score %s class of %d to %s",
        "/".join(positions), args.year, out_path,
    )


if __name__ == "__main__":
    main()
