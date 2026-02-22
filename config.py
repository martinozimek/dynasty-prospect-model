"""
Configuration loader for dynasty-prospect-model.

Reads .env from the project root and exposes path helpers.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_ROOT = Path(__file__).parent


def get_cfb_db_path() -> str:
    """Return the path to the cfb-prospect-db SQLite file."""
    raw = os.environ.get("CFB_DB_PATH", "../cfb-prospect-db/ff.db")
    return str((_ROOT / raw).resolve())


def get_nfl_db_path() -> str:
    """Return the path to the nfl-fantasy-db SQLite file."""
    raw = os.environ.get("NFL_DB_PATH", "../nfl-fantasy-db/nfl.db")
    return str((_ROOT / raw).resolve())


def get_data_dir() -> Path:
    """Return the output data directory, creating it if needed."""
    raw = os.environ.get("DATA_DIR", "data")
    d = (_ROOT / raw).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d
