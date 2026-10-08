"""SQLite storage for recorded game runs."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "results" / "games.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    policy_x TEXT NOT NULL,
    policy_o TEXT NOT NULL,
    requested_games INTEGER NOT NULL CHECK (requested_games > 0),
    seed INTEGER,
    status TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'completed', 'abandoned', 'interrupted', 'failed'))
);

CREATE TABLE IF NOT EXISTS games (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    game_number INTEGER NOT NULL CHECK (game_number > 0),
    winner INTEGER NOT NULL CHECK (winner IN (-1, 0, 1)),
    PRIMARY KEY (run_id, game_number)
);

CREATE TABLE IF NOT EXISTS run_models (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    seat TEXT NOT NULL CHECK (seat IN ('x', 'o')),
    experiment_run_id TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    checkpoint_path TEXT NOT NULL,
    checkpoint_hash TEXT NOT NULL,
    selection TEXT NOT NULL,
    protocol_id TEXT,
    PRIMARY KEY (run_id, seat)
);
"""


def connect_database(path=DEFAULT_DB_PATH):
    """Create the parent directory and initialise the database schema."""
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
    except Exception:
        connection.close()
        raise
    return connection
