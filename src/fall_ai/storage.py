import sqlite3
from pathlib import Path


class AlertStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(self.path) as db:
            db.execute('''CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY,
                created_at_utc TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                source_id TEXT NOT NULL, track_id INTEGER NOT NULL,
                source_timestamp REAL NOT NULL, score REAL NOT NULL,
                model_version TEXT NOT NULL, snapshot_path TEXT)''')

    def save(self, event, snapshot_path=None):
        with sqlite3.connect(self.path) as db:
            db.execute('''INSERT INTO alerts
                (source_id,track_id,source_timestamp,score,model_version,snapshot_path)
                VALUES (?,?,?,?,?,?)''',
                (event.source_id, event.track_id, event.timestamp, event.score,
                 event.model_version, snapshot_path))
