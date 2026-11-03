from pathlib import Path
import sqlite3
from datetime import datetime
import uuid

ROOT = Path(__file__).resolve().parent / 'data'


def connect():
    ROOT.mkdir(exist_ok=True)
    db = sqlite3.connect(ROOT / 'history.db')
    db.row_factory = sqlite3.Row
    db.execute('''CREATE TABLE IF NOT EXISTS analyses
        (id TEXT PRIMARY KEY, created TEXT, source TEXT, video TEXT,
         label TEXT, score REAL, coverage REAL, reviewed INTEGER DEFAULT 0)''')
    return db


def save_result(path, source, result):
    with connect() as db:
        db.execute('INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, 0)',
                   (uuid.uuid4().hex, datetime.now().isoformat(timespec='seconds'),
                    source, str(Path(path).resolve()), result['label'],
                    result['fall_score'], result['valid_frame_ratio']))


def records():
    with connect() as db:
        return [dict(row) for row in db.execute('SELECT * FROM analyses ORDER BY created DESC')]


def mark_reviewed(identifier):
    with connect() as db:
        db.execute('UPDATE analyses SET reviewed=1 WHERE id=?', (identifier,))
