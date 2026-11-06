from pathlib import Path
import sqlite3
from datetime import datetime
import uuid
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parent / 'data'


@contextmanager
def connect():
    ROOT.mkdir(exist_ok=True)
    db = sqlite3.connect(ROOT / 'history.db')
    try:
        db.row_factory = sqlite3.Row
        with db:
            db.execute('''CREATE TABLE IF NOT EXISTS analyses
                (id TEXT PRIMARY KEY, created TEXT, source TEXT, video TEXT,
                 label TEXT, score REAL, coverage REAL, reviewed INTEGER DEFAULT 0)''')
            yield db
    finally:
        db.close()


def save_result(path, source, result):
    with connect() as db:
        db.execute('INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, 0)',
                   (uuid.uuid4().hex, datetime.now().isoformat(timespec='seconds'),
                    source, str(Path(path).resolve()), result['label'],
                    result['fall_score'], result['valid_frame_ratio']))


def records():
    with connect() as db:
        return [dict(row) for row in db.execute('SELECT * FROM analyses ORDER BY created DESC')]


def delete_record(identifier):
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT video FROM analyses WHERE id=?', (identifier,)).fetchone()
        if row is None:
            return
        db.execute('DELETE FROM analyses WHERE id=?', (identifier,))
        shared = db.execute('SELECT 1 FROM analyses WHERE video=? LIMIT 1', (row['video'],)).fetchone()
        path = Path(row['video'])
        if (not shared and path.resolve().parent == ROOT.resolve()
                and path.suffix.lower() in {'.mp4', '.avi', '.mov'}):
            path.unlink(missing_ok=True)


def mark_reviewed(identifier):
    with connect() as db:
        db.execute('UPDATE analyses SET reviewed=1 WHERE id=?', (identifier,))
