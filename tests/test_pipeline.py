import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path
from src.fall_ai.contracts import PoseSample, Prediction
from src.fall_ai.sequence import SequenceBuffer
from src.fall_ai.alerts import AlertManager
from src.fall_ai.storage import AlertStore


def sample(t, track=1):
    return PoseSample('test', track, t, ((0., 0., 1.),) * 17,
                      (True,) * 17, (.5, .5, .2, .8))


class PipelineTests(unittest.TestCase):
    def test_sequence_waits_and_rejects_other_person(self):
        buf = SequenceBuffer(3, 1)
        self.assertIsNone(buf.push(sample(0)))
        self.assertIsNone(buf.push(sample(.1)))
        self.assertEqual(len(buf.push(sample(.2))), 3)
        with self.assertRaises(ValueError):
            buf.push(sample(.3, track=2))

    def test_sequence_clears_after_gap(self):
        buf = SequenceBuffer(2, 1, max_gap=1)
        buf.push(sample(0))
        self.assertIsNone(buf.push(sample(2)))

    def test_low_score_breaks_confirmation_and_cooldown(self):
        a = AlertManager(confirm_windows=2, cooldown=10)
        def push(t, score):
            return a.update(Prediction('test', 1, t, score), 'mock')
        self.assertIsNone(push(0, .95))
        self.assertIsNone(push(.1, .2))
        self.assertIsNone(push(.2, .95))
        self.assertIsNotNone(push(.3, .95))
        self.assertIsNone(push(.4, .95))
        self.assertIsNone(push(.5, .95))

    def test_confirmation_not_shared_between_people_or_across_gap(self):
        a = AlertManager(confirm_windows=2, max_gap=1)
        self.assertIsNone(a.update(Prediction('test', 1, 0, .95), 'mock'))
        self.assertIsNone(a.update(Prediction('test', 2, .1, .95), 'mock'))
        self.assertIsNone(a.update(Prediction('test', 1, 3, .95), 'mock'))

    def test_event_persists_to_database(self):
        a = AlertManager(confirm_windows=1)
        event = a.update(Prediction('test', 1, 0, .9), 'mock')
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / 'alerts.sqlite3'
            AlertStore(db_path).save(event)
            with closing(sqlite3.connect(db_path)) as db:
                row = db.execute('SELECT source_id,score,model_version FROM alerts').fetchone()
            self.assertEqual(row, ('test', .9, 'mock'))


if __name__ == '__main__':
    unittest.main()
