import sys
import time
import io
from threading import Event
from concurrent.futures import Future
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest
import storage

APP = str(Path(__file__).resolve().parents[1] / 'app.py')

class SteppedFeed:
    """Advance one camera read per UI tick; the real reader runs independently."""
    def __init__(self, capture):
        self.capture = capture
        self.frame = None
        self.last_seen = time.monotonic()

    def snapshot(self, after):
        ok, frame = self.capture.read()
        if ok and frame is not None and frame.size:
            self.frame, self.last_seen = frame, time.monotonic()
            return frame, self.last_seen, [(after + 1, int(self.last_seen * 1000), '')]
        return self.frame, self.last_seen, []

    def release(self):
        self.capture.release()


class DashboardTests(unittest.TestCase):
    def setUp(self):
        feed_patch = patch('camera_feed.CameraFeed', SteppedFeed)
        feed_patch.start()
        self.addCleanup(feed_patch.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.root_patch = patch.object(storage, 'ROOT', Path(self.tmp.name))
        self.root_patch.start()
        st.cache_resource.clear()

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def app(self):
        app = AppTest.from_file(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        return app

    def test_four_pages_and_empty_history(self):
        app = self.app()
        self.assertEqual(len(app.sidebar.radio[0].options), 4)
        for page in ['Giám sát trực tiếp', 'Xem lại sự cố', 'Thống kê an toàn', 'Tải video lên']:
            app.sidebar.radio[0].set_value(page).run()
            self.assertFalse(app.exception)
            self.assertTrue(any(t.value == page for t in app.title))

    def test_stopped_camera_clears_old_alert(self):
        app = self.app()
        app.session_state['last_result'] = dict(label='Fall', fall_score=.99, valid_frame_ratio=1.)
        app.session_state['alert_until'] = float('inf')
        app.session_state['camera_session'] = 'old'
        app.session_state['result_session'] = 'old'
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(any('aria-live="assertive"' in m.value for m in app.markdown))
        self.assertIsNone(app.session_state.get('last_result'))

    def test_upload_saves_result_and_retains_history(self):
        class Upload(io.BytesIO):
            name = 'sample.mp4'
            size = 4
        class Predictor:
            def predict_video(self, path):
                assert Path(path).read_bytes() == b'test'
                return dict(label='ADL', fall_score=.1, valid_frame_ratio=1.)
        with patch('streamlit.file_uploader', return_value=Upload(b'test')), patch.dict(sys.modules, {'predict':SimpleNamespace(FallPredictor=Predictor)}):
            app = self.app()
            app.sidebar.radio[0].set_value('Tải video lên').run()
            next(b for b in app.button if b.label == 'Phân tích và lưu kết quả').click().run()
            self.assertFalse(app.exception)
            self.assertEqual(len(storage.records()), 1)
            app.run()
            self.assertEqual(len(storage.records()), 1)
            self.assertTrue(Path(storage.records()[0]['video']).is_file())

    def test_review_and_statistics(self):
        storage.save_result(Path(self.tmp.name)/'missing.mp4', 'Camera trực tiếp',
                            dict(label='Fall', fall_score=.95, valid_frame_ratio=1.0))
        storage.save_result(Path(self.tmp.name)/'missing2.mp4', 'Video tải lên',
                            dict(label='ADL', fall_score=.1, valid_frame_ratio=1.0))
        app = self.app()
        app.sidebar.radio[0].set_value('Xem lại sự cố').run()
        self.assertFalse(app.exception)
        self.assertTrue(any('CẢNH BÁO TÉ NGÃ' in m.value for m in app.markdown))
        next(b for b in app.button if b.label == 'Đánh dấu đã kiểm tra').click().run()
        self.assertEqual(next(r for r in storage.records() if r['label']=='Fall')['reviewed'], 1)
        app.sidebar.radio[0].set_value('Thống kê an toàn').run()
        self.assertFalse(app.exception)
        self.assertEqual([m.value for m in app.metric], ['2','1','0','0'])
        self.assertEqual(len(app.dataframe), 1)

    def test_mark_many_reviewed(self):
        result = dict(label='Fall', fall_score=.9, valid_frame_ratio=1.)
        for name in ('a', 'b', 'c'):
            storage.save_result(storage.ROOT / f'{name}.mp4', name, result)
        ids = [r['id'] for r in storage.records()]
        storage.mark_many_reviewed(ids[:2])
        storage.mark_many_reviewed([])
        self.assertEqual(sorted(r['reviewed'] for r in storage.records()), [0, 1, 1])

    def test_delete_selected_video_and_refresh_history(self):
        result = dict(label='Fall', fall_score=.95, valid_frame_ratio=1.)
        target = storage.ROOT / 'selected.mp4'
        target.write_bytes(b'clip')
        storage.save_result(target, 'Webcam', result)
        storage.save_result(storage.ROOT / 'missing.mp4', 'Camera IP', result)
        app = self.app()
        app.sidebar.radio[0].set_value('Xem lại sự cố').run()
        identifier = next(r['id'] for r in storage.records() if r['source'] == 'Webcam')
        app.selectbox[1].set_value(identifier).run()
        next(b for b in app.button if b.label == 'Xác nhận xóa').click().run()
        self.assertFalse(app.exception)
        self.assertFalse(target.exists())
        self.assertEqual([r['source'] for r in storage.records()], ['Camera IP'])
        next(b for b in app.button if b.label == 'Xác nhận xóa').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.assertFalse(storage.records())

    def test_shared_video_and_failed_file_deletion(self):
        video = storage.ROOT / 'shared.mp4'
        video.write_bytes(b'clip')
        result = dict(label='ADL', fall_score=.1, valid_frame_ratio=1.)
        for source in ('one', 'two'):
            storage.save_result(video, source, result)
        first, second = storage.records()
        storage.delete_record(first['id'])
        self.assertTrue(video.exists())
        with patch.object(Path, 'unlink', side_effect=PermissionError), self.assertRaises(PermissionError):
            storage.delete_record(second['id'])
        self.assertEqual(storage.records(), [second])
        storage.delete_record(second['id'])
        self.assertFalse(video.exists())

    def test_live_alert_video_saved_and_camera_stopped(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def read(self): return True, np.zeros((64,64,3), dtype=np.uint8)
            def release(self): self.released = True
            def set(self, *args): return True
        class Predictor:
            def predict_video(self, path):
                self_path = Path(path)
                assert self_path.is_file() and self_path.stat().st_size > 0
                return dict(label='Fall', fall_score=.96, valid_frame_ratio=1.)
        capture = Capture()
        with patch('cv2.VideoCapture', return_value=capture), patch.dict(sys.modules, {'predict':SimpleNamespace(FallPredictor=Predictor)}):
            app = self.app()
            next(b for b in app.button if b.label=='Bắt đầu giám sát').click().run()
            self.assertFalse(any('aria-live="assertive"' in m.value for m in app.markdown))
            for _ in range(16):
                app.session_state['next_sample_at'] = 0
                app.run()
            future = app.session_state['analysis_future']
            if future is not None:
                future.result(timeout=10)
                app.run()
            self.assertFalse(app.exception)

            self.assertTrue(any('CẢNH BÁO TÉ NGÃ' in m.value for m in app.markdown))
            self.assertTrue(any('camera-alert' in m.value and 'aria-live="assertive"' in m.value for m in app.markdown))
            self.assertGreater(app.session_state['alert_until'], 0)
            rows = storage.records()
            self.assertTrue(rows)
            self.assertTrue(Path(rows[0]['video']).is_file())
            normal = Future()
            normal.set_result(dict(label='ADL', fall_score=.1, valid_frame_ratio=1.))
            app.session_state['analysis_future'] = normal
            app.session_state['analysis_session'] = app.session_state['camera_session']
            app.run()
            self.assertFalse(any('aria-live="assertive"' in m.value for m in app.markdown))
            # A completed result from a previous camera session must be ignored.
            stale = Future()
            stale.set_result(dict(label='Fall', fall_score=.99, valid_frame_ratio=1.))
            app.session_state['analysis_future'] = stale
            app.session_state['analysis_session'] = 'previous-session'
            app.run()
            self.assertFalse(any('aria-live="assertive"' in m.value for m in app.markdown))
            app.sidebar.radio[0].set_value('Xem lại sự cố').run()
            self.assertTrue(capture.released)
            self.assertFalse(app.exception)

    def test_camera_recovers_from_empty_reads_and_can_restart_after_timeout(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        camera = Mock()
        camera.read.side_effect = [(False, None), (True, frame), (True, frame[:0]),
                                   (True, frame), (False, None), (False, None), (True, frame)]
        with patch('cv2.VideoCapture', return_value=camera), patch.dict(
                sys.modules, {'predict': SimpleNamespace(FallPredictor=object)}):
            app = self.app()
            app.button[0].click().run()
            self.assertTrue(app.info)
            app.run()
            self.assertEqual(len(app.get('component_instance')), 1)
            samples = len(app.session_state['frames'])
            app.session_state['capture'].last_seen -= 1
            app.run()
            self.assertTrue(app.warning)
            self.assertEqual(len(app.get('component_instance')), 1)
            self.assertEqual(len(app.session_state['frames']), samples)
            app.run()
            self.assertFalse(app.warning)
            camera.release.assert_not_called()
            app.run()
            app.session_state['capture'].last_seen -= 6
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertFalse(app.button[0].disabled)
            camera.release.assert_called_once()
            app.button[0].click().run()
            self.assertTrue(app.session_state['running'])
            self.assertEqual(len(app.get('component_instance')), 1)
            app.button[1].click().run()

    def test_preview_keeps_updating_while_analysis_is_blocked(self):
        entered, release = Event(), Event()
        class Capture:
            reads = 0
            released = False
            def isOpened(self): return True
            def set(self, *args): return True
            def read(self):
                self.reads += 1
                return True, np.zeros((64,64,3), dtype=np.uint8)
            def release(self): self.released = True
        class Predictor:
            def predict_video(self, path):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError('Test did not release worker')
                return dict(label='Fall', fall_score=.96, valid_frame_ratio=1.)
        camera = Capture()
        with patch('cv2.VideoCapture', return_value=camera), patch.dict(sys.modules, {'predict':SimpleNamespace(FallPredictor=Predictor)}):
            app = self.app()
            next(b for b in app.button if b.label=='Bắt đầu giám sát').click().run()
            try:
                for _ in range(16):
                    app.session_state['next_sample_at'] = 0
                    app.run()
                self.assertTrue(entered.wait(5))
                future = app.session_state['analysis_future']
                reads = camera.reads
                for _ in range(5):
                    app.run()
                self.assertEqual(camera.reads, reads + 5)
                self.assertIs(app.session_state['analysis_future'], future)
                self.assertLessEqual(len(app.session_state['frames']), 16)
                self.assertFalse(app.exception)
                # Stop remains available even while inference is running.
                next(b for b in app.button if b.label=='Dừng giám sát').click().run()
                self.assertTrue(camera.released)
                self.assertFalse(app.session_state['running'])
            finally:
                release.set()
                pending = locals().get('future') or app.session_state.get('analysis_future')
                if pending is not None:
                    pending.result(timeout=10)

if __name__ == '__main__':
    unittest.main()
