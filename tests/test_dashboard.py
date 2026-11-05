import sys
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import streamlit as st
from streamlit.testing.v1 import AppTest
import storage

APP = str(Path(__file__).resolve().parents[1] / 'app.py')

class DashboardTests(unittest.TestCase):
    def setUp(self):
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

    def test_live_alert_video_saved_and_camera_stopped(self):
        class Capture:
            released = False
            def isOpened(self): return True
            def read(self): return True, np.zeros((64,64,3), dtype=np.uint8)
            def release(self): self.released = True
        class Predictor:
            def predict_video(self, path):
                self_path = Path(path)
                assert self_path.is_file() and self_path.stat().st_size > 0
                return dict(label='Fall', fall_score=.96, valid_frame_ratio=1.)
        capture = Capture()
        with patch('cv2.VideoCapture', return_value=capture), patch.dict(sys.modules, {'predict':SimpleNamespace(FallPredictor=Predictor)}):
            app = self.app()
            next(b for b in app.button if b.label=='Bắt đầu giám sát').click().run()
            for _ in range(16):
                app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any('CẢNH BÁO TÉ NGÃ' in m.value for m in app.markdown))
            self.assertTrue(any('camera-alert' in m.value and 'aria-live="assertive"' in m.value for m in app.markdown))
            self.assertGreater(app.session_state['alert_until'], 0)
            rows = storage.records()
            self.assertTrue(rows)
            self.assertTrue(Path(rows[0]['video']).is_file())
            app.sidebar.radio[0].set_value('Xem lại sự cố').run()
            self.assertTrue(capture.released)
            self.assertFalse(app.exception)

if __name__ == '__main__':
    unittest.main()
