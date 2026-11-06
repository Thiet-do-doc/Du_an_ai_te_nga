import base64
from threading import Event
import time
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from camera_feed import CameraFeed


class CameraFeedTests(unittest.TestCase):
    def test_slow_consumer_gets_recent_decodable_frames_and_recovers_after_drops(self):
        capture = Mock()
        stalled = Event()
        frame = np.zeros((48, 64, 3), np.uint8)

        def read():
            time.sleep(0.012)
            if stalled.is_set():
                return False, None
            return True, frame

        capture.read.side_effect = read
        feed = CameraFeed(capture)
        try:
            # Simulate the UI being busy while the camera continues reading.
            deadline = time.monotonic() + 3
            while capture.read.call_count < 24 and time.monotonic() < deadline:
                time.sleep(0.02)
            raw, seen, batch = feed.snapshot(0)
            self.assertIs(raw, frame)
            self.assertEqual(len(batch), 4)
            self.assertGreater(batch[0][0], 1)
            self.assertGreater(seen, 0)
            jpg = np.frombuffer(base64.b64decode(batch[-1][2]), np.uint8)
            self.assertEqual(cv2.imdecode(jpg, cv2.IMREAD_COLOR).shape, frame.shape)
            stalled.set()
            time.sleep(0.10)
            raw, seen, batch = feed.snapshot(0)
            time.sleep(0.10)
            self.assertEqual(feed.snapshot(batch[-1][0])[2], [])
            self.assertIs(feed.snapshot(0)[0], raw)
            stalled.clear()
            deadline = time.monotonic() + 2
            while not feed.snapshot(batch[-1][0])[2] and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertGreater(feed.snapshot(0)[1], seen)
        finally:
            feed.release()
        capture.release.assert_called_once()
