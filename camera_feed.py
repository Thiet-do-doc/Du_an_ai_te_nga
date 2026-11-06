"""Read the camera independently of Streamlit's browser refreshes."""
import base64
import time
from collections import deque
from threading import Event, Lock, Thread

import cv2


class CameraFeed:
    def __init__(self, capture):
        self._capture = capture
        self._lock = Lock()
        self._stop = Event()
        self._preview = deque(maxlen=4)
        self._frame = None
        self._last_seen = self._last_poll = time.monotonic()
        self._thread = Thread(target=self._read, daemon=True, name='camera-preview')
        self._thread.start()

    def _read(self):
        sequence = 0
        next_preview = 0
        try:
            while not self._stop.is_set():
                if time.monotonic() - self._last_poll > 30:
                    break  # Browser closed without pressing Stop.
                try:
                    ok, frame = self._capture.read()
                except cv2.error:
                    ok, frame = False, None
                if not ok or frame is None or not frame.size:
                    self._stop.wait(0.02)
                    continue
                now = time.monotonic()
                if now < next_preview:
                    continue
                # Drain faster devices but only encode up to 30 frames per second.
                next_preview = max(now, next_preview + 1 / 30)
                preview = frame
                if frame.shape[1] > 640:
                    height = max(1, round(frame.shape[0] * 640 / frame.shape[1]))
                    preview = cv2.resize(frame, (640, height), interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    continue
                sequence += 1
                packet = (sequence, round(now * 1000), base64.b64encode(jpeg).decode('ascii'))
                with self._lock:
                    self._preview.append(packet)
                    self._frame, self._last_seen = frame, now
        finally:
            self._capture.release()

    def snapshot(self, after):
        with self._lock:
            self._last_poll = time.monotonic()
            packets = [item for item in self._preview if item[0] > after]
            return self._frame, self._last_seen, packets

    def release(self):
        self._stop.set()
        # Never call release concurrently with a driver blocked inside read().
        self._thread.join(timeout=0.5)
