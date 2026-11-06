"""Background clip processing. No Streamlit calls from the worker thread."""
import base64
import uuid

import cv2
import numpy as np

import imageio_ffmpeg

from storage import save_result


def analyze_clip(frames, timestamps, predictor, root):
    root.mkdir(parents=True, exist_ok=True)
    path = root / (uuid.uuid4().hex + '.mp4')
    fps = max(1, min(60, (len(frames) - 1) / max(timestamps[-1] - timestamps[0], .001)))
    height, width = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(
        str(path), (width, height), fps=fps, codec='libx264',
        pix_fmt_in='bgr24', macro_block_size=1,
        output_params=['-preset', 'veryfast', '-threads', '1'],
    )
    try:
        try:
            writer.send(None)
            for frame in frames:
                writer.send(frame.tobytes())
        finally:
            writer.close()
        result = predictor.predict_video(path)
        save_result(path, 'Camera trực tiếp', result)
        return result
    except Exception:
        path.unlink(missing_ok=True)
        raise


def analyze_browser_clip(samples, predictor, root):
    if len(samples) != 16:
        raise ValueError('Đoạn webcam chưa đủ 16 khung hình.')
    frames, timestamps = [], []
    for stamp, encoded in samples:
        data = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('Khung hình webcam không hợp lệ.')
        frames.append(frame)
        timestamps.append(float(stamp))
    return analyze_clip(frames, timestamps, predictor, root)
