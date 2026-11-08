"""Kiểm tra hình học tư thế: chỉ coi là té ngã khi thân người thật sự nằm ngang.

Chỉ dùng numpy, không cần PyTorch. Đầu vào là mảng (32, 17, 3) gồm x, y chuẩn hóa
(0-1) và độ tin cậy của 17 điểm khớp theo chuẩn COCO (YOLO-Pose).
"""
import numpy as np

CONF = 0.3                # độ tin cậy tối thiểu của một điểm khớp
LYING_ANGLE = 50.0        # thân nghiêng so với phương thẳng đứng từ góc này là "nằm"
LYING_ASPECT = 1.1        # khung bao người rộng hơn cao từ tỉ lệ này là "nằm"
IMAGE_ASPECT = 4 / 3      # tỉ lệ khung hình giả định (rộng/cao) để bù tọa độ chuẩn hóa
SHOULDERS, HIPS = (5, 6), (11, 12)


def _frame_metrics(frame):
    xy, conf = frame[:, :2], frame[:, 2]
    angle = aspect = None
    s = [i for i in SHOULDERS if conf[i] > CONF]
    h = [i for i in HIPS if conf[i] > CONF]
    if s and h:
        top, bottom = xy[s].mean(axis=0), xy[h].mean(axis=0)
        dx = abs(bottom[0] - top[0]) * IMAGE_ASPECT
        dy = abs(bottom[1] - top[1])
        angle = float(np.degrees(np.arctan2(dx, dy + 1e-6)))
    seen = xy[conf > CONF]
    if len(seen) >= 5:
        width = (seen[:, 0].max() - seen[:, 0].min()) * IMAGE_ASPECT
        height = seen[:, 1].max() - seen[:, 1].min()
        if height > 1e-3:
            aspect = float(width / height)
    return angle, aspect


def posture_check(features):
    """Té ngã thật thì thân người phải nghiêng/nằm ngang ở ít nhất một thời điểm trong đoạn.

    Trả về dict: ok, max_angle (góc thân lớn nhất so với phương thẳng đứng, độ)
    và max_aspect (tỉ lệ rộng/cao lớn nhất của khung bao người).
    """
    frames = np.asarray(features, dtype=np.float32).reshape(-1, 17, 3)
    metrics = [_frame_metrics(f) for f in frames]
    angles = [m[0] for m in metrics if m[0] is not None]
    aspects = [m[1] for m in metrics if m[1] is not None]
    max_angle = max(angles) if angles else None
    max_aspect = max(aspects) if aspects else None
    lying = ((max_angle is not None and max_angle >= LYING_ANGLE)
             or (max_aspect is not None and max_aspect >= LYING_ASPECT))
    return dict(ok=bool(lying), max_angle=max_angle, max_aspect=max_aspect)
