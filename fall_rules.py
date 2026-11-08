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


# ---------------------------------------------------------------------------
# Phát hiện "đang nằm" nhanh, không phụ thuộc LSTM và không cần thấy chân.
# ---------------------------------------------------------------------------
STRICT_LYING_ANGLE = 60.0   # chặt hơn posture_check để giảm báo nhầm khi cúi nhặt đồ
STRICT_LYING_ASPECT = 1.2   # khung bao rộng/cao
UPRIGHT_ANGLE = 30.0        # thân nghiêng dưới góc này là "đang đứng/ngồi thẳng"
UPRIGHT_ASPECT = 0.9        # khung bao YOLO rộng/cao dưới mức này là "đang đứng/ngồi thẳng"
TAIL_STEPS = 12             # số bước cuối (khoảng 1/3 cửa sổ 32 bước) phải ở tư thế nằm
TAIL_RATIO = 0.8            # tỉ lệ bước cuối phải "nằm" để kết luận
HEAD_STEPS = 12             # số bước đầu dùng để kiểm tra trước đó người có đứng/ngồi thẳng không


def lying_check(features, box_aspects=None):
    """Kiểm tra người vừa chuyển từ đứng/ngồi sang nằm trong đoạn.

    features: (T, 17, 3) hoặc (T, 51). box_aspects: tùy chọn, độ dài T, là tỉ lệ
    rộng/cao của khung bao YOLO theo từng bước (0 hoặc NaN nếu không thấy người).
    Khung bao vẫn có khi YOLO mất khớp hông/đầu gối lúc người nằm, nên giúp bắt thêm ca.

    Trả về dict: lying (cuối đoạn đang nằm), was_upright (đầu đoạn còn đứng/ngồi thẳng),
    tail_ratio (tỉ lệ bước cuối được coi là nằm).
    """
    frames = np.asarray(features, dtype=np.float32).reshape(-1, 17, 3)
    n = len(frames)
    boxes = None
    if box_aspects is not None:
        boxes = np.nan_to_num(np.asarray(box_aspects, dtype=np.float32), nan=0.0)
        if len(boxes) != n:
            raise ValueError("box_aspects phải cùng độ dài với features")

    def info(i):
        angle, aspect = _frame_metrics(frames[i])
        box = float(boxes[i]) if boxes is not None and boxes[i] > 0 else None
        lying = ((angle is not None and angle >= STRICT_LYING_ANGLE)
                 or (aspect is not None and aspect >= STRICT_LYING_ASPECT)
                 or (box is not None and box >= STRICT_LYING_ASPECT))
        upright = ((angle is not None and angle < UPRIGHT_ANGLE)
                   or (box is not None and box < UPRIGHT_ASPECT))
        return lying, upright

    tail = [info(i)[0] for i in range(max(0, n - TAIL_STEPS), n)]
    head = [info(i)[1] for i in range(min(HEAD_STEPS, n))]
    tail_ratio = float(np.mean(tail)) if tail else 0.0
    return dict(lying=bool(tail_ratio >= TAIL_RATIO), was_upright=bool(any(head)),
                tail_ratio=tail_ratio)
