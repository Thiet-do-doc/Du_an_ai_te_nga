from pathlib import Path
from threading import RLock

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from model import load_lstm_model
from fall_rules import posture_check, lying_check

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 32
# Ngưỡng báo té ngã (0-1). Tăng lên nếu vẫn báo nhầm, giảm xuống nếu bỏ sót.
FALL_THRESHOLD = 0.9
# Cần thấy phần thân dưới (hông, đầu gối) thì mới kết luận, tránh nhầm khi chỉ giơ tay.
MIN_LOWER_BODY_RATIO = 0.6


def limit_cpu_threads(_predictor=None):
    # YOLO resets PyTorch's thread count when it initializes the CPU backend.
    # Keep compute capacity available for the camera and the browser.
    if torch.get_num_threads() > 2:
        torch.set_num_threads(2)


class FallPredictor:
    def __init__(self):
        self._lock = RLock()
        self.pose_model = YOLO(
            str(BASE_DIR / "models" / "yolo11n-pose.pt")
        )
        if DEVICE.type == "cpu":
            limit_cpu_threads()
            self.pose_model.add_callback("on_predict_start", limit_cpu_threads)
        self.lstm_model = load_lstm_model(
            BASE_DIR / "models" / "best_lstm.pt",
            DEVICE
        )

    def extract_keypoints(self, video_path):
        # Giữ nguyên hàm cũ cho mã đang dùng; chỉ trả về keypoint.
        return self.extract_pose(video_path)[0]

    def extract_pose(self, video_path):
        """Trả về (keypoint (32, 51), tỉ lệ rộng/cao khung bao người (32,))."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Không thể mở video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < 2:
            cap.release()
            raise ValueError("Video không có đủ frame.")

        indices = np.linspace(
            0, total_frames - 1, SEQ_LEN
        ).astype(int)

        # Short live clips repeat indices to fill the LSTM sequence. Run pose
        # inference once per distinct frame, then restore the requested order.
        unique_indices, positions = np.unique(indices, return_inverse=True)
        sequence = []
        box_aspects = []
        next_index = 0

        try:
            for idx in unique_indices:
                if idx != next_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                next_index = idx + 1
                success, frame = cap.read()

                if not success:
                    raise ValueError(
                        f"Không đọc được frame số {idx}"
                    )

                result = self.pose_model.predict(
                    frame,
                    imgsz=640,
                    conf=0.25,
                    device=0 if DEVICE.type == "cuda" else "cpu",
                    verbose=False
                )[0]

                points = np.zeros((17, 3), dtype=np.float32)
                box_aspect = 0.0  # 0 nghĩa là không thấy người

                if (
                    result.keypoints is not None
                    and len(result.keypoints) > 0
                ):
                    areas = (
                        result.boxes.xywh[:, 2]
                        * result.boxes.xywh[:, 3]
                    )

                    person_idx = int(areas.argmax().item())

                    xy = (
                        result.keypoints.xyn[person_idx]
                        .cpu()
                        .numpy()
                    )

                    confidence = result.keypoints.conf

                    scores = (
                        confidence[person_idx].cpu().numpy()
                        if confidence is not None
                        else np.ones(17, dtype=np.float32)
                    )

                    points[:, :2] = xy
                    points[:, 2] = scores
                    # Khung bao vẫn có khi YOLO mất khớp lúc người nằm.
                    box = result.boxes.xywh[person_idx]
                    box_aspect = float(box[2]) / max(float(box[3]), 1.0)

                sequence.append(points.flatten())
                box_aspects.append(box_aspect)

        finally:
            cap.release()

        features = np.asarray(sequence, dtype=np.float32)
        aspects = np.asarray(box_aspects, dtype=np.float32)

        return features[positions], aspects[positions]

    def predict_video(self, video_path):
        # The cached predictor can be used by background cameras and uploads.
        with self._lock:
            return self._predict_video(video_path)

    def _predict_video(self, video_path):
        features, box_aspects = self.extract_pose(video_path)

        if features.shape != (32, 51):
            raise ValueError("Dữ liệu keypoint không hợp lệ.")

        if not np.isfinite(features).all():
            raise ValueError("Keypoint chứa NaN hoặc Inf.")

        scores = features.reshape(32, 17, 3)[:, :, 2]
        valid_frame_ratio = float(
            ((scores > 0.25).any(axis=1) | (box_aspects > 0)).mean()
        )

        if valid_frame_ratio < 0.8:
            return {
                "label": "Không đủ dữ liệu",
                "fall_score": None,
                "valid_frame_ratio": valid_frame_ratio
            }

        lower = scores[:, 11:15] > 0.3
        lower_ok = float((lower.sum(axis=1) >= 2).mean())
        pose = lying_check(features, box_aspects)

        # Người nằm thì YOLO hay mất hông/đầu gối; khi đã thấy rõ là đang nằm
        # thì không loại đoạn này vì thiếu thân dưới.
        if lower_ok < MIN_LOWER_BODY_RATIO and not pose["lying"]:
            return {
                "label": "Không đủ dữ liệu",
                "fall_score": None,
                "valid_frame_ratio": valid_frame_ratio
            }

        tensor = torch.from_numpy(
            features
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = self.lstm_model(tensor)
            fall_score = torch.sigmoid(logits).item()

        is_fall = fall_score >= FALL_THRESHOLD and posture_check(features)["ok"]

        # "Lying": vừa chuyển từ đứng/ngồi sang nằm nhưng LSTM chưa đủ chắc là té ngã.
        # Chỉ cảnh báo mức "cần kiểm tra", không phải té ngã đã xác nhận.
        if is_fall:
            label = "Fall"
        elif pose["lying"] and pose["was_upright"]:
            label = "Lying"
        else:
            label = "ADL"

        return {
            "label": label,
            "fall_score": fall_score,
            "valid_frame_ratio": valid_frame_ratio,
            "lying": pose["lying"],
            "was_upright": pose["was_upright"]
        }
