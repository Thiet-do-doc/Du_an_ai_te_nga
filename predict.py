from pathlib import Path
from threading import RLock

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from model import load_lstm_model

BASE_DIR = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 32


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

                sequence.append(points.flatten())

        finally:
            cap.release()

        features = np.asarray(sequence, dtype=np.float32)

        return features[positions]

    def predict_video(self, video_path):
        # The cached predictor can be used by background cameras and uploads.
        with self._lock:
            return self._predict_video(video_path)

    def _predict_video(self, video_path):
        features = self.extract_keypoints(video_path)

        if features.shape != (32, 51):
            raise ValueError("Dữ liệu keypoint không hợp lệ.")

        if not np.isfinite(features).all():
            raise ValueError("Keypoint chứa NaN hoặc Inf.")

        scores = features.reshape(32, 17, 3)[:, :, 2]
        valid_frame_ratio = float(
            (scores > 0.25).any(axis=1).mean()
        )

        if valid_frame_ratio < 0.8:
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

        return {
            "label": "Fall" if fall_score >= 0.5 else "ADL",
            "fall_score": fall_score,
            "valid_frame_ratio": valid_frame_ratio
        }
