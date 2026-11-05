import time
from collections import deque

import cv2
import numpy as np
import torch

from predict import DEVICE


class WebcamAnalyzer:
    def __init__(self, predictor):
        self.predictor = predictor
        self.buffer = deque()

        self.window_seconds = 1.9
        self.last_prediction_time = 0

        self.fall_score = None
        self.valid_ratio = 0.0
        self.label = "Đang thu thập dữ liệu"

        self.fall_streak = 0
        self.last_alert_time = -1000
        self.alert_active = False

    def process_frame(self, frame):
        # 1. Chạy YOLO-Pose
        result = self.predictor.pose_model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            device=0 if DEVICE.type == "cuda" else "cpu",
            verbose=False
        )[0]

        # 2. Tạo dữ liệu 17 keypoint x 3 giá trị
        points = np.zeros((17, 3), dtype=np.float32)

        if (
            result.boxes is not None
            and len(result.boxes) > 0
            and result.keypoints is not None
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

        # 3. Lưu các đặc trưng gần nhất
        now = time.monotonic()
        self.buffer.append((now, points.flatten()))

        while (
            self.buffer
            and now - self.buffer[0][0] > self.window_seconds
        ):
            self.buffer.popleft()

        # 4. Chỉ dự đoán khi đã thu đủ khoảng thời gian
        if (
            len(self.buffer) >= 8
            and now - self.buffer[0][0] >= 1.6
            and now - self.last_prediction_time >= 0.4
        ):
            self.last_prediction_time = now

            features = np.asarray(
                [item[1] for item in self.buffer],
                dtype=np.float32
            )

            # Lấy mẫu đều thành 32 bước thời gian
            indices = np.linspace(
                0,
                len(features) - 1,
                32
            ).astype(int)

            features = features[indices]

            scores = features.reshape(
                32, 17, 3
            )[:, :, 2]

            self.valid_ratio = float(
                (scores > 0.25).any(axis=1).mean()
            )

            # 5. Kiểm tra dữ liệu tư thế
            if self.valid_ratio < 0.8:
                self.fall_score = None
                self.label = "Không đủ dữ liệu tư thế"
                self.fall_streak = 0
                self.alert_active = False

            else:
                # 6. Dự đoán bằng LSTM
                x = torch.from_numpy(
                    features.copy()
                ).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = self.predictor.lstm_model(x)
                    probability = torch.sigmoid(
                        logits
                    ).item()

                self.fall_score = probability

                if probability >= 0.5:
                    self.label = "Fall"
                    self.fall_streak += 1
                else:
                    self.label = "ADL"
                    self.fall_streak = 0

                # Cảnh báo sau 3 dự đoán Fall liên tiếp
                # Giới hạn cảnh báo lặp lại
                if (
                    self.fall_streak >= 3
                    and now - self.last_alert_time >= 10
                ):
                    self.alert_active = True
                    self.last_alert_time = now

                if self.fall_streak == 0:
                    self.alert_active = False

        # 7. Vẽ khung xương
        annotated = result.plot()

        status = self.label

        if self.fall_score is not None:
            status += f" | Fall: {self.fall_score * 100:.1f}%"

        color = (
            (0, 0, 255)
            if self.alert_active
            else (0, 200, 0)
        )

        cv2.rectangle(
            annotated,
            (5, 5),
            (min(annotated.shape[1] - 5, 750), 65),
            (20, 20, 20),
            -1
        )

        cv2.putText(
            annotated,
            status,
            (15, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

        return cv2.cvtColor(
            annotated,
            cv2.COLOR_BGR2RGB
        )