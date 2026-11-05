from typing import Protocol
from .contracts import PoseSample, Prediction


class Predictor(Protocol):
    def predict(self, sequence: tuple[PoseSample, ...]) -> Prediction: ...


class MockPredictor:
    """CHỈ DEMO: trả điểm giả, không suy luận từ hình ảnh hoặc chuyển động."""
    model_version = "MOCK-NOT-A-TRAINED-MODEL"

    def predict(self, sequence):
        last = sequence[-1]
        return Prediction(last.source_id, last.track_id, last.timestamp, 0.95)


class TrainedPredictor:
    def __init__(self, model_path):
        raise NotImplementedError("Võ: nạp model + metadata; không dùng Mock cho camera thật")

    def predict(self, sequence):
        raise NotImplementedError("Võ: vector hóa theo hợp đồng, chạy model, trả Prediction")
