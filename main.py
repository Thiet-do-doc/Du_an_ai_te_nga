"""Demo tích hợp bằng dữ liệu giả, chạy trực tiếp từ mọi thư mục."""
import argparse
import json
from pathlib import Path
from src.fall_ai.contracts import PoseSample
from src.fall_ai.sequence import SequenceBuffer
from src.fall_ai.predictor import MockPredictor
from src.fall_ai.alerts import AlertManager
from src.fall_ai.storage import AlertStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Chạy bằng dữ liệu giả")
    args = parser.parse_args()
    if not args.demo:
        parser.error("Hiện chỉ hỗ trợ --demo; camera và mô hình thật chưa triển khai")
    root = Path(__file__).resolve().parent
    cfg = json.loads((root / "config/default.json").read_text())
    buffer = SequenceBuffer(cfg["sequence_length"], cfg["sequence_stride"],
                            cfg["max_observation_gap_seconds"])
    predictor = MockPredictor()
    alerts = AlertManager(cfg["threshold"], cfg["confirm_windows"],
                          cfg["cooldown_seconds"], cfg["max_observation_gap_seconds"])
    store = AlertStore(root / cfg["database"])
    print("DEMO GIẢ LẬP — KHÔNG CÓ MÔ HÌNH AI ĐÃ HUẤN LUYỆN")
    n = 0
    for i in range(90):
        pose = PoseSample("mock-camera", 1, i / cfg["sample_fps"],
                          ((.5, .5, 1.),) * 17, (True,) * 17, (.5, .5, .3, .7))
        sequence = buffer.push(pose)
        if sequence is None:
            continue
        result = predictor.predict(sequence)
        event = alerts.update(result, predictor.model_version)
        if event:
            store.save(event)
            n += 1
            print(f"Cảnh báo GIẢ tại {event.timestamp:.2f}s, score={event.score}")
    print(f"Đã lưu {n} sự kiện giả vào {root / cfg['database']}")


if __name__ == "__main__":
    main()
