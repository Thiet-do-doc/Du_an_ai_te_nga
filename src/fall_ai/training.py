"""Võ: phần huấn luyện offline, tách khỏi ứng dụng."""


def train_baseline(config_path):
    raise NotImplementedError("Random Forest: thống kê cùng chuỗi, dùng đúng split đã chốt")


def train_lstm(config_path):
    raise NotImplementedError("LSTM: chuỗi đã chuẩn hóa, chọn checkpoint bằng validation")


def evaluate(model_path, test_manifest):
    raise NotImplementedError("Precision/Recall/F1, confusion matrix, event recall, false alarms/h, delay")
