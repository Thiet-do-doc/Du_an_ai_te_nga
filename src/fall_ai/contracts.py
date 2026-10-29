"""Giao diện chung. Thay đổi phải được Thiết, Võ và Thiện thống nhất."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PoseSample:
    source_id: str
    track_id: int
    timestamp: float  # giây từ đầu nguồn, tăng nghiêm ngặt
    joints: tuple[tuple[float, float, float], ...]  # 17 x (x_norm,y_norm,confidence)
    mask: tuple[bool, ...]  # 17; False là khớp thiếu
    bbox: tuple[float, float, float, float]  # cx/W,cy/H,w/W,h/H


@dataclass(frozen=True)
class Prediction:
    source_id: str
    track_id: int
    timestamp: float  # thời điểm kết thúc cửa sổ
    score: float  # điểm mô hình trong [0,1], chưa mặc định là xác suất đã hiệu chỉnh


@dataclass(frozen=True)
class AlertEvent:
    source_id: str
    track_id: int
    timestamp: float
    score: float
    model_version: str
