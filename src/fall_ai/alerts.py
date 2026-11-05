import math
from .contracts import AlertEvent, Prediction


class AlertManager:
    """Mốc thử nghiệm: xác nhận qua các cửa sổ liên tiếp, chờ trước khi báo lại.

    Cửa sổ chồng lấn không độc lập; tham số phải được đánh giá trên video liên tục.
    """
    def __init__(self, threshold=.8, confirm_windows=3, cooldown=10., max_gap=1.):
        if not 0 <= threshold <= 1 or confirm_windows < 1 or cooldown < 0 or max_gap <= 0:
            raise ValueError("Cấu hình cảnh báo không hợp lệ")
        self.threshold, self.confirm_windows = threshold, confirm_windows
        self.cooldown, self.max_gap = cooldown, max_gap
        self.state = {}

    def reset(self, source_id, track_id):
        self.state.pop((source_id, track_id), None)

    def update(self, p: Prediction, model_version: str):
        if not math.isfinite(p.score) or not 0 <= p.score <= 1 or not math.isfinite(p.timestamp):
            raise ValueError("Điểm/timestamp không hợp lệ")
        key = (p.source_id, p.track_id)
        count, last_seen, last_alert = self.state.get(key, (0, None, -math.inf))
        if last_seen is not None:
            if p.timestamp <= last_seen:
                raise ValueError("Timestamp phải tăng; reset khi mở lại nguồn")
            if p.timestamp - last_seen > self.max_gap:
                count = 0
        count = count + 1 if p.score >= self.threshold else 0
        event = None
        if count >= self.confirm_windows and p.timestamp - last_alert >= self.cooldown:
            event = AlertEvent(p.source_id, p.track_id, p.timestamp, p.score, model_version)
            last_alert, count = p.timestamp, 0
        self.state[key] = (count, p.timestamp, last_alert)
        return event
