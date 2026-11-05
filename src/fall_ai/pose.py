"""Thiết: nạp YOLO-Pose một lần và giữ track nhất quán."""


class PoseExtractor:
    def extract(self, frame_bgr, timestamp, source_id):
        """Trả list[PoseSample]; [] khi không có người hợp lệ."""
        raise NotImplementedError("Trích khớp, confidence/mask, bbox; không chọn lại người tùy frame")
