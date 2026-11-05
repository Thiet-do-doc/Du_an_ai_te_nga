"""Mạnh: trả frame BGR kèm timestamp; không import OpenCV ở phần dùng chung."""


def iter_frames(source, sample_fps=15):
    raise NotImplementedError("Đọc video/webcam, lấy mẫu theo thời gian; release camera khi kết thúc")
