from collections import deque
from .contracts import PoseSample


class SequenceBuffer:
    """Một buffer cho một (nguồn, người); caller xóa buffer khi mất track."""

    def __init__(self, length=30, stride=5, max_gap=1.0):
        if length < 1 or stride < 1 or max_gap <= 0:
            raise ValueError("length/stride/max_gap phải dương")
        self.items = deque(maxlen=length)
        self.stride = stride
        self.max_gap = max_gap
        self.since_emit = 0

    def push(self, sample: PoseSample):
        if len(sample.joints) != 17 or len(sample.mask) != 17:
            raise ValueError("Cần đúng 17 khớp và 17 mask")
        if self.items:
            previous = self.items[-1]
            if (previous.source_id, previous.track_id) != (sample.source_id, sample.track_id):
                raise ValueError("Không trộn người/nguồn trong một chuỗi")
            if sample.timestamp <= previous.timestamp:
                raise ValueError("Timestamp phải tăng")
            if sample.timestamp - previous.timestamp > self.max_gap:
                self.items.clear()
                self.since_emit = 0
        self.items.append(sample)
        self.since_emit += 1
        if len(self.items) == self.items.maxlen and self.since_emit >= self.stride:
            self.since_emit = 0
            return tuple(self.items)
        return None
