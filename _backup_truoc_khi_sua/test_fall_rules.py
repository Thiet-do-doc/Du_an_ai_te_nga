import unittest
import numpy as np
from fall_rules import posture_check


def pose(shoulder, hip, extra=None):
    f = np.zeros((17, 3), dtype=np.float32)
    f[:, 2] = 1.0
    f[0, :2] = (shoulder[0], shoulder[1] - 0.1)
    f[5, :2] = f[6, :2] = shoulder
    f[11, :2] = f[12, :2] = hip
    f[13:17, :2] = (hip[0], hip[1] + 0.2)
    for i, xy in (extra or {}).items():
        f[i, :2] = xy
    return f


class PostureTests(unittest.TestCase):
    def test_standing_with_raised_hand_is_not_a_fall(self):
        frame = pose((0.5, 0.3), (0.5, 0.6), {9: (0.6, 0.05), 10: (0.4, 0.5)})
        self.assertFalse(posture_check(np.stack([frame] * 32))['ok'])

    def test_lying_down_is_a_fall(self):
        standing = pose((0.5, 0.3), (0.5, 0.6))
        lying = pose((0.2, 0.7), (0.55, 0.72))
        features = np.stack([standing] * 10 + [lying] * 22)
        result = posture_check(features)
        self.assertTrue(result['ok'])
        self.assertGreater(result['max_angle'], 60)

    def test_no_pose_is_not_a_fall(self):
        self.assertFalse(posture_check(np.zeros((32, 17, 3)))['ok'])


if __name__ == '__main__':
    unittest.main()
