import unittest
import numpy as np
from fall_rules import posture_check, lying_check


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


class LyingTests(unittest.TestCase):
    standing = pose((0.5, 0.3), (0.5, 0.6))
    lying = pose((0.2, 0.7), (0.55, 0.72))

    def test_stand_then_lie_is_detected(self):
        result = lying_check(np.stack([self.standing] * 12 + [self.lying] * 20))
        self.assertTrue(result['lying'])
        self.assertTrue(result['was_upright'])

    def test_already_lying_has_no_transition(self):
        result = lying_check(np.stack([self.lying] * 32))
        self.assertTrue(result['lying'])
        self.assertFalse(result['was_upright'])

    def test_standing_is_not_lying(self):
        self.assertFalse(lying_check(np.stack([self.standing] * 32))['lying'])

    def test_box_aspect_detects_lying_when_keypoints_lost(self):
        lost = np.zeros((32, 17, 3), dtype=np.float32)
        boxes = np.array([0.4] * 12 + [2.0] * 20)
        result = lying_check(lost, boxes)
        self.assertTrue(result['lying'])
        self.assertTrue(result['was_upright'])

    def test_box_length_must_match(self):
        with self.assertRaises(ValueError):
            lying_check(np.stack([self.standing] * 32), [1.0] * 5)


if __name__ == '__main__':
    unittest.main()
