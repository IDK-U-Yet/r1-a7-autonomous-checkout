# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import unittest

import cv2
import numpy as np

from teleimager.image_client import (
    attach_jpeg_frame_metadata,
    extract_jpeg_frame_metadata,
)


class TeleimagerMetadataTest(unittest.TestCase):
    def test_metadata_round_trip_remains_valid_jpeg(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)

        framed = attach_jpeg_frame_metadata(
            encoded.tobytes(),
            frame_sequence=42,
            capture_monotonic_ns=123456,
            capture_wall_time_ns=789012,
        )

        self.assertEqual(
            extract_jpeg_frame_metadata(framed),
            {
                "frame_sequence": 42,
                "capture_monotonic_ns": 123456,
                "capture_wall_time_ns": 789012,
            },
        )
        decoded = cv2.imdecode(np.frombuffer(framed, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, image.shape)

    def test_legacy_jpeg_has_no_metadata(self):
        ok, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(ok)
        self.assertIsNone(extract_jpeg_frame_metadata(encoded.tobytes()))


if __name__ == "__main__":
    unittest.main()
