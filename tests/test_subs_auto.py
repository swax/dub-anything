"""Read both subtitle styles with one OCR call, keeping their provenance."""

import pathlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import subs


class AutomaticTextTests(unittest.TestCase):
    def test_no_text_needs_only_one_ocr_call(self):
        ocr = Mock(return_value=types.SimpleNamespace(boxes=None))
        self.assertEqual(subs.read_auto_frame(ocr, np.zeros((100, 100, 3), dtype=np.uint8)), ([], []))
        ocr.assert_called_once()

    def test_subtitle_filter_uses_only_the_bottom_and_excludes_closed_captions(self):
        top = [[0, 10], [10, 10], [10, 20], [0, 20]]
        bottom = [[0, 80], [10, 80], [10, 90], [0, 90]]
        closed = [[20, 80], [30, 80], [30, 90], [20, 90]]
        result = types.SimpleNamespace(
            boxes=[top, bottom, closed], txts=["sign", "subtitle", "caption"], scores=[0.99, 0.99, 0.99]
        )
        with patch.object(subs, "caption_box", side_effect=lambda img, box: (box is closed, "white")):
            with patch.object(subs, "subtitle_text", return_value=(True, "white")):
                cc, subtitles = subs.read_auto_frame(lambda img: result, np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual([row[0] for row in cc], ["caption"])
        self.assertEqual([row[0] for row in subtitles], ["subtitle"])

    def test_same_text_detected_in_both_styles_is_not_duplicated(self):
        row = ("Hello there", "white", 10, 450, 100, 480)
        readings = [(0, [row]), (0.25, [row]), (0.5, [])]
        events = subs.automatic_events(readings, readings, 0.25)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "closed_caption")

    def test_outlined_subtitle_candidates_remain_distinct_evidence(self):
        row = ("A translated phrase", "white", 10, 450, 100, 480)
        readings = [(0, [row]), (0.25, [row]), (0.5, [])]
        events = subs.automatic_events([], readings, 0.25)
        self.assertEqual(events[0]["kind"], "subtitle")


if __name__ == "__main__":
    unittest.main()
