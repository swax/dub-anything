"""Window corrections preserve speech that the director kept original."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import edges


class OriginalSpeechEdgesTests(unittest.TestCase):
    def apply(self, audio, lines, captions=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            part = root / "sample"
            part.mkdir()
            script = {"lines": lines, "captions": list(captions)}
            (part / "script.json").write_text(json.dumps(script), encoding="utf-8")
            (part / "segments.json").write_text('{"job_id": "sample"}', encoding="utf-8")
            with (
                patch.object(edges, "WORK", root),
                patch.object(edges, "JOBS", root),
                patch.object(edges, "decode_audio", return_value=audio),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                edges.check("sample", apply=True)
            return json.loads((part / "script.json").read_text(encoding="utf-8"))

    def test_soft_attack_moves_earlier_without_original_speech(self):
        audio = np.zeros(3 * edges.SR, dtype=np.float32)
        audio[int(1.05 * edges.SR) : int(1.8 * edges.SR)] = 0.1
        result = self.apply(audio, [{"start": 1.2, "end": 2.0, "text": "One moment."}])
        self.assertLess(result["lines"][0]["start"], 1.1)

    def test_soft_attack_stops_at_an_original_reply_even_with_empty_text(self):
        audio = np.zeros(3 * edges.SR, dtype=np.float32)
        audio[int(1.05 * edges.SR) : int(1.8 * edges.SR)] = 0.1
        original = {"start": 0.8, "end": 1.15, "text": ""}
        result = self.apply(audio, [{"start": 1.2, "end": 2.0, "text": "One moment."}], [original])
        self.assertEqual(result["lines"][0]["start"], 1.15)
        self.assertEqual(result["captions"], [original])

    def test_existing_overlap_does_not_grow(self):
        audio = np.zeros(3 * edges.SR, dtype=np.float32)
        audio[int(1.05 * edges.SR) : int(1.8 * edges.SR)] = 0.1
        result = self.apply(
            audio,
            [{"start": 1.2, "end": 2.0, "text": "One moment."}],
            [{"start": 1.1, "end": 1.4, "text": "Yes?"}],
        )
        self.assertEqual(result["lines"][0]["start"], 1.2)

    def test_release_extension_does_not_swallow_the_next_original_reply(self):
        audio = np.zeros(3 * edges.SR, dtype=np.float32)
        audio[int(0.5 * edges.SR) : int(1.7 * edges.SR)] = 0.1
        line = {"start": 0.5, "end": 1.5, "text": "Over there."}
        without_reply = self.apply(audio, [dict(line)])
        self.assertGreater(without_reply["lines"][0]["end"], 1.7)
        with_reply = self.apply(audio, [dict(line)], [{"start": 1.6, "end": 2.0, "text": "Thank you."}])
        self.assertEqual(with_reply["lines"][0]["end"], 1.5)

    def test_shared_boundary_does_not_move_through_original_speech(self):
        audio = np.zeros(3 * edges.SR, dtype=np.float32)
        audio[int(0.5 * edges.SR) : int(2.5 * edges.SR)] = 0.1
        audio[int(1.58 * edges.SR) : int(1.62 * edges.SR)] = 0
        lines = [
            {"start": 0.5, "end": 1.5, "text": "This way."},
            {"start": 1.5, "end": 2.5, "text": "All right."},
        ]
        without_reply = self.apply(audio, [dict(line) for line in lines])
        self.assertGreater(without_reply["lines"][0]["end"], 1.55)
        with_reply = self.apply(audio, [dict(line) for line in lines], [{"start": 1.55, "end": 1.65, "text": "Oh!"}])
        self.assertEqual(with_reply["lines"][0]["end"], 1.5)
        self.assertEqual(with_reply["lines"][1]["start"], 1.5)


if __name__ == "__main__":
    unittest.main()
