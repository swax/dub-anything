"""Kept takes must be scored using the current language QA rules, and the CPU hears in int8."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import lineqa


class KeptTakeTests(unittest.TestCase):
    def check_kept(self, want, current, earlier, stale_notes, stale_ratio):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            work = root / "clip"
            (work / "takes").mkdir(parents=True)
            (work / "segments.json").write_text(json.dumps({"job_id": "job"}), encoding="utf-8")
            (work / "script.json").write_text(
                json.dumps({"lines": [{"voice": "speaker_a", "text": want, "start": 0.0, "end": 1.0}]}),
                encoding="utf-8",
            )
            wav = root / "current.wav"
            wav.write_bytes(b"current")
            (work / "takes" / "earlier.wav").write_bytes(b"earlier")
            kept = {
                "line": [
                    {
                        "got": earlier,
                        "notes": stale_notes,
                        "ratio": stale_ratio,
                        "sha": "old",
                        "settings": {},
                        "file": "earlier.wav",
                    }
                ]
            }
            (work / "takes" / "kept.json").write_text(json.dumps(kept), encoding="utf-8")
            with (
                patch.object(lineqa, "WORK", root),
                patch.object(lineqa, "langs", return_value=("en", "yue")),
                patch.object(lineqa, "line_id", return_value="line"),
                patch.object(lineqa, "take_wav", return_value=wav),
                patch.object(lineqa.sf, "info", return_value=types.SimpleNamespace(duration=1.0)),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                # check() hears each new take through the ear it is given (lineqa.Ear).
                lineqa.check("clip", lambda wav, lang: current, False, False)
            return wav.read_bytes()

    def test_recognized_written_variant_can_restore_an_earlier_take(self):
        result = self.check_kept("而家我哋出發。", "而家我哋出門。", "依家我地出發。", ["match 0.10"], 0.1)
        self.assertEqual(result, b"earlier")

    def test_stale_passing_score_cannot_restore_a_wrong_number(self):
        result = self.check_kept("我哋有三本書。", "我哋有三本新書。", "我哋有四本書。", [], 1.0)
        self.assertEqual(result, b"current")


class EarTests(unittest.TestCase):
    """Whisper on the CPU (an AMD GPU, a Mac) hears in int8; on the GPU as before."""

    def ear(self, on_gpu):
        with patch.object(lineqa, "whisper_on_gpu", return_value=on_gpu):
            return lineqa.Ear()

    def test_cpu_hears_in_int8_and_gpu_as_before(self):
        for on_gpu, compute_type, tag in ((False, "int8", "v3-int8"), (True, "default", "v3")):
            with self.subTest(on_gpu=on_gpu):
                ear = self.ear(on_gpu)
                with (
                    patch.object(lineqa, "WhisperModel") as model,
                    patch.object(lineqa, "hold_gpu_for_whisper"),
                    patch.object(lineqa, "hear", return_value="Hello."),
                ):
                    self.assertEqual(ear("take.wav", "en"), "Hello.")
                self.assertEqual(model.call_args.kwargs["compute_type"], compute_type)
                self.assertEqual(ear.tag, tag)

    def test_a_take_heard_in_float32_is_heard_again_in_int8(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            work = root / "clip"
            work.mkdir()
            (work / "segments.json").write_text(json.dumps({"job_id": "job"}), encoding="utf-8")
            line = {"voice": "speaker_a", "text": "Hello.", "start": 0.0, "end": 1.0}
            (work / "script.json").write_text(json.dumps({"lines": [line]}), encoding="utf-8")
            wav = root / "take.wav"
            wav.write_bytes(b"take")
            # Heard before as GPU and float32 ears key it.
            heard = {f"v3:en:{lineqa.file_sha(wav)}": "Hello."}
            for on_gpu, hears in ((True, 0), (False, 1)):
                with self.subTest(on_gpu=on_gpu):
                    (work / "heard.json").write_text(json.dumps(heard), encoding="utf-8")
                    ear = self.ear(on_gpu)
                    with (
                        patch.object(lineqa, "WORK", root),
                        patch.object(lineqa, "langs", return_value=("ja", "en")),
                        patch.object(lineqa, "line_id", return_value="line"),
                        patch.object(lineqa, "take_wav", return_value=wav),
                        patch.object(lineqa, "WhisperModel"),
                        patch.object(lineqa, "hold_gpu_for_whisper"),
                        patch.object(lineqa, "hear", return_value="Hello.") as hear,
                        patch.object(lineqa.sf, "info", return_value=types.SimpleNamespace(duration=1.0)),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        lineqa.check("clip", ear, False, False)
                    self.assertEqual(hear.call_count, hears)
                    self.assertIn(f"{ear.tag}:en:{lineqa.file_sha(wav)}", json.loads((work / "heard.json").read_text()))
