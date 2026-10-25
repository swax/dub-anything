"""Partial rewording keeps the renderer's full-cast reference decisions."""

import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import core


class FitReferenceCastTests(unittest.TestCase):
    def test_reworded_line_reuses_full_cast_rejection_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            work = root / "sample"
            work.mkdir()
            (root / "voices").mkdir()
            job = root / "jobs" / "sample"
            job.mkdir(parents=True)
            sr = 16000
            audio = np.full(2 * sr, 0.1)
            audio[sr // 2 : sr] = 0.4
            sf.write(job / "vocals.wav", audio, sr)
            voices = {
                "speaker1": {"from": "sample", "clips": [[0, 0.5]], "ref_text": "First voice."},
                "speaker2": {"from": "sample", "clips": [[0.5, 1]], "ref_text": "Second voice."},
            }
            (root / "voices.json").write_text(json.dumps(voices), encoding="utf-8")
            for key, level in (("speaker1", 0.1), ("speaker2", 0.4)):
                sf.write(root / "voices" / f"{key}.wav", np.full(sr // 2, level), sr)
            (work / "segments.json").write_text('{"job_id": "sample"}', encoding="utf-8")
            lines = [
                {"start": 0.1, "end": 0.4, "voice": "speaker1", "ja": "Example source.", "text": "First line."},
                {"start": 1.1, "end": 1.9, "voice": "speaker2", "ja": "Example source.", "text": "Second line."},
            ]
            mode = {"line_ref": True, "dips": True, "cast_lines": True, "fill": True}
            model = Mock(return_value=object())
            draft = types.SimpleNamespace(
                SR=sr,
                decode_audio=lambda path, sampling_rate: sf.read(path, dtype="float32")[0],
                VadOptions=lambda **kwargs: kwargs,
                get_speech_timestamps=lambda audio, options: [],
                whisper_on_gpu=lambda: False,
                WhisperModel=model,
                transcribe=lambda *args, **kwargs: [{"said": "Example source."}],
                embedder=lambda: lambda audio: np.array([1.0, 0.0]) if audio.mean() < 0.25 else np.array([0.0, 1.0]),
                cast_centroids=lambda *args, **kwargs: {},
            )
            register = types.SimpleNamespace(cer=lambda a, b: float(a != b))
            with (
                patch.object(core, "WORK", root),
                patch.object(core, "JOBS", root / "jobs"),
                patch.object(core, "langs", return_value=("ja", "en")),
                patch.dict(sys.modules, {"draft": draft, "register": register}),
            ):
                ids = [core.line_id(line) for line in lines]
                # Generation compares both speakers: the second window belongs to
                # speaker1, so it must fall back to speaker2's base reference.
                generated = core.line_refs(work, "sample", lines, ids, verbose=False, dips=True, cast_lines=True)
                self.assertIn(ids[0], generated)
                self.assertNotIn(ids[1], generated)
                cache = (work / "refs" / "clean.json").read_bytes()
                for lid in ids:
                    sf.write(core.take_wav("sample", lid, "en"), np.full(sr // 3, 0.2), sr)
                first_take = core.take_wav("sample", ids[0], "en").read_bytes()
                (work / "rendered.json").write_text(json.dumps(dict.fromkeys(ids, {})), encoding="utf-8")
                lines[1]["text"] = "An updated second line."
                (work / "script.json").write_text(json.dumps({"lines": lines, "render": mode}), encoding="utf-8")
                model.side_effect = AssertionError("Rewording must reuse the source check without loading a model")
                with patch.object(core, "take_length", return_value=2.0) as estimate:
                    rows = core.fit_rows("sample", predict=True)
                estimate.assert_called_once_with(
                    root / "voices" / "speaker2.wav", "Second voice.", "An updated second line.", lang="en"
                )
                self.assertEqual(rows[1]["est"], 2.0)
                self.assertIsNotNone(rows[0]["speech"])
                self.assertEqual(core.take_wav("sample", ids[0], "en").read_bytes(), first_take)
                self.assertEqual((work / "refs" / "clean.json").read_bytes(), cache)


if __name__ == "__main__":
    unittest.main()
