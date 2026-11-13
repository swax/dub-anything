"""Visual text can be scanned and previewed without preparing original audio."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import core
import signs


class OriginalAudioSignsTests(unittest.TestCase):
    def test_standalone_scan_keeps_explicit_languages_for_later_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root / "work/sample-01"
            with (
                patch.object(signs, "ROOT", root),
                patch.object(core, "WORK", root / "work"),
                patch.object(
                    sys, "argv", ["signs.py", work.name, "--video", "opening.mp4", "--source", "en", "--target", "es"]
                ),
                patch.object(sys, "stdout", Mock()),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(signs.subs, "size", return_value=(1280, 720)),
                patch.object(signs, "cuts", return_value=[]),
                patch.object(signs.subs, "frames", return_value=[]),
                patch.object(signs.subs, "engine", return_value=(object(), False)) as engine,
                patch.object(signs.subprocess, "run", return_value=SimpleNamespace(stdout="10.0")),
                patch.dict(sys.modules, {"cv2": Mock()}),
            ):
                signs.main()
            engine.assert_called_once_with("en")
            doc = json.loads((work / "signs.json").read_text(encoding="utf-8"))
            self.assertEqual(doc["video"], str(root / "opening.mp4"))
            self.assertEqual(signs.pair(work), ("en", "es"))
            self.assertFalse((work / "segments.json").exists())
            with (
                patch.object(signs, "ROOT", root),
                patch.object(core, "WORK", root / "work"),
                patch.object(sys, "argv", ["signs.py", work.name, "--preview"]),
                patch.object(sys, "stdout", Mock()),
                patch.object(signs.subs, "size", return_value=(1280, 720)),
                patch.object(signs, "preview") as preview,
            ):
                signs.main()
            preview.assert_called_once_with(work, root / "opening.mp4", 1280, 720, set(), target="es")

    def test_unprepared_scan_requires_explicit_languages(self):
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(signs, "ROOT", Path(tmp)),
                patch.object(core, "WORK", Path(tmp) / "work"),
                patch.object(sys, "argv", ["signs.py", "sample-01", "--video", "opening.mp4"]),
                patch.object(sys, "stdout", Mock()),
                contextlib.redirect_stderr(io.StringIO()),
                patch.object(signs.subs, "engine") as engine,
            ):
                with self.assertRaises(SystemExit) as stopped:
                    signs.main()
                self.assertEqual(stopped.exception.code, 2)
                engine.assert_not_called()

    def test_prepared_metadata_takes_precedence_over_an_older_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "segments.json").write_text('{"source_lang":"ko","target_lang":"en"}', encoding="utf-8")
            (work / "signs.json").write_text('{"source_lang":"en","target_lang":"es"}', encoding="utf-8")
            self.assertEqual(signs.pair(work), ("ko", "en"))


if __name__ == "__main__":
    unittest.main()
