"""Timed subtitle placement survives part offsets and layout-only delivery."""

import contextlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import ass
import assemble
import deliver


class AssemblyLayoutTests(unittest.TestCase):
    def test_offsets_reading_linger_plain_srt_and_stale_styled_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            parts = [{"name": "sample-01", "start": 100, "end": 110}, {"name": "sample-02", "start": 110, "end": 120}]
            manifest = root / "videos/sample-parts/parts.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"video": str(root / "source.mp4"), "parts": parts}), encoding="utf-8")
            for p in parts:
                folder = root / "work" / p["name"]
                folder.mkdir(parents=True)
                script = {"lines": [{"start": 1, "end": 2, "text": "Hello."}]}
                if p == parts[1]:
                    # Starts in the reading linger, after the speech window.
                    script["subtitle_layout"] = [{"start": 2.1, "end": 4, "top": 0.19}]
                (folder / "script.json").write_text(json.dumps(script), encoding="utf-8")
            out = root / "assembled.mkv"
            styled, forced = out.with_suffix(".en.ass"), out.with_suffix(".signs.ass")
            forced.write_text("stale", encoding="utf-8")
            with (
                patch.object(assemble, "ROOT", root),
                patch.object(assemble, "WORK", root / "work"),
                patch.object(assemble, "out_dir", return_value=root),
                patch.object(
                    assemble, "probe", return_value={"streams": [{"codec_type": "video", "width": 1280, "height": 720}]}
                ),
                patch.object(sys, "argv", ["assemble.py", "sample", "--subtitles", "--out", str(out)]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                assemble.main()
                text = styled.read_text(encoding="utf-8-sig")
                self.assertIn("0:00:11.00,0:00:12.50,Dialogue,,0,0,0,,{\\an8\\pos(640,137)}Hello.", text)
                self.assertIn("0:00:01.00,0:00:02.50,Dialogue,,0,0,0,,Hello.", text)
                self.assertNotIn("\\pos", out.with_suffix(".en.srt").read_text(encoding="utf-8"))
                self.assertFalse(forced.exists())
                path = root / "work/sample-02/script.json"
                script = json.loads(path.read_text(encoding="utf-8"))
                del script["subtitle_layout"]
                path.write_text(json.dumps(script), encoding="utf-8")
                assemble.main()
                self.assertFalse(styled.exists())
                self.assertFalse(forced.exists())


@unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
class LayoutDeliveryTests(unittest.TestCase):
    def test_mixed_variants_keep_styled_layout_without_a_forced_sign_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            source = root / "Sample.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=128x72:duration=1",
                            "-f", "lavfi", "-i", "sine=duration=1", "-c:v", "libx264", "-c:a", "aac",
                            "-shortest", str(source)], check=True)  # fmt: skip
            dubs = {v: root / f"sample.{v}.mp4" for v in ("actor", "american")}
            for path in dubs.values():
                shutil.copyfile(source, path)
                path.with_suffix(".en.srt").write_text("1\n00:00:00,100 --> 00:00:00,900\nHello.\n\n", encoding="utf-8")
            ass.write(
                dubs["american"].with_suffix(".en.ass"),
                128,
                72,
                [(0.1, 0.9, "Hello.")],
                subtitle_layout=[{"start": 0, "end": 1, "top": 0.19}],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                out = deliver.deliver(source, dubs, "actor")
            self.assertEqual(out.suffix, ".mkv")

            def subtitle_tracks(path):
                info = json.loads(
                    subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)])
                )
                return [s for s in info["streams"] if s["codec_type"] == "subtitle"]

            tracks = subtitle_tracks(out)
            self.assertEqual([s["codec_name"] for s in tracks], ["subrip", "ass"])
            self.assertFalse(any(s.get("disposition", {}).get("forced") for s in tracks))
            self.assertEqual(
                [deliver.name_of(s) for s in tracks], ["English (actor-voice dub)", "English (American dub)"]
            )
            # A nondefault variant can also supply a real signs-only track.
            ass.write(
                dubs["american"].with_suffix(".signs.ass"),
                128,
                72,
                signs=[{"start": 0, "end": 1, "text": "Station", "box": [0.1, 0.1, 0.4, 0.2]}],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                out = deliver.deliver(out, dubs, "actor")
            tracks = subtitle_tracks(out)
            self.assertEqual([s["codec_name"] for s in tracks], ["subrip", "ass", "ass"])
            self.assertEqual([s["disposition"]["forced"] for s in tracks], [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
