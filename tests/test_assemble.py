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
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import ass
import assemble
import deliver


class SubtitlePacketChecks(unittest.TestCase):
    def check(self, packets):
        proc = MagicMock()
        proc.__enter__.return_value = proc
        proc.stdout = io.StringIO(packets)
        proc.stderr = io.StringIO("")
        proc.wait.return_value = 0
        with patch.object(deliver.subprocess, "Popen", return_value=proc):
            deliver.verify_subtitle_packets("sample.mkv", {0}, {3})

    def test_late_packets_fail_even_though_the_track_contains_the_cues(self):
        with self.assertRaisesRegex(SystemExit, "10.000s after its video; the source is untouched"):
            self.check("stream_index=0|pts_time=12.2|dts_time=12.0\nstream_index=3|pts_time=2.0|dts_time=2.0\n")

    def test_normal_reordering_and_retained_subtitles_do_not_fail(self):
        self.check(
            "stream_index=0|pts_time=2.2|dts_time=2.0\n"
            "stream_index=3|pts_time=1.9|dts_time=1.9\n"
            "stream_index=4|pts_time=0.0|dts_time=0.0\n"
        )


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
    def test_long_signs_do_not_delay_overlapping_dialogue_packets(self):
        for subtitles_only in (False, True):
            with self.subTest(subtitles_only=subtitles_only), tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                source = root / "Sample.mp4"
                subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=128x72:rate=25:duration=90",
                                "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=90", "-c:v", "libx264",
                                "-c:a", "aac", "-shortest", str(source)], check=True)  # fmt: skip
                base = root / "sample.dubbed.mp4"
                shutil.copyfile(source, base)
                base.with_suffix(".en.srt").write_text(
                    "1\n00:00:01,000 --> 00:00:02,000\nFirst remark.\n\n", encoding="utf-8"
                )
                signs = [
                    {"start": 10, "end": 80, "text": "Location", "box": [0.1, 0.8, 0.4, 0.9]},
                    {"start": 20, "end": 23, "text": "Notice", "box": [0.5, 0.1, 0.7, 0.2]},
                    {"start": 85, "end": 88, "text": "Exit", "box": [0.5, 0.1, 0.7, 0.2]},
                ]
                dialogue = [(0, 0.5, "Before the sign.")]
                dialogue += [(12 + i * 0.75, 12.4 + i * 0.75, "A passing remark.") for i in range(80)]
                assemble.write_srt(base.with_suffix(".ja.srt"), [(a, b, "An original remark.") for a, b, _ in dialogue])
                ass.write(base.with_suffix(".en.ass"), 128, 72, dialogue, signs)
                if not subtitles_only:
                    ass.write(base.with_suffix(".signs.ass"), 128, 72, signs=signs)
                with contextlib.redirect_stdout(io.StringIO()):
                    out = (
                        deliver.deliver_subtitles(source, base)
                        if subtitles_only
                        else deliver.deliver(source, {"actor": base}, "actor")
                    )
                streams = deliver.probe(out)["streams"]
                video = {s["index"] for s in streams if s["codec_type"] == "video"}
                subs = {s["index"] for s in streams if s["codec_type"] == "subtitle"}
                packets = json.loads(
                    subprocess.check_output(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_packets",
                            "-show_entries",
                            "packet=stream_index,pts_time,dts_time",
                            "-of",
                            "json",
                            str(out),
                        ]
                    )
                )["packets"]
                latest_video = float("-inf")
                subtitle_starts = []
                for packet in packets:
                    if packet["stream_index"] in video:
                        latest_video = max(latest_video, float(packet.get("dts_time", packet["pts_time"])))
                    elif packet["stream_index"] in subs:
                        start = float(packet["pts_time"])
                        subtitle_starts.append(start)
                        self.assertLessEqual(
                            latest_video - start, 0.5, "Subtitle data arrived after its video playback time"
                        )
                self.assertIn(12.0, subtitle_starts)
                self.assertIn(42.0, subtitle_starts)

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
