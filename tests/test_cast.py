"""Standalone casting must select the parts' cast before loading centroids."""

import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import cast
import core


class CastJobSelectionTests(unittest.TestCase):
    def test_the_first_part_selects_its_job_before_centroids_are_loaded(self):
        class StopAfterCentroids(Exception):
            pass

        seen = []

        def centroids(_embed, selected):
            seen.append((core.active_job(), set(core.load_voices()), selected))
            raise StopAfterCentroids

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            work = root / "work"
            job = work / "jobs" / "sample_feature"
            job.mkdir(parents=True)
            (job / "job.json").write_text(json.dumps({"slug": "sample_feature"}), encoding="utf-8")
            (job / "voices.json").write_text(json.dumps({"speaker_local": {}}), encoding="utf-8")
            (work / "voices.json").write_text(json.dumps({"unrelated_shared_voice": {}}), encoding="utf-8")
            with (
                patch.object(core, "ROOT", root),
                patch.object(core, "WORK", work),
                patch.object(core, "_job", {"slug": None}),
                patch.dict(os.environ, {"DUB_JOB": ""}),
                patch.object(sys, "argv", ["cast.py", "sample_feature-02", "--cast", "speaker_local"]),
                patch.object(cast.draft, "embedder", return_value=object()),
                patch.object(cast.draft, "cast_centroids", side_effect=centroids),
                self.assertRaises(StopAfterCentroids),
            ):
                cast.main()
        self.assertEqual(seen, [("sample_feature", {"speaker_local"}, {"speaker_local"})])


if __name__ == "__main__":
    unittest.main()
