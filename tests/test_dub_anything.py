"""A resumed watcher must stop for a new failure on a previously blocked part."""

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import dub_anything as runner


class PauseCheckpointTests(unittest.TestCase):
    def test_pause_waits_for_background_checks_and_skips_the_next_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = pathlib.Path(tmp) / "pause"
            marker.touch()
            background = Mock()
            with (
                patch.dict(runner.os.environ, {"DUB_PAUSE_FILE": str(marker)}),
                patch.object(runner, "SECOND_RUN", {"sample": (background, [])}),
                patch.object(runner.subprocess, "run") as launch,
            ):
                with self.assertRaises(SystemExit) as stopped:
                    runner.tool({"slug": "sample"}, "next step", [], "next")
                self.assertEqual(stopped.exception.code, 75)
                background.wait.assert_called_once()
                launch.assert_not_called()

    def test_direct_dubs_have_no_pause_marker(self):
        with patch.dict(runner.os.environ, {"DUB_PAUSE_FILE": ""}):
            runner.pause_checkpoint()


class WatchBlockTests(unittest.TestCase):
    def run_changed_block(self, before, after, known=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            part = {"name": "sample-01"}
            path = root / part["name"] / "state.json"
            path.parent.mkdir()
            path.write_text(json.dumps(before), encoding="utf-8")

            def advance(job, name):
                path.write_text(json.dumps(after), encoding="utf-8")
                return "blocked"

            with (
                patch.object(runner, "WORK", root),
                patch.object(runner, "load_job", return_value={"slug": "sample", "voices": ["actor"]}),
                patch.object(runner.backend, "ensure", return_value=True),
                patch.object(runner, "split", return_value=True),
                patch.object(runner, "parts", return_value=[part]),
                patch.object(runner, "save_job"),
                patch.object(runner, "second"),
                patch.object(runner, "ready", return_value=True),
                patch.object(runner, "original_only", return_value=False),
                patch.object(runner, "drafted", return_value=True),
                patch.object(runner, "variant", return_value=part["name"]),
                patch.object(runner, "advance", side_effect=advance),
                patch.object(runner, "prepare_step", return_value="idle"),
                patch.object(runner, "finished", return_value=False),
                patch.object(runner, "report") as report,
                patch.object(runner.time, "sleep", side_effect=AssertionError("watcher waited on a new block")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                args = types.SimpleNamespace(slug="sample", watch=True)
                if known:
                    with self.assertRaisesRegex(AssertionError, "watcher waited on a new block"):
                        runner.cmd_run(args)
                    report.assert_not_called()
                else:
                    result = runner.cmd_run(args)
                    report.assert_called_once()
                    self.assertEqual(result, 2)

    def test_changed_failure_on_previously_blocked_part_stops_watch(self):
        self.run_changed_block(
            {"script": "old", "problem": "lines don't fit: L01"},
            {"script": "old", "problem": "lines don't fit: L05"},
        )

    def test_same_failure_after_script_change_stops_watch(self):
        self.run_changed_block(
            {"script": "old", "problem": "lines don't fit: L01"},
            {"script": "new", "problem": "lines don't fit: L01"},
        )

    def test_unchanged_known_block_keeps_waiting(self):
        state = {"script": "old", "problem": "lines don't fit: L01"}
        self.run_changed_block(state, state, known=True)
