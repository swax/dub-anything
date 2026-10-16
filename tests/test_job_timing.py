"""Clocks exclude waiting; estimates learn each stage and voice independently."""

import datetime
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from integrations import job_timing as timing


class JobTimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = pathlib.Path(self.temp.name)
        self.job = {"id": "a" * 32, "state": "preparing", "active_seconds": 90, "active_since": 100}
        self.folder = self.data / "jobs" / self.job["id"]
        self.folder.mkdir(parents=True)
        self.view = {
            "total": 3,
            "left": 3,
            "names": ["sample-01", "sample-02", "sample-03"],
            "phase": {"key": "prep", "label": "Preparing audio", "left": 2},
            "stages": [
                {"key": "prep", "label": "Preparing audio", "left": 2},
                {"key": "direction", "label": "Directing dialogue", "left": 3},
            ],
        }

    def log(self, text):
        (self.folder / "pipeline.log").write_text(text, encoding="utf-8")

    def test_clock_counts_pause_checkpoint_but_not_paused_or_queued_time(self):
        timing.transition(self.job, "preparing", "preparing", 130)
        self.assertEqual(timing.elapsed(self.job, 140), 130)
        timing.transition(self.job, "preparing", "paused", 150)
        self.job["state"] = "paused"
        self.assertEqual(timing.elapsed(self.job, 1000), 140)
        timing.transition(self.job, "paused", "queued", 1000)
        timing.transition(self.job, "queued", "staging", 1100)
        self.job["state"] = "staging"
        self.assertEqual(timing.elapsed(self.job, 1110), 150)

    def test_crash_recovery_counts_only_to_last_heartbeat(self):
        self.job["last_active_at"] = 135
        timing.transition(self.job, "preparing", "interrupted", 1000, recovery=True)
        self.job["state"] = "interrupted"
        self.assertEqual(timing.elapsed(self.job, 2000), 125)
        self.assertIsNone(self.job["active_since"])

    def test_clock_rollback_cannot_make_runtime_negative(self):
        self.assertEqual(timing.elapsed(self.job, 80), 90)
        timing.transition(self.job, "preparing", "paused", 80)
        self.assertEqual(self.job["active_seconds"], 90)

    def test_legacy_sessions_exclude_resume_gaps_and_merge_overlapping_logs(self):
        date = "2026-01-01"

        def stamp(clock):
            return datetime.datetime.fromisoformat(date + " " + clock).timestamp()

        self.log(f"--- {date} 10:00:00 ---\n[10:01:00] saved\n--- {date} 11:00:00 ---\n[11:02:00] saved\n")
        other = self.folder / "director-01.log"
        other.write_text(f"--- {date} 11:01:00 ---\nfinished\n", encoding="utf-8")
        os.utime(self.folder / "pipeline.log", (stamp("11:02:00"),) * 2)
        os.utime(other, (stamp("11:03:00"),) * 2)
        self.assertEqual(timing.legacy_elapsed(self.folder, stamp("12:00:00")), 240)

    def test_legacy_timestamps_cross_midnight(self):
        self.log("--- 2026-01-01 23:59:00 ---\n[00:01:00] saved\n--- 2026-01-02 01:00:00 ---\n")
        end = datetime.datetime.fromisoformat("2026-01-02 01:00:00").timestamp()
        os.utime(self.folder / "pipeline.log", (end,) * 2)
        self.assertEqual(timing.legacy_elapsed(self.folder, end), 120)

    def test_reused_batch_units_and_failed_steps_do_not_skew_rates(self):
        self.log(
            "[10:00:00] separate sample-01..02 ok (100 s)\n"
            "[10:02:00] separate sample-02..03 ok (30 s)\n"
            "[10:03:00] separate sample-01..03 ok (1 s)\n"
            "[10:04:00] prep sample-01 FAILED (900 s)\n"
        )
        rates = timing.samples(self.folder, self.view)
        self.assertEqual(rates["separate"], 50)
        self.assertNotIn("prep", rates)

    def test_stage_eta_available_while_full_estimate_is_learning(self):
        self.log("[10:00:00] prep sample-01 ok (60 s)\n")
        result = timing.snapshot(self.job, self.data, self.view, now=120)
        self.assertEqual(result["elapsed_seconds"], 110)
        self.assertEqual(result["stage_seconds"], 120)
        self.assertIsNone(result["remaining_seconds"])
        self.assertEqual(result["missing_stages"], ["Directing dialogue"])

    def test_full_estimate_weights_stages_separately(self):
        self.log("[10:00:00] prep sample-01 ok (60 s)\n")
        (self.folder / "timings.jsonl").write_text(
            json.dumps({"kind": "direction", "seconds": 300}) + "\npartial-json", encoding="utf-8"
        )
        result = timing.snapshot(self.job, self.data, self.view, now=120)
        self.assertEqual(result["remaining_seconds"], 1020)
        self.assertEqual(result["status"], "estimated")

    def test_voice_rates_include_qa_rounds_and_remaining_voices_independently(self):
        self.view.update(
            phase={"key": "render", "label": "Rendering voices", "left": 2},
            stages=[{"key": "render", "label": "Rendering voices", "left": 2}],
            voices={"actor": {"finished": [1, 2], "left": 1}, "american": {"finished": [1], "left": 2}},
        )
        self.log(
            "[10:00:00] render sample-01 ok (100 s)\n"
            "[10:01:00] qa sample-01 ok (10 s)\n"
            "[10:02:00] round 1 sample-01 ok (20 s)\n"
            "[10:03:00] qa sample-01 (round 1) ok (10 s)\n"
            "[10:04:00] render sample-01.us ok (300 s)\n"
            "[10:05:00] qa sample-01.us ok (20 s)\n"
        )
        result = timing.snapshot(self.job, self.data, self.view, now=120)
        self.assertEqual(result["remaining_seconds"], 140 + 2 * 320)
        self.view["voices"]["american"]["finished"] = []
        self.assertIsNone(timing.snapshot(self.job, self.data, self.view)["remaining_seconds"])

    def test_all_parts_finished_does_not_claim_job_has_finished_assembly(self):
        self.view.update(left=0, stages=[{"key": "render", "label": "Rendering", "left": 0}])
        result = timing.snapshot(self.job, self.data, self.view, now=120)
        self.assertEqual(result["status"], "finishing")
        self.assertIsNone(result["remaining_seconds"])
        for state, expected in (("completed", "complete"), ("awaiting_review", "review")):
            self.job.update(state=state, active_since=None)
            result = timing.snapshot(self.job, self.data, self.view, now=120)
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["running"])


if __name__ == "__main__":
    unittest.main()
