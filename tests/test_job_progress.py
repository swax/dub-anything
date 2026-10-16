"""Part counts follow saved work, current batches, approval, and render freshness."""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from integrations import director, progress
from dub_anything import derived_script, script_sha


class JobProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.data, self.jobs = self.root / "service", self.root / "voice-jobs"
        self.job = {"id": "a" * 32, "slug": "sample", "state": "preparing", "stage": "Preparing", "voices": "actor"}
        self.names = [f"sample-{n:02}" for n in range(1, 5)]
        self.write("videos/sample-parts/parts.json", {"parts": [{"name": name} for name in self.names]})
        self.write("work/jobs/sample/job.json", {"captions": "auto", "signs": True})

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def log(self, text):
        path = self.data / "jobs" / self.job["id"] / "pipeline.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def snapshot(self):
        return progress.snapshot(self.job, self.data, self.root, self.jobs)

    def prepare(self, n):
        name = self.names[n - 1]
        self.write(f"work/{name}/segments.json", {"job_id": f"voice_{n}"})
        self.write(f"voice-jobs/voice_{n}/separator.json", {"model": "sample"})
        for file, value in (("draft", {"lines": []}), ("subs", {"events": []}), ("signs", {"events": []})):
            self.write(f"work/{name}/{file}.json", value)

    def script(self, n):
        name = self.names[n - 1]
        script = {
            "ready": True,
            "lines": [{"start": 1, "end": 2, "voice": "speaker_a", "text": "Hello.", "text_us": "Hello there."}],
        }
        self.write(f"work/{name}/script.json", script)
        self.job.setdefault("approved_parts", {})[name] = director.script_signature(script)
        return script

    def test_current_batch_advances_when_a_part_draft_is_saved(self):
        self.prepare(1)
        self.prepare(3)
        self.log("[12:00:00] draft sample-02..04 ...\n")
        result = self.snapshot()
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["current"]["parts"], [2])
        self.assertEqual(result["current"]["batch"], [2, 3, 4])
        self.assertEqual((result["phase"]["done"], result["phase"]["left"]), (2, 2))
        self.prepare(2)
        self.assertEqual(self.snapshot()["current"]["parts"], [4])

    def test_current_tool_and_phase_have_separate_remaining_counts(self):
        self.prepare(1)
        self.log("[12:00:00] subs sample-02 ...\n")
        result = self.snapshot()
        self.assertEqual(result["current"]["label"], "Reading captions")
        self.assertEqual(result["current"]["parts"], [2])
        self.assertEqual(result["phase"]["left"], 3)
        self.assertEqual(result["left"], 4)

    def test_both_voices_and_current_script_must_be_final_to_finish_a_part(self):
        self.job["state"], self.job["voices"] = "rendering", "both"
        self.prepare(1)
        script = self.script(1)
        self.write("work/sample-01/state.json", {"stage": "final", "script": script_sha(script)})
        self.assertEqual(self.snapshot()["finished"], 0)
        self.write(
            "work/sample-01.us/state.json", {"stage": "final", "script": script_sha(derived_script(script, "american"))}
        )
        self.assertEqual(self.snapshot()["finished"], 1)
        script["lines"][0]["text"] = "A revised line."
        self.write("work/sample-01/script.json", script)
        self.job["approved_parts"]["sample-01"] = director.script_signature(script)
        self.assertEqual(self.snapshot()["finished"], 0)

    def test_model_ready_flag_alone_does_not_count_as_approved_direction(self):
        self.prepare(1)
        self.script(1)
        self.job["approved_parts"] = {}
        self.job.update(state="directing", work_stage="directing", work_description="codex: directing part 1 of 4")
        result = self.snapshot()
        self.assertEqual(result["phase"]["done"], 0)
        self.assertEqual(result["current"]["parts"], [1])

    def test_original_only_parts_do_not_stay_in_remaining_work(self):
        self.write("work/sample-01/script.json", {"lines": []})
        result = self.snapshot()
        self.assertEqual(result["finished"], 1)
        self.assertEqual(result["left"], 3)
        self.assertTrue(all(row["done"] == 1 for row in result["stages"]))

    def test_paused_job_retains_current_step_and_counts_without_claiming_running(self):
        self.prepare(1)
        self.job["state"] = "paused"
        self.log("[12:00:00] prep sample-02 ...\n[12:01:00] prep sample-02 ok (60 s)\n")
        result = self.snapshot()
        self.assertFalse(result["current"]["active"])
        self.assertEqual(result["current"]["status"], "completed")
        self.assertEqual(result["current"]["parts"], [2])

    def test_unknown_part_count_and_partially_written_files_do_not_break_job_polling(self):
        self.log("[12:00:00] split ...\n")
        (self.root / "videos/sample-parts/parts.json").write_text('{"parts":')
        result = self.snapshot()
        self.assertIsNone(result["total"])
        self.assertEqual(result["current"]["step"], "split")
        self.write("videos/sample-parts/parts.json", {"parts": [{"name": "../outside"}]})
        self.assertIsNone(self.snapshot()["total"])

    def test_resumed_log_does_not_present_a_previous_runs_tool_as_current(self):
        self.log("[12:00:00] prep sample-01 ...\n\n--- new run ---\n")
        self.assertIsNone(self.snapshot()["current"])

    def test_disabled_caption_and_sign_steps_are_omitted(self):
        self.write("work/jobs/sample/job.json", {"captions": False, "signs": False})
        keys = [row["key"] for row in self.snapshot()["stages"]]
        self.assertNotIn("subs", keys)
        self.assertNotIn("signs", keys)


if __name__ == "__main__":
    unittest.main()
