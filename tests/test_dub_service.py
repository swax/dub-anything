"""Exercise the integration boundary without GPU work or external model calls."""

import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from integrations import director
from integrations import dub_service as service


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.media = self.root / "Sample Movie.mkv"
        self.media.write_bytes(b"original-video")
        self.store = service.Store(self.root / "service")
        self.request = {
            "path": str(self.media),
            "title": "Sample Movie",
            "item_id": "a" * 32,
            "source_lang": "ja",
            "target_lang": "en",
            "voices": "actor",
            "director": "codex",
        }

    def create(self, **changes):
        return self.store.create({**self.request, **changes})

    def test_persistent_queue_and_duplicate_detection(self):
        job = self.create()
        self.assertEqual(service.Store(self.store.data).get(job["id"])["state"], "queued")
        with self.assertRaisesRegex(ValueError, "already have a job"):
            self.create(director="claude")
        self.store.action(job["id"], "cancel")
        self.assertNotEqual(self.create()["id"], job["id"])

    def test_pair_voice_and_path_validation(self):
        for changes in (
            {"director": "unknown"},
            {"source_lang": "de123"},
            {"source_lang": "en", "target_lang": "ja", "voices": "both"},
            {"path": "relative.mkv"},
            {"path": str(self.root)},
            {"item_id": "../file"},
            {"model": "--bad-flag"},
            {"review_before_publish": "false"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.create(**changes)

    def test_resume_and_publish_state_guards(self):
        job = self.create()
        with self.assertRaises(ValueError):
            self.store.action(job["id"], "publish")
        self.store.action(job["id"], "cancel")
        resumed = self.store.action(job["id"], "resume")
        self.assertFalse(resumed["cancel_requested"])
        self.store.update(job["id"], state="awaiting_review")
        self.assertEqual(self.store.action(job["id"], "publish")["state"], "publish_queued")

    def test_queued_pause_resume_and_cancel(self):
        job = self.create()
        self.assertEqual(self.store.action(job["id"], "pause")["state"], "paused")
        self.assertIsNone(self.store.claim())
        with self.assertRaises(ValueError):
            self.store.action(job["id"], "pause")
        resumed = self.store.action(job["id"], "resume")
        self.assertEqual(resumed["state"], "queued")
        self.assertFalse(resumed["pause_requested"])
        self.store.action(job["id"], "pause")
        self.assertEqual(self.store.action(job["id"], "cancel")["state"], "cancelled")

    def test_clock_follows_claim_pause_resume_review_and_publish(self):
        job = self.create()
        with patch.object(service.time, "time", return_value=100):
            self.assertEqual(self.store.claim()["active_since"], 100)
        with patch.object(service.time, "time", return_value=120):
            self.store.action(job["id"], "pause")
        with patch.object(service.time, "time", return_value=130):
            self.assertEqual(self.store.update(job["id"], state="paused")["active_seconds"], 30)
        with patch.object(service.time, "time", return_value=1000):
            self.store.action(job["id"], "resume")
            self.store.claim()
        with patch.object(service.time, "time", return_value=1040):
            self.assertEqual(self.store.update(job["id"], state="awaiting_review")["active_seconds"], 70)
        with patch.object(service.time, "time", return_value=2000):
            self.store.action(job["id"], "publish")
            self.store.claim()
        with patch.object(service.time, "time", return_value=2010):
            result = self.store.update(job["id"], state="completed")
        self.assertEqual(result["active_seconds"], 80)
        self.assertIsNone(result["active_since"])

    def test_clock_recovery_excludes_worker_downtime(self):
        job = self.create()
        with patch.object(service.time, "time", return_value=100):
            self.store.claim()
        with patch.object(service.time, "time", return_value=120):
            service.Worker(self.store).cancelled(job)
        with patch.object(service.time, "time", return_value=2000):
            self.store.recover()
        self.assertEqual(self.store.get(job["id"])["active_seconds"], 20)

    def test_pending_pause_stays_active_until_checkpoint_and_survives_restart(self):
        job = self.create()
        self.store.claim()
        paused = self.store.action(job["id"], "pause")
        self.assertEqual(paused["state"], "staging")
        self.assertTrue(paused["pause_requested"])
        self.store.recover()
        self.assertEqual(self.store.get(job["id"])["state"], "paused")

    def test_pause_keeps_publish_queue_intent_but_cannot_interrupt_publication(self):
        job = self.create()
        self.store.update(job["id"], state="publish_queued")
        self.store.action(job["id"], "pause")
        self.assertEqual(self.store.action(job["id"], "resume")["state"], "publish_queued")
        self.store.update(job["id"], state="publishing")
        with self.assertRaises(ValueError):
            self.store.action(job["id"], "pause")

    def test_unknown_languages_are_queued_for_setup_and_text_is_automatic(self):
        job = self.create(source_lang="French", target_lang="German", signs=False, captions=False)
        self.assertEqual(job["source_lang"], "French")
        self.assertEqual(job["target_lang"], "German")
        self.assertTrue(job["auto_text"])
        self.assertTrue(job["signs"])

    def test_recovery_never_automatically_restarts_active_jobs(self):
        job = self.create()
        self.store.update(job["id"], state="rendering")
        self.store.recover()
        self.assertEqual(self.store.get(job["id"])["state"], "interrupted")

    def test_only_one_worker_owns_the_queue(self):
        with service.service_lock(self.store.data):
            with self.assertRaises(RuntimeError):
                with service.service_lock(self.store.data):
                    self.fail("a second worker acquired the lease")

    def test_publish_preserves_original_and_is_idempotent(self):
        job = self.create()
        render = self.root / "render.mkv"
        render.write_bytes(b"dubbed-with-original-track")
        job = self.store.update(job["id"], rendered_path=str(render), state="awaiting_review")
        worker = service.Worker(self.store)
        worker.publish(job)
        result = self.store.get(job["id"])
        self.assertEqual(result["state"], "completed")
        self.assertEqual(pathlib.Path(result["published_path"]).read_bytes(), render.read_bytes())
        self.assertEqual(self.media.read_bytes(), b"original-video")
        worker.publish(job)  # Reconcile a crash between rename and queue update.
        self.assertEqual(self.media.read_bytes(), b"original-video")

    def test_publish_refuses_to_overwrite_existing_file(self):
        job = self.create()
        render = self.root / "render.mkv"
        render.write_bytes(b"dubbed")
        job = self.store.update(job["id"], rendered_path=str(render))
        destination = self.media.with_name(f"{self.media.stem} - Dub Anything en actor {job['id'][:8]}.mkv")
        destination.write_bytes(b"user-file")
        with self.assertRaises(FileExistsError):
            service.Worker(self.store).publish(job)
        self.assertEqual(destination.read_bytes(), b"user-file")

    def test_cancelled_copy_does_not_publish(self):
        job = self.create()
        self.store.action(job["id"], "cancel")
        destination = self.root / "copy.mkv"
        with self.assertRaises(service.Cancelled):
            service.Worker(self.store).copy(job, self.media, destination)
        self.assertFalse(destination.exists())

    def test_process_cancellation_stops_spawned_work(self):
        job = self.create()
        worker = service.Worker(self.store)
        timer = threading.Timer(0.4, lambda: self.store.action(job["id"], "cancel"))
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(service.Cancelled):
                worker.run(job, [sys.executable, "-c", "import time; time.sleep(30)"], "cancel-test")
        finally:
            timer.join()
        self.assertLess(time.monotonic() - started, 10)

    def test_pause_finishes_current_process_before_stopping(self):
        job = self.create()
        self.store.claim()
        worker = service.Worker(self.store)
        saved = self.root / "completed-step.txt"
        timer = threading.Timer(0.2, lambda: self.store.action(job["id"], "pause"))
        timer.start()
        try:
            with self.assertRaises(service.Paused):
                worker.run(
                    job,
                    [
                        sys.executable,
                        "-c",
                        "import pathlib,sys,time; time.sleep(.7); pathlib.Path(sys.argv[1]).write_text('saved')",
                        str(saved),
                    ],
                    "pause-test",
                )
        finally:
            timer.join()
        self.assertEqual(saved.read_text(), "saved")

    def test_pipeline_pause_reaches_checkpoint_without_becoming_failure(self):
        job = self.create()
        self.store.claim()
        worker = service.Worker(self.store)
        timer = threading.Timer(0.2, lambda: self.store.action(job["id"], "pause"))
        timer.start()
        try:
            with self.assertRaises(service.Paused):
                worker.run(
                    job,
                    [
                        sys.executable,
                        "-c",
                        "import os,pathlib,time; marker=pathlib.Path(os.environ['DUB_PAUSE_FILE']); deadline=time.time()+5\nwhile not marker.exists() and time.time()<deadline: time.sleep(.05)\nraise SystemExit(75 if marker.exists() else 9)",
                    ],
                    "pipeline",
                )
        finally:
            timer.join()

    def test_worker_records_pause_and_does_not_claim_it_again(self):
        job = self.create()
        worker = service.Worker(self.store)

        def pause(current):
            self.store.action(current["id"], "pause")
            worker.cancelled(current)

        with patch.object(worker, "process", side_effect=pause):
            worker.thread.start()
            try:
                deadline = time.monotonic() + 5
                while self.store.get(job["id"])["state"] != "paused" and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertEqual(self.store.get(job["id"])["state"], "paused")
                self.assertIsNone(self.store.claim())
            finally:
                worker.stop.set()
                worker.thread.join(timeout=5)

    def test_capabilities_do_not_expose_service_token(self):
        result = service.capabilities()
        self.assertEqual({d["id"] for d in result["directors"]}, {"codex", "claude"})
        self.assertNotIn("token", result)
        self.assertIn({"source": "ja", "target": "en", "label": "Japanese → English"}, result["pairs"])

    def test_authenticated_http_create_cancel_and_errors(self):
        server = service.Server({"host": "127.0.0.1", "port": 0, "token": "private-test-token"}, self.store)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"

        def request(path, data=None, token=True):
            headers = {"Authorization": "Bearer private-test-token"} if token else {}
            if data is not None:
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(
                base + path, data=json.dumps(data).encode() if data is not None else None, headers=headers
            )
            with urllib.request.urlopen(req, timeout=3) as response:
                return json.load(response)

        try:
            self.assertEqual(request("/health", token=False)["status"], "ok")
            with self.assertRaises(urllib.error.HTTPError) as error:
                request("/jobs", token=False)
            self.assertEqual(error.exception.code, 401)
            job = request("/jobs", self.request)
            self.assertEqual(request("/jobs")[0]["id"], job["id"])
            self.assertIn("progress", request("/jobs")[0])
            self.assertIn("progress", request(f"/jobs/{job['id']}"))
            self.assertEqual(request(f"/jobs/{job['id']}")["timing"]["elapsed_seconds"], 0)
            self.assertNotIn("timing", self.store.get(job["id"]))
            self.assertNotIn("progress", self.store.get(job["id"]))
            self.assertEqual(request(f"/jobs/{job['id']}/pause", {})["state"], "paused")
            self.assertEqual(request(f"/jobs/{job['id']}/resume", {})["state"], "queued")
            self.assertEqual(request(f"/jobs/{job['id']}/cancel", {})["state"], "cancelled")
            with self.assertRaises(urllib.error.HTTPError) as error:
                request("/jobs", [])
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_directing_retries_timing_and_requires_supervisor_approval(self):
        import dub_anything as runner

        job = self.create()
        part = {"name": job["slug"] + "-01", "start": 0, "end": 10}
        directory = self.root / "work" / part["name"]
        directory.mkdir(parents=True)
        script_path = directory / "script.json"
        voice = job["slug"] + "_speaker"
        (self.root / "work" / "voices.json").write_text(json.dumps({voice: {}}))
        script = {
            "ready": True,
            "lines": [{"start": 1, "end": 3, "voice": voice, "ja": "こんにちは", "text": "Hello."}],
        }
        # A model's ready flag from an interrupted run must not bypass checks.
        script_path.write_text(json.dumps(script))
        rendered = self.root / "finished.mkv"
        pipeline_job = {}
        prompts, renders = [], []

        def pipeline(job, *args):
            if args[0] == "run":
                final = json.loads(script_path.read_text(encoding="utf-8"))
                self.assertTrue(final["ready"])
                self.assertIn(part["name"], self.store.get(job["id"])["approved_parts"])
                renders.append(True)
                rendered.write_bytes(b"finished dub")
                pipeline_job["delivered"] = {"path": str(rendered)}

        def run(job, args, name, prompt=None, **kwargs):
            if name.startswith("director"):
                prompts.append(prompt)
                script_path.write_text(json.dumps(script))
            if name.startswith("fit") and len(prompts) == 1:
                return "L01 OVER? Hello."
            return ""

        worker = service.Worker(self.store)
        with (
            patch.object(service, "ROOT", self.root),
            patch.object(director, "command", return_value=["fake"]),
            patch.object(service.language_setup, "ensure", return_value=job),
            patch.object(runner, "load_job", return_value=pipeline_job),
            patch.object(runner, "parts", return_value=[part]),
            patch.object(runner, "drafted", return_value=True),
            patch.object(runner, "original_only", return_value=False),
            patch.object(worker, "pipeline", side_effect=pipeline),
            patch.object(worker, "run", side_effect=run),
        ):
            worker.process(job)
        self.assertEqual(len(prompts), 2)
        self.assertIn("OVER?", prompts[1])
        self.assertEqual(len(renders), 1)
        self.assertEqual(self.store.get(job["id"])["state"], "awaiting_review")
        self.assertEqual(self.media.read_bytes(), b"original-video")


class DirectorTests(unittest.TestCase):
    def setUp(self):
        self.job = {"slug": "jf_test", "source_lang": "ja", "target_lang": "en"}
        self.part = {"name": "jf_test-01", "start": 0, "end": 10}
        self.voices = {"jf_test_speaker": {}}

    def script(self):
        return {
            "ready": True,
            "lines": [{"start": 1, "end": 3, "voice": "jf_test_speaker", "ja": "こんにちは", "text": "Hello."}],
        }

    def test_validation_clears_model_ready_flag(self):
        result = director.validate_script(self.script(), self.part, self.job, self.voices)
        self.assertFalse(result["ready"])

    def test_rejects_invalid_windows_and_cross_job_cast(self):
        for change in (
            {"start": -1},
            {"end": 11},
            {"start": float("nan")},
            {"end": True},
            {"text": ""},
            {"voice": "another_show_speaker"},
        ):
            script = self.script()
            script["lines"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                director.validate_script(script, self.part, self.job, self.voices)

    def test_empty_speech_part_needs_explicit_reason(self):
        with self.assertRaises(ValueError):
            director.validate_script({"lines": []}, self.part, self.job, self.voices)
        director.validate_script({"lines": [], "original_reason": "Instrumental credits"}, self.part, self.job, {})

    def test_commands_use_installed_harnesses_and_no_global_permission_bypass(self):
        with patch.object(director, "executable", side_effect=lambda kind, env=None: kind):
            codex = director.command("codex")
            claude = director.command("claude", "configured-model")
        self.assertEqual(codex[:2], ["codex", "exec"])
        self.assertIn("workspace-write", codex)
        self.assertIn("dontAsk", claude)
        self.assertNotIn("--dangerously-skip-permissions", claude)
        self.assertEqual(claude[-2:], ["--model", "configured-model"])

    def test_json_reported_failure_is_not_a_successful_director(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "director.log"
            for kind, event in (("codex", {"type": "turn.failed"}), ("claude", {"type": "result", "is_error": True})):
                log.write_text(json.dumps(event) + "\n", encoding="utf-8")
                with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                    director.check_result(log, 0, kind)
            log.write_text('{"type":"turn.completed"}\n', encoding="utf-8")
            director.check_result(log, 0, "codex")


if __name__ == "__main__":
    unittest.main()
