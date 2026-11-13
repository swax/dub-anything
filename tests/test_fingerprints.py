"""A remark costs no render: fingerprints cover what is rendered and delivered."""

import contextlib
import copy
import hashlib
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

import core
import dub_anything as runner


def script():
    return {
        "lines": [
            {"start": 0.0, "end": 2.0, "voice": "speaker_a", "ja": "荷物が届きました。", "text": "The parcel is here."},
            {"start": 3.0, "end": 5.0, "voice": "speaker_b", "ja": "ありがとう。", "text": "Thank you very much."},
        ],
        "render": {"fill": True, "voice": "actor"},
    }


class RenderFingerprintTests(unittest.TestCase):
    def test_remarks_leave_the_render_alone(self):
        before = runner.script_sha(script())
        noted = script()
        noted["lines"][0].update(listen="speaker_a or the neighbour?", note="heard twice", notes=["checked"])
        noted.update(notes="scene at the door", listen="the whole part", ready=True, signs=[{"text": "Station"}])
        self.assertEqual(runner.script_sha(noted), before)

    def test_what_a_render_reads_changes_it(self):
        before = runner.script_sha(script())
        for change in (
            lambda s: s["lines"][0].update(text="The parcel has come."),
            lambda s: s["lines"][0].update(end=2.4),
            lambda s: s["lines"][1].update(voice="speaker_a"),
            lambda s: s["lines"][1].update(ja="ありがとうございます。"),
            lambda s: s["lines"][1].update(take=2),
            lambda s: s["render"].update(fill=False),
            # A field no script knows: a new take rather than a stale one.
            lambda s: s["lines"][0].update(whispered=True),
        ):
            changed = script()
            change(changed)
            self.assertNotEqual(runner.script_sha(changed), before)

    def test_american_words_render_the_american_voice_only(self):
        plain, worded = script(), script()
        worded["lines"][0]["text_us"] = "The package is here."
        self.assertEqual(runner.script_sha(worded), runner.script_sha(plain))
        self.assertNotEqual(
            runner.script_sha(runner.derived_script(worded, "american")),
            runner.script_sha(runner.derived_script(plain, "american")),
        )

    def test_script_without_remarks_keeps_its_earlier_fingerprint(self):
        self.assertEqual(runner.script_sha(script()), runner.legacy_script_sha(script()))


class VoiceFingerprintTests(unittest.TestCase):
    VOICES = {
        "speaker_a": {"from": "sample-01", "clips": [[1.0, 4.5], ["sample-02", 8.0, 11.0]], "ref_text": "こんにちは。"},
        "speaker_a_calm": {
            "from": "sample-01",
            "clips": [[20.0, 24.0]],
            "ref_text": "そうですか。",
            "register_of": "speaker_a",
            "register": "calm",
            "reference_selection": "neutral4",
        },
    }

    def sig(self, voices, key):
        with patch.object(core, "load_voices", return_value=voices):
            return core.voice_sig(key), core.line_id({"voice": key, "text": "The parcel is here."})

    def test_unchanged_entry_keeps_its_earlier_fingerprint(self):
        for key, spec in self.VOICES.items():
            earlier = hashlib.sha1(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            self.assertEqual(self.sig(self.VOICES, key)[0], earlier)

    def test_remark_keeps_the_voice_and_its_takes(self):
        noted = copy.deepcopy(self.VOICES)
        noted["speaker_a"].update(note="the landlord; second clip is from the phone call", notes=["v1"], listen="?")
        self.assertEqual(self.sig(noted, "speaker_a"), self.sig(self.VOICES, "speaker_a"))

    def test_changed_reference_makes_the_takes_again(self):
        before = self.sig(self.VOICES, "speaker_a")
        for change in (
            lambda v: v["clips"].append([30.0, 33.0]),
            lambda v: v.update(ref_text="こんにちは、皆さん。"),
            lambda v: v.update(profile="existing-profile"),
        ):
            changed = copy.deepcopy(self.VOICES)
            change(changed["speaker_a"])
            sig, lid = self.sig(changed, "speaker_a")
            self.assertNotEqual(sig, before[0])
            self.assertNotEqual(lid, before[1])


class RecordedFingerprintTests(unittest.TestCase):
    """State and delivery records written before the fingerprints were narrowed."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = pathlib.Path(temporary.name)
        self.part = {"name": "sample-01", "start": 0}
        self.work = self.root / self.part["name"]
        self.work.mkdir()
        self.script = script()
        self.script["ready"] = True
        self.script["lines"][0]["listen"] = "speaker_a or the neighbour?"
        self.script["lines"][0]["text_us"] = "The package is here."
        self.job = {"slug": "sample", "source": "sample.mp4", "langs": ["ja", "en"], "voices": ["actor"]}
        self.saved = []
        self.ids = {
            "speaker_a": "Lfirst",
            "speaker_b": "Lsecond",
        }  # each line's id, which holds its voice's fingerprint
        for patcher in (
            patch.object(core, "WORK", self.root),
            patch.object(runner, "parts", return_value=[self.part]),
            patch.object(runner, "job_dir_of", return_value=None),
            patch.object(runner, "load_job", return_value=self.job),
            patch.object(runner, "save_job", side_effect=lambda job: self.saved.append(copy.deepcopy(job))),
            patch.object(runner, "line_id", side_effect=lambda line: self.ids[line["voice"]]),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.write("script.json", self.script)
        self.write(
            "state.json",
            {
                "script": runner.legacy_script_sha(self.script),
                "stage": "final",
                "round": 1,
                "qa_take_hashes": {"Lfirst": "aa", "Lsecond": "bb"},
            },
        )

    def write(self, name, value):
        (self.work / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def state(self):
        return json.loads((self.work / "state.json").read_text(encoding="utf-8"))

    def status(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.object(runner, "finished", return_value=True):
            runner.cmd_status(types.SimpleNamespace(slug="sample"))
        return output.getvalue()

    def test_earlier_record_still_counts_before_it_is_adopted(self):
        self.assertNotEqual(runner.legacy_script_sha(self.script), runner.script_sha(self.script))
        self.assertTrue(runner.rendered_from(self.state(), self.script))
        self.assertIn("actor: final r1", self.status())

    def test_adopted_part_stays_final_through_a_new_remark(self):
        runner.adopt_fingerprints(self.job, [self.part])
        self.assertEqual(self.state()["script"], runner.script_sha(self.script))
        self.assertEqual(self.state()["qa_take_hashes"], {"Lfirst": "aa", "Lsecond": "bb"})
        self.script["lines"][1]["listen"] = "the name may be wrong"
        self.script["lines"][0]["text_us"] = "Your package is here."
        self.write("script.json", self.script)
        self.assertIn("actor: final r1", self.status())
        self.script["lines"][1]["text"] = "Thanks a lot."
        self.write("script.json", self.script)
        self.assertIn("actor: script changed; render pending", self.status())

    def test_adopted_record_names_the_cast_it_was_rendered_in(self):
        runner.adopt_fingerprints(self.job, [self.part])
        self.assertEqual(self.state()["cast"], runner.cast_sha(self.script))
        self.assertIn("actor: final r1", self.status())

    def test_record_of_a_rebuilt_voice_is_left_to_render_again(self):
        before = self.state()
        self.ids["speaker_b"] = "Lsecond-rebuilt"
        runner.adopt_fingerprints(self.job, [self.part])
        self.assertEqual(self.state(), before)
        self.assertIn("actor: a voice changed; render pending", self.status())

    def test_unchanged_delivery_is_not_delivered_again(self):
        self.job["delivered"] = {
            "path": "sample (dubbed).mkv",
            "state": runner.legacy_delivery_sha(self.job, [self.part]),
        }
        runner.adopt_fingerprints(self.job, [self.part])
        self.assertEqual(self.job["delivered"]["state"], runner.delivery_sha(self.job, [self.part]))
        self.assertEqual(self.saved[-1]["delivered"], self.job["delivered"])
        delivered = self.job["delivered"]["state"]
        self.script["lines"][1]["listen"] = "the name may be wrong"
        self.script["notes"] = "heard on the second viewing"
        self.script.pop("ready")
        self.write("script.json", self.script)
        self.assertEqual(runner.delivery_sha(self.job, [self.part]), delivered)

    def test_delivery_changed_since_is_left_to_deliver_again(self):
        self.job["delivered"] = {"path": "sample (dubbed).mkv", "state": "earlier"}
        runner.adopt_fingerprints(self.job, [self.part])
        self.assertEqual(self.job["delivered"]["state"], "earlier")
        self.assertEqual(self.saved, [])

    def test_what_a_delivery_carries_changes_it(self):
        before = runner.delivery_sha(self.job, [self.part])
        self.script["signs"] = [{"start": 0, "end": 2, "text": "Station", "note": "platform two"}]
        self.write("script.json", self.script)
        signed = runner.delivery_sha(self.job, [self.part])
        self.assertNotEqual(signed, before)
        self.script["signs"][0]["note"] = "platform three"
        self.write("script.json", self.script)
        self.assertEqual(runner.delivery_sha(self.job, [self.part]), signed)
        self.write("state.json", {**self.state(), "qa_take_hashes": {"Lfirst": "aa", "Lsecond": "cc"}})
        self.assertNotEqual(runner.delivery_sha(self.job, [self.part]), signed)


if __name__ == "__main__":
    unittest.main()
