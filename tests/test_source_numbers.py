"""Japanese number spelling must not reject a clean reference or bypass its speaker check."""

import difflib
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
from languages import ja


class JapaneseSourceNumbersTests(unittest.TestCase):
    def test_equivalent_spellings_have_the_same_text_and_ordered_values(self):
        for want, heard in (
            ("三千六百円です。", "3,600円です。"),
            ("三万六千円です。", "３万６千円です。"),
            ("二〇三一年に三人です。", "２０３１年に３人です。"),
            ("一人で一万円です。", "1人で1万円です。"),
            ("〇三号です。", "03号です。"),
            ("零点五秒です。", "0.5秒です。"),
            ("三点五万円です。", "35,000円です。"),
            ("マイナス三点五度です。", "-3.5度です。"),
        ):
            with self.subTest(want=want, heard=heard):
                self.assertEqual(ja.normalize_source(want), ja.normalize_source(heard))

    def test_names_and_non_numeric_compounds_stay_written_as_they_are(self):
        text = "一緒に三田さんと千葉で十分な時間をとって一般の話をする。"
        self.assertEqual(ja.normalize_source(text), (text, []))

    def test_leading_zeroes_and_changed_signs_remain_distinct(self):
        self.assertNotEqual(ja.normalize_source("03号"), ja.normalize_source("3号"))
        self.assertNotEqual(ja.normalize_source("-3度"), ja.normalize_source("3度"))

    def test_malformed_groups_and_numerals_are_not_guessed(self):
        self.assertNotEqual(ja.normalize_source("1,23円"), ja.normalize_source("123円"))
        self.assertNotEqual(ja.normalize_source("三百二百円"), ja.normalize_source("500円"))
        self.assertNotEqual(ja.normalize_source("零十円"), ja.normalize_source("10円"))

    def test_long_decimal_values_are_compared_without_rounding(self):
        self.assertNotEqual(
            ja.normalize_source("123456789012345678901234567890.1円"),
            ja.normalize_source("123456789012345678901234567890.2円"),
        )


class SourceWindowNumberTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = pathlib.Path(tmp.name)
        self.work = self.root / "sample"
        (self.work / "refs").mkdir(parents=True)
        (self.root / "voices").mkdir()
        job = self.root / "jobs" / "sample"
        job.mkdir(parents=True)
        self.sr = 16000
        self.audio = np.full(2 * self.sr, 0.1, dtype="float32")
        self.audio[self.sr :] = 0.4
        self.source = job / "vocals.wav"
        self.write_source()
        voices = {}
        for key, level in (("speaker_a", 0.1), ("speaker_b", 0.4)):
            voices[key] = {"from": "sample", "clips": [[0, 0.5]], "ref_text": "Reference."}
            sf.write(self.root / "voices" / f"{key}.wav", np.full(self.sr // 2, level), self.sr)
        (self.root / "voices.json").write_text(json.dumps(voices), encoding="utf-8")
        self.lines = [
            {"start": 0.1, "end": 0.9, "voice": "speaker_a", "ja": "三千六百円です。"},
            {"start": 1.1, "end": 1.9, "voice": "speaker_b", "ja": "次です。"},
        ]
        self.heard = "3,600円です。"
        self.model = Mock(return_value=object())
        self.transcribe = Mock(
            side_effect=lambda *args, **kwargs: (
                [{"said": self.heard}] if self.transcribe.call_count % 2 else [{"said": "次です。"}]
            )
        )
        self.embed = Mock(
            side_effect=lambda audio: np.array([1.0, 0.0]) if audio.mean() < 0.25 else np.array([0.0, 1.0])
        )
        self.draft = types.SimpleNamespace(
            SR=self.sr,
            decode_audio=lambda path, sampling_rate: sf.read(path, dtype="float32")[0],
            VadOptions=lambda **kwargs: kwargs,
            get_speech_timestamps=lambda *args: [],
            whisper_on_gpu=lambda: False,
            WhisperModel=self.model,
            transcribe=self.transcribe,
            embedder=lambda: self.embed,
            cast_centroids=lambda *args, **kwargs: {},
        )
        self.enterContext(patch.object(core, "WORK", self.root))
        self.enterContext(patch.object(core, "JOBS", self.root / "jobs"))
        self.enterContext(patch.object(core, "langs", return_value=("ja", "en")))
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    "draft": self.draft,
                    "edges": types.SimpleNamespace(frames_db=lambda audio: np.zeros(20)),
                    "register": types.SimpleNamespace(cer=lambda a, b: 1 - difflib.SequenceMatcher(None, a, b).ratio()),
                },
            )
        )

    def write_source(self):
        sf.write(self.source, self.audio, self.sr)

    def check(self):
        return core.unclean_windows(self.work, "sample", self.lines, ["L00", "L01"])

    def legacy_cache(self, first):
        with patch.object(ja, "SOURCE_CHECK_VERSION", "", create=True):
            keys = core.source_check_keys(self.work, self.lines)
        core.save_json(self.work / "refs" / "clean.json", {keys[0]: first, keys[1]: None})
        return keys

    def test_digits_and_kanji_pass_the_actual_reference_check_in_both_directions(self):
        self.assertEqual(self.check()[0], (None, None))
        self.lines[0]["ja"], self.heard = self.heard, self.lines[0]["ja"]
        self.transcribe.reset_mock()
        self.assertEqual(self.check()[0], (None, None))

    def test_changed_value_is_rejected_even_in_a_long_otherwise_matching_sentence(self):
        prefix = "こちらの説明にある品物の価格をもう一度確認してお伝えします。"
        self.lines[0]["ja"] = prefix + "三千六百円です。"
        self.heard = prefix + "3,700円です。"
        self.assertIn("number differs", self.check()[0][0])

    def test_swapped_values_are_rejected_in_their_original_order(self):
        prefix = "図に書いてある大きさをもう一度確認してお伝えします。"
        self.lines[0]["ja"] = prefix + "横三、縦五です。"
        self.heard = prefix + "横5、縦3です。"
        self.assertIn("number differs", self.check()[0][0])

    def test_wrong_speaker_still_rejects_an_equivalent_number(self):
        self.audio[: self.sr] = 0.4
        self.write_source()
        self.assertEqual(self.check()[0], ("voice nearer speaker_b", None))

    def test_legacy_rejection_is_rechecked_including_the_previously_skipped_speaker(self):
        keys = self.legacy_cache("heard as digits (60% off)")
        self.audio[: self.sr] = 0.4
        self.write_source()
        self.assertEqual(self.check()[0], ("voice nearer speaker_b", None))
        self.assertEqual(self.transcribe.call_count, 2)
        self.assertEqual(core.load_json(self.work / "refs" / "clean.json")[keys[0]], "heard as digits (60% off)")

    def test_legacy_acceptance_does_not_bypass_the_new_numeric_guard(self):
        self.legacy_cache(None)
        self.heard = "3,700円です。"
        self.assertIn("number differs", self.check()[0][0])

    def test_current_decision_is_reused_without_loading_models(self):
        first = self.check()
        self.model.side_effect = AssertionError("A current cache hit must not load Whisper")
        self.assertEqual(self.check(), first)

    def test_unrelated_words_still_have_to_match(self):
        self.heard = "三千六百円で電車は別の駅から出発しました。"
        self.assertIn("heard as", self.check()[0][0])

    def test_numeric_equivalence_does_not_override_a_clipped_source_window(self):
        self.draft.get_speech_timestamps = lambda *args: [{"start": 0, "end": self.sr}]
        self.assertEqual(self.check()[0], ("speech runs on past the window's end", None))

    def test_policy_version_changes_japanese_keys_but_not_other_languages(self):
        japanese = core.source_check_keys(self.work, self.lines)
        with patch.object(core, "langs", return_value=("es", "en")):
            spanish = core.source_check_keys(self.work, self.lines)
        with patch.object(ja, "SOURCE_CHECK_VERSION", "ja-numbers-next"):
            self.assertNotEqual(core.source_check_keys(self.work, self.lines), japanese)
            with patch.object(core, "langs", return_value=("es", "en")):
                self.assertEqual(core.source_check_keys(self.work, self.lines), spanish)


if __name__ == "__main__":
    unittest.main()
