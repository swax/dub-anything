"""Cantonese (yue) language module tests."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from languages import ISO3, LANGUAGES, NAMES, PAIRS, pair_of


class CantoneseLanguageTests(unittest.TestCase):
    def test_languages_are_two_or_three_letter_modules_only(self):
        modules = {p.stem for p in (pathlib.Path(__file__).resolve().parents[1] / "scripts" / "languages").glob("*.py")}
        self.assertEqual(set(LANGUAGES), {m for m in modules if 2 <= len(m) <= 3 and m.isalpha()})
        self.assertTrue({"hanzi", "kana", "latin"} <= modules - set(LANGUAGES))

    def test_cantonese_metadata(self):
        self.assertIn("yue", LANGUAGES)
        yue = LANGUAGES["yue"]
        self.assertEqual(yue.NAME, "Cantonese")
        self.assertEqual(yue.ISO3, "yue")
        self.assertEqual(yue.NATIVE, "粵語")
        self.assertEqual(NAMES["yue"], "Cantonese")
        self.assertEqual(ISO3["yue"], "yue")
        self.assertIn(("en", "yue"), PAIRS)
        self.assertEqual(pair_of({"source_lang": "en", "target_lang": "yue"}), ("en", "yue"))
        self.assertEqual(yue.VOICES, ("actor",))
        self.assertEqual(yue.JOIN, "")

    def test_cantonese_units(self):
        units = LANGUAGES["yue"].units
        self.assertEqual(units("你 好，ＡＢＣ１２！"), units("你好abc12"))
        self.assertNotEqual(units("我哋去"), units("我哋唔去"))

    def test_common_cantonese_written_variants_do_not_fail_qa(self):
        compare = LANGUAGES["yue"].compare
        self.assertEqual(compare("而家我哋出發。", "依家我地出發。")[0], [])
        self.assertEqual(compare("你哋同佢哋去。", "你地同佢地去。")[0], [])
        self.assertEqual(compare("做咩呀？", "做咩啊？")[0], [])

    def test_written_variant_folding_preserves_meaning_checks(self):
        units = LANGUAGES["yue"].units
        self.assertNotEqual(units("我哋出發"), units("你哋出發"))
        self.assertIn("negation differs", LANGUAGES["yue"].compare("我哋唔去。", "我地去。")[0])

    def test_question_particle_is_not_a_negative_imperative(self):
        compare = LANGUAGES["yue"].compare
        self.assertNotIn("negation differs", compare("係咪你？", "係你？")[0])
        self.assertIn("negation differs", compare("咪去。", "去。")[0])
        self.assertIn("negation differs", compare("係。咪去。", "係。去。")[0])

    def test_cantonese_compare(self):
        compare = LANGUAGES["yue"].compare
        # Exact match
        notes, ratio = compare("你好，歡迎收睇。", "你好歡迎收睇")
        self.assertEqual(notes, [])
        self.assertGreaterEqual(ratio, 0.8)

        # Negation check
        notes_neg, _ = compare("我得閒", "我唔得閒")
        self.assertTrue(any("negation differs" in n for n in notes_neg))

        # Number check
        notes_num, _ = compare("有三個", "有四個")
        self.assertTrue(any("match" in n or "number" in n for n in notes_num))

    def test_cantonese_subtitles(self):
        yue = LANGUAGES["yue"]
        text = "我哋今日準備出發，聽日到咗之後再一齊討論跟住落嚟嘅計劃。"
        wrapped = yue.subtitle(text)
        self.assertEqual(wrapped.replace("\n", ""), text)
        self.assertEqual(wrapped.count("\n"), 1)

    def test_changed_number_in_long_line_is_flagged(self):
        notes, ratio = LANGUAGES["yue"].compare("我哋有三個蘋果喺枱面", "我哋有四個蘋果喺枱面")
        self.assertGreaterEqual(ratio, 0.8)
        self.assertTrue(any("number differs" in n for n in notes))

    def test_digit_and_written_numbers_compare_equally(self):
        for written, digits in (
            ("三", "3"),
            ("兩", "2"),
            ("廿三", "23"),
            ("一百零二", "102"),
            ("二〇二六", "2026"),
            ("三點五", "3.5"),
            ("負三", "-3"),
            ("一萬二千三百四十五", "12,345"),
            ("一億二千萬", "120000000"),
        ):
            with self.subTest(written=written):
                notes, ratio = LANGUAGES["yue"].compare(f"有{written}個", f"有{digits}個")
                self.assertEqual(notes, [])
                self.assertEqual(ratio, 1)

    def test_missing_extra_and_changed_decimal_numbers_are_flagged(self):
        for want, got in (
            ("有三個蘋果喺枱面", "有蘋果喺枱面"),
            ("有蘋果喺枱面", "有三個蘋果喺枱面"),
            ("價錢係三點五元", "價錢係三十五元"),
        ):
            with self.subTest(want=want):
                notes, _ = LANGUAGES["yue"].compare(want, got)
                self.assertTrue(any("number differs" in n for n in notes))
