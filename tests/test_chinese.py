"""Chinese source support and the independent recogniser's model contract."""

import pathlib
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from languages import ISO3, LANGUAGES, PAIR, PAIRS, pair_of


class ChineseSourceTests(unittest.TestCase):
    def test_pair_and_delivery_metadata(self):
        zh = LANGUAGES["zh"]
        self.assertIn(("zh", "en"), PAIRS)
        self.assertNotIn(("en", "zh"), PAIRS)
        self.assertEqual(pair_of({"source_lang": "zh", "target_lang": "en"}), ("zh", "en"))
        self.assertEqual(pair_of({}), PAIR)
        self.assertEqual(ISO3["zh"], "zho")
        self.assertEqual(zh.OCR, ("CH", "SERVER"))
        self.assertIn("中文", zh.TITLES["transcript"])
        self.assertEqual(len(zh.SECOND_OPINIONS), 2)
        self.assertIn("actor", LANGUAGES["en"].VOICES)

    def test_character_units_ignore_formatting_but_keep_word_changes(self):
        units = LANGUAGES["zh"].units
        self.assertEqual(units("你 好，ＡＢＣ１２！"), units("你好abc12"))
        self.assertNotEqual(units("我去"), units("我不去"))
        self.assertNotEqual(units("三个人"), units("四个人"))
        self.assertNotEqual(units("张"), units("章"))

    def test_reference_window_check_ignores_chinese_punctuation(self):
        from core import bare

        self.assertEqual(bare("你好，世界；今天：【出发】。"), bare("你好世界今天出发"))

    def test_subtitles_preserve_text_and_existing_breaks(self):
        zh = LANGUAGES["zh"]
        self.assertEqual(zh.subtitle("你好。"), "你好。")
        self.assertEqual(zh.subtitle("第一行\n第二行"), "第一行\n第二行")
        text = "我们今天准备出发，明天到达以后再一起讨论接下来的计划。"
        wrapped = zh.subtitle(text)
        self.assertEqual(wrapped.replace("\n", ""), text)
        self.assertEqual(wrapped.count("\n"), 1)

    def test_digit_and_written_numbers_compare_equally(self):
        for written, digits in (
            ("三", "3"),
            ("两", "2"),
            ("一百零二", "102"),
            ("三点五", "3.5"),
            ("一万二千", "12,000"),
        ):
            with self.subTest(written=written):
                self.assertEqual(LANGUAGES["zh"].compare(f"有{written}个", f"有{digits}个"), ([], 1))

    def test_missing_extra_and_changed_numbers_are_flagged(self):
        for want, got in (
            ("我们有三个苹果在桌上", "我们有四个苹果在桌上"),
            ("我们有三个苹果在桌上", "我们有苹果在桌上"),
            ("我们有苹果在桌上", "我们有三个苹果在桌上"),
            ("价格是三点五元", "价格是三十五元"),
        ):
            with self.subTest(want=want, got=got):
                notes, _ = LANGUAGES["zh"].compare(want, got)
                self.assertTrue(any("number differs" in n for n in notes))

    def test_english_take_qa_still_checks_negation_and_numbers(self):
        compare = LANGUAGES["en"].compare
        self.assertEqual(compare("I have 12 books.", "I have twelve books.")[0], [])
        self.assertIn("negation differs", compare("I can go.", "I can't go.")[0])
        self.assertTrue(any("number differs" in n for n in compare("I have 12 books.", "I have thirteen books.")[0]))


class ChineseRecognizerTests(unittest.TestCase):
    def test_zipformer_ctc_loads_its_single_model_not_transducer_files(self):
        import second

        with (
            patch.object(second, "snapshot_download", return_value="model-cache") as download,
            patch.object(second.sherpa_onnx.OfflineRecognizer, "from_zipformer_ctc") as build,
        ):
            result = second.recogniser("zipformer_ctc", LANGUAGES["zh"].SECOND_OPINIONS["zipformer_ctc"], "zh")
        self.assertIs(result, build.return_value)
        self.assertEqual(download.call_args.kwargs["allow_patterns"], ["model.int8.onnx", "tokens.txt"])
        self.assertEqual(pathlib.Path(build.call_args.kwargs["model"]).name, "model.int8.onnx")
        self.assertEqual(build.call_args.kwargs["sample_rate"], 16000)
        self.assertEqual(build.call_args.kwargs["feature_dim"], 80)
        self.assertEqual(build.call_args.kwargs["decoding_method"], "greedy_search")
        self.assertAlmostEqual(second.dist("你 好，世界！", "你好世界", "zh"), 0)


if __name__ == "__main__":
    unittest.main()
