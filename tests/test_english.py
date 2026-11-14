"""The drink's pronunciation cue must not change the ordinary English idiom."""

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from languages import en


class BeveragePronunciationTests(unittest.TestCase):
    def test_the_drink_has_two_syllables_without_rewriting_the_idiom(self):
        self.assertEqual(
            en.speakable("Saké, please. For her sake, wait. It costs 150,000 yen."),
            "sah-kay, please. For her sake, wait. It costs 150000 yen.",
        )
        self.assertEqual(en.speakable("Sake\u0301, please."), "sah-kay, please.")

    def test_qa_accepts_the_beverage_spellings(self):
        for heard in (
            "Two bottles of sake.",
            "Two bottles of sah-kay.",
            "Two bottles of sa kay.",
            "Two bottles of sake\u0301.",
        ):
            with self.subTest(heard=heard):
                self.assertEqual(en.compare("Two bottles of saké.", heard)[0], [])

    def test_the_pronunciation_alias_does_not_hide_meaning_changes(self):
        self.assertTrue(en.compare("Two bottles of saké.", "Three bottles of sake.")[0])
        self.assertTrue(en.compare("Do not drink saké.", "Do drink sake.")[0])


if __name__ == "__main__":
    unittest.main()
