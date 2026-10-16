"""The UI accepts language names; new code crosses a checked, restricted boundary."""

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from integrations import language_setup as setup


class LanguageSetupTests(unittest.TestCase):
    def test_known_names_native_names_and_codes_resolve(self):
        for name in ("English", "eng", "en", "ENGLISH"):
            self.assertEqual(setup.language(name), "en")
        self.assertEqual(setup.language("日本語"), "ja")
        self.assertEqual(setup.language("French"), "French")
        self.assertEqual(setup.language("fr"), "fr")

    def test_language_inputs_are_not_shell_fragments(self):
        for value in ("", "French; run command", "en123", [], "x" * 81):
            with self.subTest(value=value), self.assertRaises(ValueError):
                setup.language(value)

    def test_unknown_pairs_enter_setup(self):
        self.assertTrue(setup.supported("ja", "en"))
        self.assertFalse(setup.supported("French", "en"))
        self.assertFalse(setup.supported("ja", "es"))

    def test_only_language_code_docs_and_new_tests_can_be_published(self):
        for path in (
            "scripts/languages/fr.py",
            "scripts/languages/__init__.py",
            "docs/languages/fr.md",
            "docs/languages/README.md",
            "tests/test_language_french.py",
        ):
            self.assertTrue(setup.allowed(path), path)
        for path in (
            "scripts/backend.py",
            "AGENTS.md",
            "work/voices.json",
            "scripts/languages/../../bad.py",
            "scripts/languages/latin.py",
            "tests/test_dub_service.py",
        ):
            self.assertFalse(setup.allowed(path), path)

    def fixture(self, root, work):
        for base in (root, work):
            for folder in ("scripts/languages", "docs/languages", "tests"):
                (base / folder).mkdir(parents=True)
        original = root / "scripts/languages/__init__.py"
        original.write_text("# original registry\n", encoding="utf-8")
        (work / "scripts/languages/__init__.py").write_text("# updated registry\n", encoding="utf-8")
        for code in ("fr", "en"):
            (work / "scripts/languages" / f"{code}.py").write_text("# generic module\n", encoding="utf-8")
            (work / "docs/languages" / f"{code}.md").write_text("Generic language notes\n", encoding="utf-8")
        return {"scripts/languages/__init__.py": setup.digest(original)}

    def test_validated_language_files_can_be_imported_without_media(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work = pathlib.Path(tmp) / "repo", pathlib.Path(tmp) / "scratch"
            hashes = self.fixture(root, work)
            changed = setup.apply_support(work, hashes, "fr", "en", root)
            self.assertIn("scripts/languages/fr.py", changed)
            self.assertEqual((root / "scripts/languages/fr.py").read_text(), "# generic module\n")

    def test_concurrent_user_changes_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work = pathlib.Path(tmp) / "repo", pathlib.Path(tmp) / "scratch"
            hashes = self.fixture(root, work)
            (root / "scripts/languages/__init__.py").write_text("# user edit\n")
            with self.assertRaisesRegex(ValueError, "changed while"):
                setup.apply_support(work, hashes, "fr", "en", root)
            self.assertFalse((root / "scripts/languages/fr.py").exists())

    def test_generic_tool_changes_require_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, work = pathlib.Path(tmp) / "repo", pathlib.Path(tmp) / "scratch"
            hashes = self.fixture(root, work)
            (work / "scripts/backend.py").write_text("# unrelated change\n")
            with self.assertRaisesRegex(ValueError, "normal code review"):
                setup.apply_support(work, hashes, "fr", "en", root)


if __name__ == "__main__":
    unittest.main()
