"""New VoiceStudio language helpers must coexist with the dubbing core module."""

import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import core


class VoiceStudioNormalizerTests(unittest.TestCase):
    def test_backend_language_helpers_do_not_replace_dubbing_core(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = pathlib.Path(tmp)
            (backend / "core").mkdir()
            (backend / "services").mkdir()
            (backend / "core" / "language_variants.py").write_text(
                "def dialect_for(language):\n    return 'pt-BR'\n", encoding="utf-8"
            )
            (backend / "services" / "text_normalization.py").write_text(
                "def normalize_text(text, language):\n"
                "    from core.language_variants import dialect_for\n"
                "    return text + ':' + dialect_for(language)\n",
                encoding="utf-8",
            )
            original_root = core.ROOT
            with (
                patch.object(core, "__path__", [], create=True),
                patch.object(core, "language_variants", None, create=True),
                patch.dict(sys.modules),
            ):
                sys.modules.pop("core.language_variants", None)
                normalize = core._voicestudio_normalizer(backend)
                self.assertEqual(normalize("12", "Portuguese"), "12:pt-BR")
                self.assertIs(sys.modules["core"], core)
                self.assertEqual(core.ROOT, original_root)
                self.assertTrue(callable(core.take_length))
