"""Failed attempts must not relabel the settings of published takes."""

import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from dub import _record_published_settings


class PublishedSettingsTests(unittest.TestCase):
    def record(self, rendered, *, published, selected=None, modified=10, regen=None):
        wav = SimpleNamespace(exists=lambda: True, stat=lambda: SimpleNamespace(st_mtime=modified))
        with patch("dub.take_wav", return_value=wav):
            _record_published_settings(
                rendered,
                {"a": {"speed": 0.8}, "b": {"speed": 1.1}},
                "job",
                "en",
                10,
                regen,
                published=published,
                selected=selected,
            )

    def test_failed_request_keeps_settings_even_for_recent_files(self):
        rendered = {"a": {"speed": 1.0}}
        self.record(rendered, published=False)
        self.assertEqual(rendered, {"a": {"speed": 1.0}})

    def test_success_records_only_selected_published_takes(self):
        rendered = {"a": {"speed": 1.0}}
        self.record(rendered, published=True, selected={"b"})
        self.assertEqual(rendered, {"a": {"speed": 1.0}, "b": {"speed": 1.1}})

    def test_success_can_record_reused_old_takes(self):
        rendered = {}
        self.record(rendered, published=True, modified=0, regen=["a"])
        self.assertEqual(rendered, {"b": {"speed": 1.1}})
