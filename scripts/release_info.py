"""Release identity shared by the worker, packager and update coordinator."""

import pathlib
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY = "swax/dub-anything"
FEED = f"https://raw.githubusercontent.com/{REPOSITORY}/main/releases/updates.json"
CATALOG = f"https://raw.githubusercontent.com/{REPOSITORY}/main/releases/manifest.json"
PLUGIN_ID = "6d52a0cd-425d-4802-ad0c-368a83da8e45"
PROTOCOL = 2
UPDATER_PROTOCOL = 1


def metadata(root=ROOT):
    props = ET.parse(root / "Directory.Build.props").getroot()
    return {
        "version": props.findtext(".//Version"),
        "jellyfin_min": props.findtext(".//JellyfinVersion") + ".0",
        "jellyfin_max": "13.0.0.0",
        "worker_protocol": PROTOCOL,
        "updater_protocol": UPDATER_PROTOCOL,
    }
