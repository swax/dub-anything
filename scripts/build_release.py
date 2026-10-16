"""Build public release archives and both feeds without including local job data."""

import argparse
import hashlib
import json
import pathlib
import subprocess
import zipfile
from datetime import datetime, timezone

import release_info

ROOT = release_info.ROOT
RUNTIME_DIRS = {"scripts", "docs", "tests", "eval", "patches"}
RUNTIME_FILES = {"AGENTS.md", "CLAUDE.md", "README.md", "LICENSE", "ruff.toml", "Directory.Build.props"}


def digest(path, algorithm="sha256"):
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def runtime_files(root=ROOT):
    paths = subprocess.check_output(["git", "ls-files", "-z", "--cached"], cwd=root).decode().split("\0")
    for name in sorted(set(paths)):
        path = pathlib.PurePosixPath(name)
        if not name or (path.parts[0] not in RUNTIME_DIRS and name not in RUNTIME_FILES):
            continue
        if name not in RUNTIME_FILES and path.suffix.lower() not in (".py", ".md", ".toml", ".patch"):
            continue
        if any(part.startswith(".") for part in path.parts):
            continue
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Not a regular release source file: {name}")
        yield name, source


def build(output, notes, root=ROOT):
    identity = release_info.metadata(root)
    version = identity["version"]
    output.mkdir(parents=True, exist_ok=True)
    dll = root / "integrations/jellyfin/bin/Release/net10.0/Jellyfin.Plugin.DubAnything.dll"
    if not dll.is_file():
        raise ValueError("Build the Release DLL first")
    timestamp = datetime.now(timezone.utc).isoformat()
    plugin_meta = {
        "guid": release_info.PLUGIN_ID,
        "name": "Dub Anything",
        "version": version,
        "targetAbi": identity["jellyfin_min"],
        "owner": "swax",
        "category": "General",
        "description": "Dub library videos with a Codex or Claude Code director.",
        "overview": "Directed dubbing with cloned voices",
        "changelog": notes,
        "timestamp": timestamp,
        "status": "Active",
        "autoUpdate": False,
        "assemblies": [dll.name],
    }
    plugin = output / f"dub-anything-jellyfin-{version}.zip"
    with zipfile.ZipFile(plugin, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(dll, dll.name)
        archive.writestr("meta.json", json.dumps(plugin_meta))
    bundle = output / f"dub-anything-{version}.zip"
    hashes = {}
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in runtime_files(root):
            hashes[name] = digest(path)
            archive.write(path, "runtime/" + name)
        archive.write(dll, "plugin/" + dll.name)
        archive.writestr("plugin/meta.json", json.dumps(plugin_meta))
        archive.writestr("release.json", json.dumps({**identity, "files": hashes}))
    base = f"https://github.com/{release_info.REPOSITORY}/releases/download/jellyfin-v{version}"
    entry = {
        **identity,
        "url": base + "/" + bundle.name,
        "sha256": digest(bundle),
        "notes": notes,
        "timestamp": timestamp,
    }
    feeds = root / "releases"
    previous_path = feeds / "updates.json"
    previous = (
        json.loads(previous_path.read_text(encoding="utf-8"))
        if previous_path.exists()
        else {"schema": 1, "releases": []}
    )
    if any(item["version"] == version for item in previous["releases"]):
        raise ValueError("This version is already published; bump Directory.Build.props")
    updates = {"schema": 1, "releases": [entry, *previous["releases"]]}
    manifest_path = feeds / "manifest.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8"))[0]["versions"] if manifest_path.exists() else []
    manifest = [
        {
            "guid": release_info.PLUGIN_ID,
            "name": "Dub Anything",
            "owner": "swax",
            "category": "General",
            "description": plugin_meta["description"],
            "overview": plugin_meta["overview"],
            "versions": [
                {
                    "version": version,
                    "targetAbi": identity["jellyfin_min"],
                    "sourceUrl": base + "/" + plugin.name,
                    "checksum": digest(plugin, "md5"),
                    "timestamp": timestamp,
                    "changelog": notes,
                },
                *old,
            ],
        }
    ]
    write_json(output / "updates.json", updates)
    write_json(output / "manifest.json", manifest)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "work/release")
    parser.add_argument("--notes", type=pathlib.Path, default=ROOT / "releases/notes.md")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.notes.read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
