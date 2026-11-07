"""Initialize the pinned VoiceStudio fork and install its Python dependencies.

Run with a standard Python: python scripts/setup.py [--init-only]. Models and
optional GPU configurations are installed separately (docs/setup.md).
"""

import argparse
import os
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUBMODULE = "vendor/VoiceStudio"


def initialize_voicestudio():
    """Check out the superproject's recorded commit, including nested submodules."""
    if not shutil.which("git"):
        raise SystemExit("setup: git not found on PATH (see docs/setup.md)")
    for args in (
        ["submodule", "sync", "--recursive", "--", SUBMODULE],
        ["submodule", "update", "--init", "--recursive", "--", SUBMODULE],
    ):
        subprocess.run(["git", *args], cwd=ROOT, check=True)
    bundled = ROOT / SUBMODULE
    if not (bundled / "backend").is_dir():
        raise SystemExit("setup: VoiceStudio submodule is missing; use a Git clone of Dub Anything (see docs/setup.md)")
    return bundled


def setup(init_only=False):
    bundled = initialize_voicestudio()
    if init_only:
        print(f"setup: pinned VoiceStudio sources ready at {bundled}")
        return
    checkout = pathlib.Path(os.environ.get("VOICESTUDIO_DIR") or bundled).expanduser().resolve()
    if not (checkout / "backend").is_dir():
        raise SystemExit(f"setup: no VoiceStudio checkout at {checkout} (check VOICESTUDIO_DIR)")
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("setup: uv not found on PATH (see docs/setup.md)")
    # Keep the project's environment independent of the Python running setup.
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    subprocess.run([uv, "sync", "--locked"], cwd=checkout, env=env, check=True)
    subprocess.run([uv, "run", "--no-sync", "python", "scripts/setup.py"], cwd=checkout, env=env, check=True)
    print(f"setup: VoiceStudio ready at {checkout}; follow docs/setup.md to install models and start the backend")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--init-only", action="store_true", help="initialize pinned sources without installing dependencies"
    )
    args = parser.parse_args()
    try:
        setup(args.init_only)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"setup: {exc}") from exc


if __name__ == "__main__":
    main()
