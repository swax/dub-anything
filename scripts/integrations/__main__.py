"""Run an integration component: python scripts/integrations <component> ..."""

import argparse
import pathlib
import runpy
import sys

# Directory execution works from any working directory and avoids changing the
# established imports used by the shared dubbing engine in scripts/.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
COMPONENTS = {"service": "dub_service", "updates": "dub_updates", "build-release": "build_release"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=COMPONENTS)
    args = parser.parse_args(sys.argv[1:2])
    sys.argv.pop(1)
    runpy.run_module("integrations." + COMPONENTS[args.component], run_name="__main__")


if __name__ == "__main__":
    main()
