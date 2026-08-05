#!/usr/bin/env python3
"""Build the `bipower_core` C++ extension, on any of the three supported OSes.

    python build.py              # configure + build into python/
    python build.py --clean      # discard the build tree first
    python build.py --debug      # unoptimised, with assertions

This replaces the shell script it grew out of, because CI builds the same module
on Windows and `build.sh` could not. Three portability details it exists to
handle, each of which silently produces a broken or missing module rather than
an error:

* **The interpreter.** CMake's `FindPython` searches the system before it
  searches a virtualenv, so a build launched from an activated venv can link
  against a *different* Python than the one that will import the result. Passing
  `sys.executable` explicitly is what pins them together.
* **Multi-config generators.** MSBuild and Xcode choose the configuration at
  build time, not configure time, so `--config` has to be passed to the build
  step as well; Make and Ninja take it at configure time and ignore it later.
* **Stale modules.** An extension built for a previous Python version keeps its
  old ABI tag, so it lingers next to the new one and imports in preference to
  it on some paths. Cleaning removes every `bipower_core.*` rather than the one
  name this interpreter happens to produce.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BUILD_DIR = REPO_ROOT / "build"
MODULE_DIR = REPO_ROOT / "python"
MODULE_SUFFIXES = ("*.so", "*.pyd", "*.dylib")


def run(command: list[str]) -> None:
    """Run a command, echoing it first so a CI log shows what was attempted."""
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT)


def clean_modules() -> None:
    """Remove previously built extensions, whatever ABI tag they carry."""
    for directory in (MODULE_DIR, REPO_ROOT):
        for pattern in MODULE_SUFFIXES:
            for stale in directory.glob(f"bipower_core{pattern}"):
                print(f"removing {stale.relative_to(REPO_ROOT)}")
                stale.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true",
                        help="delete the build tree before configuring")
    parser.add_argument("--debug", action="store_true",
                        help="build unoptimised, with assertions enabled")
    parser.add_argument("--jobs", type=int, default=0,
                        help="parallel build jobs; 0 lets CMake decide")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the post-build import check")
    args = parser.parse_args()

    if shutil.which("cmake") is None:
        print("error: cmake is not on PATH. Install CMake 3.18+ and a C++20 compiler.",
              file=sys.stderr)
        return 1

    if args.clean and BUILD_DIR.exists():
        print(f"removing {BUILD_DIR.relative_to(REPO_ROOT)}/")
        shutil.rmtree(BUILD_DIR)
    clean_modules()

    config = "Debug" if args.debug else "Release"
    run([
        "cmake",
        "-S", str(REPO_ROOT),
        "-B", str(BUILD_DIR),
        f"-DCMAKE_BUILD_TYPE={config}",
        # The interpreter that will import the module is the one to build against.
        f"-DPython_EXECUTABLE={sys.executable}",
    ])

    build = ["cmake", "--build", str(BUILD_DIR), "--config", config]
    if args.jobs:
        build += ["--parallel", str(args.jobs)]
    run(build)

    # `compile_commands.json` is what clangd and other tooling read; CMake writes
    # it into the build tree, and editors look for it at the repository root.
    compile_commands = BUILD_DIR / "compile_commands.json"
    if compile_commands.exists():
        shutil.copy2(compile_commands, REPO_ROOT / "compile_commands.json")

    built = sorted(
        path for pattern in MODULE_SUFFIXES for path in MODULE_DIR.glob(f"bipower_core{pattern}")
    )
    if not built:
        print(f"error: build reported success but no module landed in {MODULE_DIR}",
              file=sys.stderr)
        return 1
    print(f"built {', '.join(p.name for p in built)}")

    if not args.no_verify:
        # Import in a subprocess rooted at `python/`, which is how every script in
        # this project resolves the module — an import that only works from the
        # repository root is not the one that matters.
        subprocess.run(
            [sys.executable, "-c",
             "import bipower_core; print('bipower_core imports:', bipower_core.__doc__)"],
            check=True, cwd=MODULE_DIR,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
