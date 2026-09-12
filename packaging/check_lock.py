"""Hold a bundle's dependency versions to the ones that were tested.

The build manifest records what a build *was*. This decides what a build *may
be*: ``packaging/lockfile.txt`` pins every dependency whose version can change
how the application behaves, and this script fails the build when what is
installed does not match it.

Without it the release workflow installs ranges - ``ctranslate2>=4.8,<5`` and
the like - so rebuilding the same commit in six months can produce software
that was never field-tested, with nothing to say so. An air-gapped operational
build is exactly the case where that matters: the bundle on the removable media
is the only copy anyone will ever run.

Usage::

    python packaging/check_lock.py              # verify; non-zero on drift
    python packaging/check_lock.py --write      # re-pin to what is installed

``--write`` is how a dependency update happens: run it on a branch, read the
diff, test the bundle it produces, then merge. An update nobody chose is the
thing this exists to prevent; an update somebody chose is one commit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispr.dependencies import (  # noqa: E402
    TRACKED_PACKAGES,
    canonical,
    drift,
    format_lock,
    parse_lock,
)

LOCKFILE = Path(__file__).resolve().parent / "lockfile.txt"

HEADER = """\
# Operational dependency lock for the Whispers offline bundle.
#
# These are the versions an operational bundle is built from, resolved on the
# build platform (Python 3.11). The point is not that they are the newest: it is
# that the bundle which gets tested and the bundle which gets rebuilt six months
# later are the same software. Whatever is fielded was built from this file.
#
# This is a pip *constraints* file: it does not install anything, it decides
# which version is used for whatever the release workflow does install. So a
# build that never installs the translation engine is unaffected by the pins
# for it.
#
# To change a version deliberately:
#
#     python packaging/check_lock.py --write
#
# on a branch, read the diff, build a bundle, test it, then merge. Nothing else
# in the pipeline may change these, and a build whose installed versions drift
# from this file fails rather than shipping.
#
# Generated and checked by packaging/check_lock.py.
"""


def installed_versions(names: "Optional[List[str]]" = None) -> "Dict[str, str]":
    """Versions of the named packages that are actually installed."""
    from importlib.metadata import PackageNotFoundError, version

    if names is None:
        from importlib.metadata import distributions

        found: Dict[str, str] = {}
        for dist in distributions():
            name = dist.metadata["Name"]
            if name:
                found[canonical(name)] = dist.version
        return found
    out: Dict[str, str] = {}
    for name in names:
        try:
            out[canonical(name)] = version(name)
        except PackageNotFoundError:
            continue
    return out


def write_lock(pins: "Dict[str, str]", path: Path = LOCKFILE) -> None:
    """Write the lock file, one pin per line, sorted."""
    path.write_text(format_lock(pins, HEADER), encoding="utf-8")


def main(argv: "Optional[List[str]]" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="re-pin the lock to the versions currently installed",
    )
    args = parser.parse_args(argv)

    if args.write:
        pins = (
            parse_lock(LOCKFILE.read_text(encoding="utf-8"))
            if LOCKFILE.is_file()
            else {}
        )
        found = installed_versions()
        for name in list(pins) + [canonical(n) for n in TRACKED_PACKAGES]:
            if name in found:
                pins[name] = found[name]
        write_lock(pins)
        print(f"lock -> {LOCKFILE} ({len(pins)} pins)")
        return 0

    if not LOCKFILE.is_file():
        print(f"no lock file at {LOCKFILE}", file=sys.stderr)
        return 1
    pins = parse_lock(LOCKFILE.read_text(encoding="utf-8"))
    installed = installed_versions()
    problems = drift(pins, installed)
    if problems:
        print("dependency lock drift:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nThe bundle would not be the software that was tested. Either fix "
            "the install, or update the lock deliberately with\n"
            "  python packaging/check_lock.py --write\n"
            "on a branch, and test the bundle it produces.",
            file=sys.stderr,
        )
        return 1
    checked = sum(1 for name in pins if name in installed)
    print(
        f"dependency lock: {checked} of {len(pins)} pinned packages installed, all matching"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
