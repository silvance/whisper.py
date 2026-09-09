"""Record every file a built bundle shipped, so a copy of it can be checked.

Run after PyInstaller, against the output directory::

    python packaging/write_inventory.py dist/whispr

Writes ``bundle-inventory.json`` beside the executable. ``whispr --verify``
reads it back on the deployment machine and reports which files are missing,
truncated or altered - the difference between broken software and a copy that
did not survive the journey to an air-gapped machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whispr.bundle_check import INVENTORY_NAME, build_inventory  # noqa: E402


def main(argv: "list[str] | None" = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1:
        print(__doc__)
        return 2
    root = Path(args[0])
    if not root.is_dir():
        print(f"{root} is not a directory", file=sys.stderr)
        return 2
    inventory = build_inventory(root)
    out = root / INVENTORY_NAME
    out.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    size_mb = inventory["total_bytes"] / (1024 * 1024)
    print(
        f"bundle inventory -> {out} ({inventory['file_count']} files, {size_mb:.0f} MB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
