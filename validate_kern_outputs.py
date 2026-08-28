#!/usr/bin/env python3
"""Lightweight structural checks for generated Humdrum files."""

from __future__ import annotations

import argparse
from pathlib import Path


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    spine_width: int | None = None
    saw_exclusive = False
    saw_terminator = False

    for line_number, line in enumerate(lines, start=1):
        if not line:
            errors.append(f"{line_number}: blank line")
            continue
        if line.startswith("!!!") or line.startswith("!!"):
            continue

        fields = line.split("\t")
        if spine_width is None:
            spine_width = len(fields)
        elif len(fields) != spine_width:
            errors.append(
                f"{line_number}: expected {spine_width} spine fields, found {len(fields)}"
            )

        if line.startswith("**kern\t**mxhm"):
            saw_exclusive = True
        if line == "*-\t*-":
            saw_terminator = True

    if not saw_exclusive:
        errors.append("missing **kern/**mxhm exclusive interpretation")
    if not saw_terminator:
        errors.append("missing *- terminator")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generated .krn files.")
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories to validate.")
    args = parser.parse_args()

    files: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            files.extend(sorted(path.glob("*.krn")))
        else:
            files.append(path)

    checked = 0
    failed = 0
    for file_path in files:
        checked += 1
        errors = validate_file(file_path)
        if errors:
            failed += 1
            print(file_path)
            for error in errors[:20]:
                print(f"  {error}")
            if len(errors) > 20:
                print(f"  ... {len(errors) - 20} more")

    print(f"Validated {checked} files; failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
