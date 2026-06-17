#!/usr/bin/env python3
"""Check a ZIP/archive and list a small sample of contents without extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip_path", required=True, type=Path, help="Archive to inspect")
    return parser.parse_args()


def _seven_zip() -> str | None:
    return shutil.which("7z") or shutil.which("7za")


def _inspect_with_7z(executable: str, zip_path: Path) -> int:
    test = subprocess.run(
        [executable, "t", str(zip_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print("OK" if test.returncode == 0 else "BAD")
    if test.stdout:
        print(test.stdout.strip())

    listing = subprocess.run(
        [executable, "l", str(zip_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print("First archive listing lines:")
    for line in listing.stdout.splitlines()[:100]:
        print(line)
    return test.returncode


def _inspect_with_zipfile(zip_path: Path) -> int:
    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad_member = archive.testzip()
            print("OK" if bad_member is None else f"BAD: first corrupt member {bad_member}")
            print("First archive files:")
            for name in archive.namelist()[:100]:
                print(name)
            return 0 if bad_member is None else 1
    except zipfile.BadZipFile as exc:
        print(f"BAD: {exc}")
        return 1


def main() -> int:
    args = parse_args()
    zip_path = args.zip_path.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Archive does not exist: {zip_path}")

    executable = _seven_zip()
    if executable is not None:
        print(f"Using {executable} for integrity check")
        return _inspect_with_7z(executable, zip_path)

    print("7z/7za not found; falling back to Python zipfile")
    return _inspect_with_zipfile(zip_path)


if __name__ == "__main__":
    raise SystemExit(main())
