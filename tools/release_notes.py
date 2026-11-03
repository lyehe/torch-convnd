"""Print a release's changelog section, the body of its GitHub release."""

import argparse
import re
from pathlib import Path


def release_notes(changelog: str, version: str) -> str:
    """Return the entries under ``## <version>``, up to the next release heading."""
    heading = re.compile(rf"^## {re.escape(version)}[ \t]*$\n(.*?)(?=^## |\Z)", re.M | re.S)
    match = heading.search(changelog)
    if match is None or not match.group(1).strip():
        raise ValueError(f"CHANGELOG.md has no entries for {version}")
    return match.group(1).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="release version or tag, such as 0.6.0 or v0.6.0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    try:
        notes = release_notes(changelog, args.version.removeprefix("v"))
    except ValueError as error:
        parser.error(str(error))
    print(notes, end="")


if __name__ == "__main__":
    main()
