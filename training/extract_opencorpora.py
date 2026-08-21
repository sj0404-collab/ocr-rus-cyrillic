"""Stream Russian sentences from the OpenCorpora 2025 tar.gz package."""

from __future__ import annotations

import argparse
import re
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-sentences", type=int, default=100000)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[str] = set()
    with tarfile.open(args.archive, "r:gz") as archive:
        member = next((m for m in archive.getmembers() if m.name.endswith("opencorpora_annot_2025.xml")), None)
        if member is None:
            raise FileNotFoundError("OpenCorpora XML not found in archive")
        stream = archive.extractfile(member)
        if stream is None:
            raise OSError("Could not open OpenCorpora XML member")
        with stream, args.output.open("w", encoding="utf-8") as out:
            for _event, element in ET.iterparse(stream, events=("end",)):
                if not element.tag.endswith("sentence"):
                    continue
                source = element.findtext("source", default="")
                source = re.sub(r"\s+", " ", source).strip()
                if len(source) >= 12 and CYRILLIC.search(source) and source not in seen:
                    out.write(source + "\n")
                    seen.add(source)
                    count += 1
                    if count >= args.max_sentences:
                        break
                element.clear()
    print(f"Extracted {count} Russian sentences to {args.output}")


if __name__ == "__main__":
    main()
