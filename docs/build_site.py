#!/usr/bin/env python3
"""Assemble the cover page, /docs reference, and legacy redirects.

Requires an existing Sphinx HTML build. Output must not already exist.
"""
import argparse
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote


def redirect_page(target):
    escaped = html.escape(target, quote=True)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GLASS documentation has moved</title><link rel="canonical" href="{escaped}">
<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>
</head><body><p>The GLASS documentation has moved. <a href="{escaped}">Continue to this page</a>.</p></body></html>
'''


def assemble(sphinx, landing, output):
    if not (sphinx / "index.html").is_file():
        raise ValueError(f"Missing Sphinx build: {sphinx}")
    if output.exists():
        raise ValueError(f"Output already exists; choose a fresh path: {output}")
    # Keep legacy assets/downloads and /badges URLs at their published paths.
    shutil.copytree(sphinx, output)
    shutil.copytree(sphinx, output / "docs")
    for path in sphinx.rglob("*.html"):
        relative = path.relative_to(sphinx)
        if relative == Path("index.html") or any(part.startswith("_") for part in relative.parts):
            continue
        prefix = "../" * (len(relative.parts) - 1)
        target = prefix + "docs/" + quote(relative.as_posix(), safe="/")
        (output / relative).write_text(redirect_page(target))
    shutil.copytree(landing, output / "landing", ignore=shutil.ignore_patterns("README.md", "index.html"))
    shutil.copyfile(landing / "index.html", output / "index.html")
    (output / ".nojekyll").touch()
    print(f"Review site: {output}")


def main():
    docs = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphinx", type=Path, default=docs / "build" / "html")
    parser.add_argument("--output", type=Path, default=docs / "build" / "site")
    args = parser.parse_args()
    assemble(args.sphinx.resolve(), docs / "landing", args.output.resolve())


if __name__ == "__main__":
    main()
