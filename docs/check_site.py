#!/usr/bin/env python3
"""Check assembled local links, paper asset hashes, and legacy redirect targets."""
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for attr in ("href", "src"):
            if attr in attrs:
                self.links.append(attrs[attr])


def check(root):
    errors = []
    # Theme template fragments copied as static assets are not rendered pages.
    pages = [p for p in root.rglob("*.html")
             if not any(part.startswith("_") for part in p.relative_to(root).parts)]
    for path in pages:
        parser = Links()
        parser.feed(path.read_text())
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (root / unquote(url.path.lstrip("/")) if url.path.startswith("/")
                      else path.parent / unquote(url.path)) if url.path else path
            if target.is_dir():
                target = target / "index.html"
            if not target.exists():
                errors.append(f"{path.relative_to(root)}: missing {link}")
            if path == root / "index.html" and not url.path and url.fragment:
                if unquote(url.fragment) not in parser.ids:
                    errors.append(f"Landing page missing anchor: {link}")
    for doc in (root / "docs").rglob("*.html"):
        rel = doc.relative_to(root / "docs")
        if str(rel) == "index.html" or any(part.startswith("_") for part in rel.parts):
            continue
        legacy = root / rel
        p = Links()
        p.feed(legacy.read_text())
        expected = doc.resolve()
        if not any((legacy.parent / unquote(urlsplit(link).path)).resolve() == expected
                   for link in p.links):
            errors.append(f"Legacy redirect target mismatch: {rel}")
        if "location.search + location.hash" not in legacy.read_text():
            errors.append(f"Legacy query/fragment preservation missing: {rel}")
    paper = root / "docs" / "_static" / "paper"
    manifest = json.loads((paper / "provenance.json").read_text())
    for stem, record in manifest["figures"].items():
        for ext in ("pdf", "png"):
            digest = hashlib.sha256((paper / (stem + "." + ext)).read_bytes()).hexdigest()
            if digest != record[ext + "_sha256"]:
                errors.append(f"Paper figure hash mismatch: {stem}.{ext}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(pages)} HTML pages; local targets, legacy redirects, and paper asset hashes")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
