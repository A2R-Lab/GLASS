#!/usr/bin/env python3
"""Import existing release figures and render web previews; no benchmark runs."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

FIGURES = (
    "device_vendor_vs_full_glass_orin", "competitive_controls",
    "tier_heatmap_three_arch", "policy_transfer_native_matrix",
    "host_performance_orin_heatmap", "riccati_tradeoff_orin",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, required=True)
    source = parser.parse_args().paper_dir.resolve()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip():
        raise SystemExit("Paper checkout must be clean to identify the imported figures.")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    out = Path(__file__).parent / "source" / "_static" / "paper"
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for stem in FIGURES:
        src = source / "figs" / (stem + ".pdf")
        pdf = out / src.name
        shutil.copyfile(src, pdf)
        subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-scale-to", "1800",
                        "-png", str(pdf), str(out / stem)], check=True)
        records[stem] = {"source": "figs/" + src.name, "pdf_sha256": sha256(pdf),
                         "png_sha256": sha256(out / (stem + ".png"))}
    (out / "provenance.json").write_text(json.dumps({
        "paper_repository": "https://github.com/plancherb1/glass-paper-arxiv",
        "paper_commit": commit,
        "conversion": "pdftoppm -f 1 -singlefile -scale-to 1800 -png",
        "figures": records,
    }, indent=2) + "\n")
    print(f"Imported {len(records)} figures from {commit}")


if __name__ == "__main__":
    main()
