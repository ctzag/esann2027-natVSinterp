"""Compare raw/ and out_planA_native/ with data_manifest_sha256.txt (standard library only).

Usage, from the repository root: python scripts/check_manifest.py [MANIFEST]
The manifest has the format of `shasum -a 256`: sha256, two spaces, path relative to the repository root.
Prints MATCH, DIFF or MISSING per file. Exit code 0 only if every file matches.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTE = ("New downloads can differ from the files of the paper: ENTSO-E revises published data, Yahoo Finance revises "
        "past closes, and the coal series is a manual download from investing.com. The package checks the sha256 of "
        "the three model input files on every load and stops on a difference. To run it on your own files, set "
        "ESANN27_ALLOW_HASH_MISMATCH=1: it then warns instead of stopping, and results can differ from the paper's.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    manifest = Path(argv[0]) if argv else ROOT / "data_manifest_sha256.txt"
    counts = {"MATCH": 0, "DIFF": 0, "MISSING": 0}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        ref, rel = line.split("  ", 1)
        p = ROOT / rel
        status = "MISSING" if not p.is_file() else "MATCH" if sha256(p) == ref else "DIFF"
        counts[status] += 1
        print(f"{status:8s}{rel}")
    print(", ".join(f"{v} {k}" for k, v in counts.items()))
    if counts["DIFF"]:
        print("\n" + NOTE)
    return 0 if counts["DIFF"] == counts["MISSING"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
