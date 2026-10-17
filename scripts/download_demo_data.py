"""Download a small subset of mdCATH, chosen by a rule fixed in advance.

Default rule: the N smallest domain files in the dataset. Choosing by file size
(rather than by looking at results) keeps the demo honest: nothing is
cherry-picked. The rule, the dataset revision and a SHA-256 for every file are
saved to data/manifest.json so the evidence passports can cite them.

    pip install huggingface_hub
    python scripts/download_demo_data.py --n 8 --dry-run    # see what would download
    python scripts/download_demo_data.py --n 8
    python scripts/download_demo_data.py --ids 1a92A00 1a0aA00
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

REPO = "compsciencelab/mdCATH"
PREFIX = "data/mdcath_dataset_"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def file_size(entry) -> int:
    size = getattr(entry, "size", None)
    if size:
        return int(size)
    lfs = getattr(entry, "lfs", None)
    return int(getattr(lfs, "size", 0) or 0) if lfs is not None else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=8, help="how many of the smallest domain files to fetch")
    parser.add_argument("--ids", nargs="*", help="specific domain ids instead of the smallest-files rule")
    parser.add_argument("--root", default=".", help="repository root; files land in <root>/data/")
    parser.add_argument("--dry-run", action="store_true", help="list files and sizes without downloading")
    args = parser.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    revision = api.dataset_info(REPO).sha
    print(f"mdCATH revision {revision}")

    entries = [
        e
        for e in api.list_repo_tree(REPO, path_in_repo="data", repo_type="dataset", revision=revision)
        if e.path.startswith(PREFIX) and e.path.endswith(".h5")
    ]
    sizes = {e.path: file_size(e) for e in entries}
    print(f"{len(sizes)} domain files listed")

    if args.ids:
        chosen = [f"{PREFIX}{i}.h5" for i in args.ids]
        missing = [c for c in chosen if c not in sizes]
        if missing:
            raise SystemExit(f"Not in the dataset: {', '.join(missing)}")
        rule = f"explicit domain ids: {', '.join(args.ids)}"
    else:
        chosen = sorted(sizes, key=lambda p: (sizes[p], p))[: args.n]
        rule = f"the {args.n} smallest domain files in mdCATH (by file size), revision {revision}"

    total = sum(sizes[c] for c in chosen)
    for c in chosen:
        print(f"  {Path(c).name:<34} {sizes[c] / 1e6:8.1f} MB")
    print(f"Total: {total / 1e9:.2f} GB  |  rule: {rule}")
    if args.dry_run:
        return

    root = Path(args.root)
    records = []
    for c in chosen:
        local = Path(hf_hub_download(REPO, c, repo_type="dataset", revision=revision, local_dir=root))
        print(f"downloaded {local}")
        records.append({"path": c, "size_bytes": sizes[c], "sha256": sha256_file(local)})

    manifest = {
        "dataset": REPO,
        "revision": revision,
        "licence": "CC BY 4.0",
        "citation": "Mirarchi, A., Giorgino, T. & De Fabritiis, G. Scientific Data 11, 1299 (2024)",
        "selection_rule": rule,
        "downloaded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": records,
    }
    manifest_path = root / "data" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
