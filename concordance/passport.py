"""Evidence passports: a small JSON record that lets anyone replay an answer.

The passport id is a hash of the question, data fingerprints, method, result and
software versions. The same question on the same data with the same code always
gets the same id; change anything and the id changes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from . import __version__

MDCATH_SOURCE = {
    "name": "mdCATH",
    "repository": "https://huggingface.co/datasets/compsciencelab/mdCATH",
    "citation": "Mirarchi, A., Giorgino, T. & De Fabritiis, G. mdCATH: A Large-Scale MD Dataset for "
    "Data-Driven Computational Biophysics. Scientific Data 11, 1299 (2024).",
    "licence_note": "The Scientific Data paper states CC BY 4.0; some Hugging Face card revisions say CC BY-ND 4.0. "
    "Concordance redistributes no mdCATH data.",
}

LIMITATIONS = [
    "Measures agreement between replicate runs, not whether the force field or the physics is right.",
    "Agreement is not proof: runs that start from the same structure can agree and still all be wrong. "
    "Disagreement, though, is a clear warning.",
    "Five replicates is a small sample; the thresholds are explicit conventions, not statistical guarantees.",
    "mdCATH's replicates at one temperature share one equilibrated starting structure, so they are not fully independent.",
]

_HASH_CACHE: dict[tuple[str, float, int], str] = {}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime, stat.st_size)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(chunk), b""):
                digest.update(block)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def load_manifest(data_dir: str | Path) -> dict | None:
    path = Path(data_dir) / "manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def software_versions() -> dict:
    versions = {"python": platform.python_version(), "concordance": __version__}
    for package in ("MDAnalysis", "numpy", "scipy", "h5py"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True, text=True, check=True
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def source_record(an, data_dir: str | Path | None) -> dict:
    rset = an.rset
    record = {
        "kind": rset.kind,
        "name": rset.name,
        "condition": rset.condition,
        "replicas": len(rset.replica_ids),
        "synthetic_test_data": rset.synthetic,
        "files": [{"file": Path(p).name, "sha256": sha256_file(p)} for p in rset.data_files],
    }
    if rset.kind == "files":
        record["unwrapped_by_concordance"] = rset.unwrap
    if rset.kind == "mdcath":
        record["dataset"] = MDCATH_SOURCE
        manifest = load_manifest(data_dir) if data_dir else None
        if manifest:
            record["dataset_revision"] = manifest.get("revision")
            record["selection_rule"] = manifest.get("selection_rule")
    return record


def build_passport(an, *, question: dict, result: dict, replay: str, data_dir: str | Path | None = None) -> dict:
    passport = {
        "tool": "concordance",
        "question": question,
        "source": source_record(an, data_dir),
        "window": an.window.as_dict(),
        "method": an.method.as_dict(),
        "engine_check": {k: v for k, v in an.check.items() if k != "per_run"},
        "chain_check": an.geometry,
        "result": result,
        "replay": replay,
        "limitations": LIMITATIONS,
        "software": software_versions(),
    }
    canonical = json.dumps(passport, sort_keys=True, separators=(",", ":"), default=str)
    passport["passport_id"] = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    passport["git_commit"] = git_commit()
    passport["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    return passport


def write_passport(passport: dict, out_dir: str | Path) -> Path:
    folder = Path(out_dir) / "passports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{passport['source']['name']}_{passport['passport_id']}.json"
    path.write_text(json.dumps(passport, indent=2, default=str) + "\n")
    return path


def find_passport(passport_id: str, out_dir: str | Path) -> dict | None:
    for path in (Path(out_dir) / "passports").glob(f"*_{passport_id}.json"):
        return json.loads(path.read_text())
    return None
