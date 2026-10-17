"""Make the example data for the no-install page (concordance/assets/example.json).

Five synthetic replicate runs with a designed answer, laid onto a real protein shape so the 3D
views look like a protein: adenylate kinase (PDB 4AKE, backbone only, scripts/adk_4ake_backbone.pdb).
The per-residue RMSF values are made up, written in the layout of `gmx rmsf -res`:

  residues 122-159 (the LID domain)  move a lot in all five runs
  residues 30-59   (the NMP domain)  move a lot in runs 1-3 only
  everything else                    small, noisy movement

The pattern is loosely inspired by adenylate kinase's two mobile lids, but the numbers are not
from a simulation, and the page says so. No real dataset is included.
Run: uv run python scripts/make_example.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "concordance" / "assets" / "example.json"
STRUCTURE = HERE / "adk_4ake_backbone.pdb"
LID, NMP, SOME_RUNS = (122, 159), (30, 59), (0, 1, 2)


def residue_numbers(pdb_text: str) -> list[int]:
    seen = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            seen.append(int(line[22:26]))
    return seen


def make_runs(numbers: list[int], seed: int = 4) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.asarray(numbers, dtype=float)
    shape = 0.055 + 0.012 * np.sin(x / 9.0) + 0.008 * np.cos(x / 4.3)  # a gently varying rigid baseline (nm)
    runs = []
    for k in range(5):
        values = shape * rng.lognormal(0.0, 0.12, size=len(x))
        lid = (x >= LID[0]) & (x <= LID[1])
        values[lid] += 0.16 + 0.08 * np.sin((x[lid] - LID[0]) / (LID[1] - LID[0]) * np.pi) + rng.normal(0, 0.015, lid.sum())
        if k in SOME_RUNS:
            nmp = (x >= NMP[0]) & (x <= NMP[1])
            values[nmp] += 0.12 + 0.06 * np.sin((x[nmp] - NMP[0]) / (NMP[1] - NMP[0]) * np.pi) + rng.normal(0, 0.015, nmp.sum())
        values[:3] += [0.25, 0.15, 0.08]  # chain ends flop about
        values[-3:] += [0.08, 0.15, 0.25]
        runs.append(np.round(values, 4))
    return runs


def main() -> None:
    pdb_text = STRUCTURE.read_text()
    numbers = residue_numbers(pdb_text)
    files = []
    for k, values in enumerate(make_runs(numbers), start=1):
        lines = [
            "# Synthetic example made by Concordance (scripts/make_example.py), in the layout of gmx rmsf -res",
            "# The values are made up; the structure is adenylate kinase (PDB 4AKE).",
            '@    title "RMS fluctuation"',
            '@    xaxis  label "Residue"',
            '@    yaxis  label "(nm)"',
            "@TYPE xy",
        ]
        lines += [f"{n:5d}   {v:8.4f}" for n, v in zip(numbers, values)]
        files.append({"file": f"run{k}_rmsf.xvg", "text": "\n".join(lines) + "\n"})
    example = {
        "note": ("Synthetic test data with a designed answer, drawn on the real shape of adenylate kinase (PDB 4AKE): "
                 "residues 122-159 move a lot in every run, 30-59 in only three of five. The values are not from a simulation."),
        "name": "adenylate kinase (example)",
        "files": files,
        "structure": {"file": "adk_4ake_backbone.pdb", "text": pdb_text},
    }
    OUT.write_text(json.dumps(example))
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
