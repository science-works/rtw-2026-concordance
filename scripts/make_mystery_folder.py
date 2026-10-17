"""Make a folder of real simulation files with unhelpful names, to try `concordance identify` and `reproduce`.

The files are MDAnalysis's own test data (adenylate kinase from CHARMM and GROMACS, an AMBER
system, an NMR ensemble), copied under misleading names, as data shared with a paper often arrives.
It also writes `published_rmsf.tsv`: a STAND-IN for a paper's published table, made here from the
two CHARMM runs in angstrom, with residue numbers offset by 100 and its columns in the opposite
order, so `reproduce` has something to line up. It is not from any paper.

    uv run --with MDAnalysisTests python scripts/make_mystery_folder.py
    uv run concordance identify outputs/mystery
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np


def main(out: str = "outputs/mystery") -> None:
    try:
        import MDAnalysisTests.datafiles as D
    except ImportError:
        sys.exit("Needs MDAnalysis's test files: run with  uv run --with MDAnalysisTests python scripts/make_mystery_folder.py")
    import MDAnalysis as mda

    from concordance.rmsf import superimpose

    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    names = {D.PSF: "topology.txt", D.DCD: "run_b", D.DCD2: "run_c.bin", D.TPR: "system.bin", D.XTC: "run_a.dat",
             D.PRMncdf: "params", D.NCDF: "traj_c.out", D.PDB_multiframe: "ensemble.txt"}
    for src, dst in names.items():
        shutil.copy(src, folder / dst)

    cols = []
    for dcd in (D.DCD2, D.DCD):  # deliberately the other way round
        u = mda.Universe(D.PSF, dcd)
        ca = u.select_atoms("protein and name CA")
        x = np.array([ca.positions.copy() for _ in u.trajectory])
        aligned = superimpose(x, x[0])
        cols.append(np.sqrt(((aligned - aligned.mean(0)) ** 2).sum(2).mean(0)))  # angstrom
    rows = "".join(f"{i + 101}\t{a:.3f}\t{b:.3f}\n" for i, (a, b) in enumerate(zip(*cols)))
    (folder / "published_rmsf.tsv").write_text("residue\tRMSF_R1\tRMSF_R2\n" + rows)
    print(f"Wrote {len(names) + 1} files to {folder}/ (published_rmsf.tsv is a made-up stand-in, not from a paper).")
    print(f"Next:  uv run concordance identify {folder}")


if __name__ == "__main__":
    main(*sys.argv[1:])
