"""Per-residue flexibility (C-alpha RMSF), computed two independent ways.

Engine: MDAnalysis's own documented recipe (MDAnalysis User Guide, "Calculating the
root mean square fluctuation over a trajectory"):

    AverageStructure (frames fitted to frame 0)  ->  AlignTraj onto that average  ->  RMSF

Cross-check: a small NumPy Kabsch implementation of the same idea (fit to frame 0,
then refit to the mean structure). Two implementations agreeing is part of how
Concordance shows its numbers can be trusted; see `concordance validate`.

This is the quantity GROMACS users get from `gmx rmsf -res`, computed here for every
replicate at once.
"""

from __future__ import annotations

import warnings

import numpy as np


def mdanalysis_rmsf(universe, select: str = "name CA") -> np.ndarray:
    """RMSF in angstrom for the selected atoms, using MDAnalysis's documented recipe.

    The universe's trajectory is aligned in memory (it is modified in place).
    """
    from MDAnalysis.analysis import align, rms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        average = align.AverageStructure(universe, universe, select=select, ref_frame=0).run()
        align.AlignTraj(universe, average.results.universe, select=select, in_memory=True).run()
        atoms = universe.select_atoms(select)
        return np.asarray(rms.RMSF(atoms).run().results.rmsf, dtype=float)


def _kabsch_rotations(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Best-fit rotation for each frame. mobile: F x R x 3 (centred); reference: R x 3 (centred)."""
    h = np.einsum("fri,rj->fij", mobile, reference)
    u, _, vt = np.linalg.svd(h)
    v = np.transpose(vt, (0, 2, 1))
    ut = np.transpose(u, (0, 2, 1))
    d = np.sign(np.linalg.det(v @ ut))
    d[d == 0] = 1.0
    correction = np.zeros_like(h)
    correction[:, 0, 0] = 1.0
    correction[:, 1, 1] = 1.0
    correction[:, 2, 2] = d
    return v @ correction @ ut


def superimpose(coords: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Centre every frame and rotate it onto the (centred) reference."""
    centred = coords - coords.mean(axis=1, keepdims=True)
    ref = reference - reference.mean(axis=0)
    rotations = _kabsch_rotations(centred, ref)
    return np.einsum("fri,fji->frj", centred, rotations)


def ca_rmsf(coords: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Independent NumPy cross-check: RMSF per residue, in the units of `coords`.

    coords: frames x residues x 3. Frames are fitted to frame 0, then refitted to the
    average structure (`iterations` rounds in total), matching MDAnalysis's recipe.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3:
        raise ValueError("coords must have shape frames x residues x 3")
    if coords.shape[0] < 2:
        raise ValueError("Need at least two frames to measure fluctuation.")
    reference = coords[0]
    aligned = coords
    for _ in range(max(1, iterations)):
        aligned = superimpose(coords, reference)
        reference = aligned.mean(axis=0)
    deviations = aligned - reference
    return np.sqrt((deviations ** 2).sum(axis=2).mean(axis=0))
