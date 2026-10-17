"""Synthetic replicate data with a known answer, laid out exactly like an mdCATH file.

Used by the test suite and by `concordance selftest`, so anyone can check that the tool
recovers an answer that was designed in, without downloading anything. Files are marked
`synthetic=True`, and every report and passport made from them says so.

Default design (residue numbers):
  21-23  flexible in every run
  9-11   flexible in runs 0, 1, 2 only
  1-7    helical in every run
  14-19  helical in runs 3 and 4 only
  runs 3 and 4 drift away from the starting structure (RMSD)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

ATOM_NAMES = ("N", "CA", "C", "O")
REPLICAS = ("0", "1", "2", "3", "4")


@dataclass(frozen=True)
class Design:
    n_residues: int = 30
    flex_all: tuple[int, int] = (21, 23)
    flex_some: tuple[int, int] = (9, 11)
    flex_some_runs: frozenset = field(default_factory=lambda: frozenset({"0", "1", "2"}))
    helix_all: tuple[int, int] = (1, 7)
    helix_some: tuple[int, int] = (14, 19)
    helix_some_runs: frozenset = field(default_factory=lambda: frozenset({"3", "4"}))
    drift_runs: frozenset = field(default_factory=lambda: frozenset({"3", "4"}))
    rigid_sigma: float = 0.15  # angstrom
    flexible_sigma: float = 1.2  # angstrom; RMSF about 8x the rigid residues, chain stays connected


def backbone_pdb(n_residues: int) -> str:
    """A simple ALA backbone along a gentle curve (N, CA, C, O per residue)."""
    i = np.arange(n_residues)
    ca = np.column_stack([3.8 * i, 4.0 * np.sin(i / 2.5), 4.0 * np.cos(i / 3.0)])
    offsets = np.array([[-1.2, 0.6, 0.0], [0.0, 0.0, 0.0], [1.2, -0.6, 0.0], [1.4, 0.6, 0.0]])
    lines, serial = [], 1
    for r in range(n_residues):
        for name, off in zip(ATOM_NAMES, offsets):
            x, y, z = ca[r] + off
            lines.append(
                f"ATOM  {serial:5d} {name:<4s} ALA A{r + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {name[0]:>2s}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def _parse(pdb_text: str):
    names, residues, coords, keys = [], [], [], []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            names.append(line[12:16].strip())
            residues.append(line[17:20].strip())
            keys.append((line[21:22].strip(), int(line[22:26]), line[26:27].strip()))
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return names, residues, keys, np.asarray(coords, dtype=float)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _in(number: int, span: tuple[int, int]) -> bool:
    return span[0] <= number <= span[1]


def write_synthetic_domain(
    path: str | Path,
    domain: str = "1abcA00",
    design: Design | None = None,
    pdb_text: str | None = None,
    n_frames: int = 80,
    temperatures: tuple[str, ...] = ("320", "348"),
    seed: int = 0,
) -> Path:
    """Write a synthetic file in mdCATH's documented layout. `pdb_text` lets you use a real protein's shape."""
    design = design or Design()
    pdb_text = pdb_text or backbone_pdb(design.n_residues)
    names, resnames, keys, base = _parse(pdb_text)
    residue_keys = list(dict.fromkeys(keys))
    numbers = np.array([k[1] for k in residue_keys])
    atom_residue = np.array([residue_keys.index(k) for k in keys])
    n_res = len(residue_keys)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        g = f.create_group(domain)
        g.attrs["numResidues"] = n_res
        g.attrs["synthetic"] = True
        g["pdbProteinAtoms"] = pdb_text
        g["pdb"] = pdb_text
        g["psf"] = "synthetic"
        g["resid"] = np.array([k[1] for k in keys])
        g["resname"] = np.array([r.encode() for r in resnames])
        g["chain"] = np.array([(k[0] or "A").encode() for k in keys])
        g["element"] = np.array([n[0].encode() for n in names])
        for t_index, temperature in enumerate(temperatures):
            tg = g.create_group(temperature)
            for replica in REPLICAS:
                rng = np.random.default_rng(seed + 100 * t_index + int(replica))
                sigma = np.full(n_res, design.rigid_sigma)
                flexible = np.array([_in(n, design.flex_all) for n in numbers])
                if replica in design.flex_some_runs:
                    flexible |= np.array([_in(n, design.flex_some) for n in numbers])
                sigma[flexible] = design.flexible_sigma
                coords = np.empty((n_frames, len(names), 3))
                for frame in range(n_frames):
                    shift = rng.normal(size=(n_res, 3)) * sigma[:, None]
                    moved = base + shift[atom_residue]
                    coords[frame] = moved @ _random_rotation(rng).T + rng.normal(scale=10.0, size=3)
                dssp = np.full((n_frames, n_res), "-", dtype="<U1")
                helix = np.array([_in(n, design.helix_all) for n in numbers])
                if replica in design.helix_some_runs:
                    helix |= np.array([_in(n, design.helix_some) for n in numbers])
                dssp[:, helix] = "H"
                dssp[rng.random(dssp.shape) < 0.05] = "T"
                t = np.linspace(0, 1, n_frames)
                rmsd = 0.15 + 0.02 * rng.normal(size=n_frames) + (0.6 * t if replica in design.drift_runs else 0.0)
                rg = tg.create_group(replica)
                rg.attrs["numFrames"] = n_frames
                rg["coords"] = coords.astype(np.float32)
                rg["dssp"] = dssp.astype("S1")
                rg["rmsf"] = (np.sqrt(3.0) * sigma / 10.0).astype(np.float32)
                rg["rmsd"] = rmsd.astype(np.float32)
                rg["gyrationRadius"] = np.full(n_frames, 1.1)
                rg["box"] = np.eye(3) * 8.0
    return path


def write_trajectory_files(h5_path: str | Path, out_dir: str | Path, temperature: str = "320", fmt: str = "xtc") -> tuple[Path, list[Path]]:
    """Export an mdCATH-layout file as a PDB topology plus one trajectory file per replicate."""
    import warnings

    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader
    from MDAnalysis.lib.util import NamedStream
    import io

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        domain = next(iter(f.keys()))
        pdb_text = f[domain]["pdbProteinAtoms"][()]
        pdb_text = pdb_text.decode() if isinstance(pdb_text, bytes) else str(pdb_text)
        topology = out_dir / f"{domain}.pdb"
        topology.write_text(pdb_text)
        trajectories = []
        group = f[domain][temperature]
        for replica in sorted(group.keys(), key=int):
            coords = np.asarray(group[replica]["coords"][()], dtype=np.float32)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                u = mda.Universe(NamedStream(io.StringIO(pdb_text), "topology.pdb"), coords, format=MemoryReader)
                target = out_dir / f"run{int(replica) + 1}.{fmt}"
                with mda.Writer(str(target), u.atoms.n_atoms) as writer:
                    for _ in u.trajectory:
                        writer.write(u.atoms)
            trajectories.append(target)
    return topology, trajectories
