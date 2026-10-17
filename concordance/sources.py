"""Where replicate trajectories come from.

Two kinds of input, one shape of output:

* an mdCATH domain file (HDF5) at one temperature; or
* ordinary trajectory files, one per replicate, in any format MDAnalysis reads
  (GROMACS .tpr/.gro + .xtc/.trr, AMBER, NAMD/CHARMM .dcd, ...).

Either way Concordance extracts the C-alpha coordinates of each replicate, and every
flexibility number is then computed by MDAnalysis's own analysis code.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from .mdcath import DomainInfo, Residue, open_domain, parse_pdb_atoms, read_dssp, read_stored

FILES_SELECT = "protein and name CA"  # 'protein' keeps calcium ions (also named CA) out
DISPLAY_ATOMS = {"N", "CA", "C", "O", "CB"}  # enough for a cartoon, small enough to embed


@dataclass
class ReplicateSet:
    """Replicate runs of the same system under the same condition."""

    name: str
    kind: str  # "mdcath" or "files"
    condition: str  # e.g. "320 K" or "trajectory files"
    residues: list[Residue]
    pdb_text: str
    replica_ids: list[str]
    data_files: list[Path]
    temperature: str | None = None
    info: DomainInfo | None = None
    topology: Path | None = None
    trajectories: list[Path] = field(default_factory=list)
    synthetic: bool = False
    unwrap: bool = False
    supplied_rmsf: dict | None = None  # RMSF (nm) read from files (gmx rmsf, cpptraj, CSV) instead of computed
    notes: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)  # kind "mixed": one dict per run (files, formats, residue slice)

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def residue_positions(self, start: int, end: int) -> np.ndarray:
        positions = [i for i, r in enumerate(self.residues) if start <= r.number <= end]
        if not positions:
            first, last = self.residues[0].number, self.residues[-1].number
            raise ValueError(f"No residues numbered {start}-{end} in {self.name} (it runs from {first} to {last}).")
        return np.asarray(positions, dtype=int)

    def frame_counts(self) -> dict[str, int]:
        if self.kind == "mdcath":
            with h5py.File(self.info.path, "r") as f:
                group = f[self.info.domain_id][self.temperature]
                return {r: int(group[r]["coords"].shape[0]) for r in self.replica_ids}
        if self.kind == "mixed":
            return {r: int(run["frames"]) for r, run in zip(self.replica_ids, self.runs)}
        return {r: int(self._universe(t).trajectory.n_frames) for r, t in zip(self.replica_ids, self.trajectories)}

    def ca_coords(self, start: int = 0, stop: int | None = None, windows: dict | None = None) -> dict[str, np.ndarray]:
        """C-alpha coordinates (frames x residues x 3, angstrom) per replicate, frames start:stop
        (or, for runs from different sources, each run's own window)."""
        out = {}
        if self.kind == "mixed":
            for replica, run in zip(self.replica_ids, self.runs):
                a, b = (windows or {}).get(replica, (start, stop))
                u = self._run_universe(run)
                lo, hi = run["slice"]
                ca = u.select_atoms(FILES_SELECT)[lo:hi]
                out[replica] = np.array([ca.positions.copy() for _ in u.trajectory[a:b]], dtype=np.float64)
            return out
        if self.kind == "mdcath":
            with h5py.File(self.info.path, "r") as f:
                group = f[self.info.domain_id][self.temperature]
                for replica in self.replica_ids:
                    coords = group[replica]["coords"][start:stop]
                    out[replica] = np.asarray(coords[:, self.info.ca_index, :], dtype=np.float64)
            return out
        for replica, trajectory in zip(self.replica_ids, self.trajectories):
            u = self._universe(trajectory)
            ca = u.select_atoms(FILES_SELECT)
            out[replica] = np.array([ca.positions.copy() for _ in u.trajectory[start:stop]], dtype=np.float64)
        return out

    def display_pdb(self) -> str:
        """Backbone (+C-beta) of the starting structure, for the 3D views."""
        lines = [
            line
            for line in self.pdb_text.splitlines()
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() in DISPLAY_ATOMS
        ]
        return "\n".join(lines + ["END"]) + "\n"

    def _run_universe(self, run: dict):
        from .formats import open_universe

        u = open_universe(run["topology"], run.get("trajectory"))
        box = u.trajectory.ts.dimensions
        if self.unwrap and (box is None or not np.any(box[:3])):
            return u  # no periodic box (vacuum or implicit solvent): nothing to unwrap
        if self.unwrap:
            from MDAnalysis import transformations

            if not hasattr(u.atoms, "bonds") or len(u.atoms.bonds) == 0:
                raise ValueError(f"--unwrap needs a topology with bonds; {Path(run['topology']).name} has none.")
            u.trajectory.add_transformations(transformations.unwrap(u.select_atoms("protein")))
        return u

    def _universe(self, trajectory: Path):
        import MDAnalysis as mda

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(str(self.topology), str(trajectory))
            if self.unwrap:
                from MDAnalysis import transformations

                if not hasattr(u.atoms, "bonds") or len(u.atoms.bonds) == 0:
                    raise ValueError("--unwrap needs a topology with bonds (.tpr, .psf, .prmtop). "
                                     "Otherwise make molecules whole first with: gmx trjconv -pbc mol")
                u.trajectory.add_transformations(transformations.unwrap(u.select_atoms("protein")))
            return u

    # -- values stored in mdCATH files ---------------------------------------------

    def require_mdcath(self, what: str) -> None:
        if self.kind != "mdcath":
            raise ValueError(f"{what} uses values stored in mdCATH files; it is not available for trajectory files yet.")

    def stored(self, field_name: str) -> dict[str, np.ndarray]:
        self.require_mdcath(f"The stored '{field_name}' values")
        return read_stored(self.info, self.temperature, field_name)

    def dssp(self) -> dict[str, np.ndarray]:
        self.require_mdcath("The 'folded' question")
        return read_dssp(self.info, self.temperature)


def from_mdcath(path: str | Path, temperature: str) -> ReplicateSet:
    info = open_domain(path)
    temperature = str(temperature)
    replicas = info.replicas_at(temperature)
    with h5py.File(info.path, "r") as f:
        synthetic = bool(f[info.domain_id].attrs.get("synthetic", False))
    return ReplicateSet(
        name=info.domain_id,
        kind="mdcath",
        condition=f"{temperature} K",
        residues=info.residues,
        pdb_text=info.pdb_text,
        replica_ids=list(replicas),
        data_files=[info.path],
        temperature=temperature,
        info=info,
        synthetic=synthetic,
    )


def _structure_residues(structure: Path) -> tuple[dict[int, Residue], str]:
    import MDAnalysis as mda

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(structure))
        handle, tmp = tempfile.mkstemp(suffix=".pdb")
        os.close(handle)
        try:
            u.select_atoms("protein").write(tmp)
            pdb_text = Path(tmp).read_text()
        finally:
            os.unlink(tmp)
    by_number: dict[int, Residue] = {}
    for name, resname, chain, number, insertion in parse_pdb_atoms(pdb_text):
        if name == "CA" and number not in by_number:
            by_number[number] = Residue(chain, number, resname, insertion)
    return by_number, pdb_text


def from_xvg(paths: list[str | Path], structure: str | Path | None = None, name: str | None = None, units: str | None = None) -> ReplicateSet:
    """Per-residue RMSF you already have: one file per run (gmx rmsf .xvg, cpptraj .dat, CSV, text),
    or a single table with one column per run. Optionally with a structure for 3D.

    Units: gmx rmsf writes nm and cpptraj writes angstrom, and both say so in their headers. A file
    that doesn't say is taken as nm unless `units` is given. Units only change the numbers shown:
    the flexible-or-not rule compares each value with its own run's median, so verdicts are the same.
    """
    from .xvg import read_rmsf_table

    paths = [Path(p) for p in paths]
    for path in paths:
        if not path.exists():
            raise ValueError(f"File not found: {path}")
    tables = [read_rmsf_table(p) for p in paths]
    if len(paths) == 1:
        table = tables[0]
        if len(table.columns) < 2:
            raise ValueError("Give at least two runs: one file per run, or one table with a column per run.")
        numbers = table.numbers
        runs = [(label, values) for label, values in table.columns]
        detected = {table.unit}
    else:
        for path, table in zip(paths, tables):
            if len(table.columns) != 1:
                raise ValueError(f"{path.name} has {len(table.columns)} value columns. Give one file per run, or one table with a column per run.")
        numbers = tables[0].numbers
        for path, table in zip(paths[1:], tables[1:]):
            if not np.array_equal(table.numbers, numbers):
                raise ValueError(f"{path.name} covers different residues from {paths[0].name}; every run must list the same residues.")
        runs = [(path.name, table.columns[0][1]) for path, table in zip(paths, tables)]
        detected = {t.unit for t in tables}
    if len(set(numbers.tolist())) != len(numbers):
        raise ValueError("Residue numbers repeat (several chains?). Give one chain at a time for now.")
    numbered = any(t.numbered for t in tables)
    known = {u for u in detected if u}
    if units is None:
        if len(known) > 1:
            raise ValueError("Some files are in nm and some in angstrom; convert them to the same unit first.")
        units, unit_source = (known.pop(), "from the file header") if known else ("nm", "assumed, as the files don't say")
    else:
        unit_source = "given with --units"
    scale = 0.1 if units == "A" else 1.0
    gaps = np.diff(numbers)
    notes = {
        "looks_like_atom_numbers": bool(len(gaps) and np.median(gaps) > 1.5),
        "units": units,
        "unit_source": unit_source,
        "columns": [label for label, _ in runs],
    }
    structure = Path(structure) if structure else None
    if structure is not None:
        if not structure.exists():
            raise ValueError(f"File not found: {structure}")
        by_number, pdb_text = _structure_residues(structure)
        if numbered and len(by_number) == len(numbers):
            # the files had no residue column: take the structure's numbering, in order
            numbers = np.asarray(list(by_number), dtype=int)
            notes["numbering"] = "the files have no residue column, so residues are numbered as in the structure, in order"
        elif numbered:
            notes["numbering"] = (f"the files have no residue column, so residues are numbered 1 to {len(numbers)}; "
                                  f"the structure has {len(by_number)} residues, so its numbering was not used")
        residues = [by_number.get(int(n), Residue("", int(n), "")) for n in numbers]
    else:
        pdb_text = ""
        if numbered:
            notes["numbering"] = f"the files have no residue column, so residues are numbered 1 to {len(numbers)}"
        residues = [Residue("", int(n), "") for n in numbers]
    skipped = sorted({name for t in tables for name in t.skipped})
    if skipped:
        notes["skipped_columns"] = skipped
    ids = [str(i) for i in range(1, len(runs) + 1)]
    all_xvg = all(p.suffix.lower() == ".xvg" for p in paths)
    return ReplicateSet(
        name=name or (structure.stem if structure else ("gmx_rmsf" if all_xvg else paths[0].stem)),
        kind="xvg",
        condition="gmx rmsf output" if all_xvg else "RMSF from files",
        residues=residues,
        pdb_text=pdb_text,
        replica_ids=ids,
        data_files=[*paths, *([structure] if structure else [])],
        topology=structure,
        trajectories=paths,  # for kind "xvg" these are the RMSF files
        supplied_rmsf={r: np.asarray(values, dtype=float) * scale for r, (_, values) in zip(ids, runs)},
        notes=notes,
    )


def _attr(atoms, name: str, i: int, default: str = "") -> str:
    try:
        value = getattr(atoms, name)[i]
    except Exception:  # MDAnalysis raises NoDataError when a format lacks the attribute
        return default
    return str(value or default).strip()


ONE_LETTER = {
    "ALA": "A", "VAL": "V", "LEU": "L", "ILE": "I", "MET": "M", "CYS": "C", "CYX": "C", "CYM": "C", "PHE": "F", "TRP": "W", "TYR": "Y",
    "SER": "S", "THR": "T", "ASN": "N", "GLN": "Q", "LYS": "K", "LYN": "K", "ARG": "R", "HIS": "H", "HSD": "H", "HSE": "H", "HSP": "H",
    "HID": "H", "HIE": "H", "HIP": "H", "ASP": "D", "ASH": "D", "GLU": "E", "GLH": "E", "GLY": "G", "PRO": "P", "MSE": "M",
}


def one_letter(name: str) -> str:
    """One-letter code for a residue name, across force-field naming (HSD, HIE, LYSH, HISA, CYS2, NMET, CGLY, ...)."""
    name = str(name).upper()
    if name in ONE_LETTER:
        return ONE_LETTER[name]
    if len(name) == 4 and name[0] in "NC" and name[1:] in ONE_LETTER:  # AMBER terminal residues
        return ONE_LETTER[name[1:]]
    if name[:3] in ONE_LETTER:  # GROMACS variants: LYSH, HISA, HISD, CYSH, CYS2, ASPH, GLUH
        return ONE_LETTER[name[:3]]
    return "X"


def sequence_of(residues) -> str:
    return "".join(one_letter(r.name) for r in residues)


def best_offset(ref: str, seq: str, min_overlap: int = 5) -> tuple[int, float, int]:
    """Shift of `seq` against `ref` with the most identical residues: (offset, identity, overlap).
    Residue i of seq lines up with residue i + offset of ref."""
    best = (0, 0.0, 0)
    for offset in range(-(len(seq) - min_overlap), len(ref) - min_overlap + 1):
        lo, hi = max(0, -offset), min(len(seq), len(ref) - offset)
        n = hi - lo
        if n < min_overlap:
            continue
        same = sum(seq[i] == ref[i + offset] for i in range(lo, hi))
        identity = same / n
        if (same, identity) > (best[1] * best[2], best[1]):
            best = (offset, identity, n)
    return best


def from_runs(runs: list[tuple[str | Path, str | Path | None]], name: str | None = None, unwrap: bool = False) -> ReplicateSet:
    """Replicates given run by run, each with its own files, possibly from different programs and formats
    (a GROMACS .tpr + .xtc, a CHARMM .psf + .dcd, an AMBER .prmtop + .nc, a multi-model PDB ensemble, ...).

    Formats are read from the files' contents. Runs are matched residue by residue by their protein
    sequence, so different numbering or a few missing terminal residues don't matter; only the residues
    every run has are compared, and the notes say exactly what was done.
    """
    from .formats import describe, open_universe, sniff

    if len(runs) < 2:
        raise ValueError("Give at least two runs (repeat --run for each one).")
    specs, sequences, labels = [], [], []
    for k, (top, traj) in enumerate(runs, start=1):
        top = Path(top)
        traj = Path(traj) if traj else None
        for path in [top, *([traj] if traj else [])]:
            if not path.exists():
                raise ValueError(f"File not found: {path}")
        top_info = sniff(top)
        traj_info = sniff(traj) if traj else None
        if traj is None and top_info.kind not in ("ensemble", "trajectory"):
            raise ValueError(f"Run {k}: {top.name} is {describe(top_info).lower() or 'not recognised'}; give a topology and a trajectory, or a multi-model ensemble.")
        try:
            u = open_universe(top, traj)
        except Exception as exc:
            raise ValueError(f"Run {k}: MDAnalysis could not read {top.name}{' + ' + traj.name if traj else ''} ({exc}).") from exc
        ca = u.select_atoms(FILES_SELECT)
        if len(ca) == 0:
            raise ValueError(f"Run {k}: no protein C-alpha atoms in {top.name}.")
        residues = [Residue(_attr(ca, "chainIDs", i) or _attr(ca, "segids", i)[:1], int(a.resid), str(a.resname), _attr(ca, "icodes", i)) for i, a in enumerate(ca)]
        u.trajectory[-1]
        last_frame = ca.positions.copy()
        sequences.append(sequence_of(residues))
        label = (traj_info or top_info)
        labels.append(f"{label.engine} {label.format}" if label.format else Path(label.path).suffix)
        specs.append({"topology": str(top), "trajectory": str(traj) if traj else None, "frames": int(u.trajectory.n_frames),
                      "ensemble": traj is None and top_info.kind == "ensemble", "residues": residues, "last_frame": last_frame,
                      "formats": [top_info.format, traj_info.format if traj_info else None],
                      "described": f"{Path((traj_info or top_info).path).name}: {describe(traj_info or top_info)} ({(traj_info or top_info).engine})"})
    # line every run up with run 1 by sequence, then keep the residues all runs share
    ref = sequences[0]
    offsets, notes = [0], []
    for k, seq in enumerate(sequences[1:], start=2):
        offset, identity, overlap = best_offset(ref, seq)
        if identity < 0.9 or overlap < max(10, 0.5 * min(len(ref), len(seq))):
            raise ValueError(f"Run {k} doesn't look like the same protein as run 1 (best match {identity:.0%} identical over {overlap} residues). "
                             "Concordance compares repeat runs of one protein.")
        if identity < 1.0:
            notes.append(f"run {k} differs from run 1 at {round((1 - identity) * overlap)} of {overlap} aligned residues (e.g. mutations or naming)")
        offsets.append(offset)
    lo = max(offsets)  # first reference index every run covers
    hi = min(off + len(seq) for off, seq in zip(offsets, sequences))
    if hi - lo < 10:
        raise ValueError("The runs share fewer than 10 residues.")
    for k, (spec, off) in enumerate(zip(specs, offsets), start=1):
        spec["slice"] = (lo - off, hi - off)
        first = spec["residues"][lo - off]
        ref_first = specs[0]["residues"][lo]
        if first.number != ref_first.number:
            notes.append(f"run {k} numbers its residues differently (its {first.number} is run 1's {ref_first.number}); matched by sequence")
    # the same simulation saved twice (say as .xtc and .trr) is not an independent repeat
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            a, b = specs[i], specs[j]
            if a["frames"] == b["frames"] and a["last_frame"].shape == b["last_frame"].shape and np.abs(a["last_frame"] - b["last_frame"]).max() < 0.02:
                notes.append(f"runs {i + 1} and {j + 1} have identical coordinates: they look like the same simulation saved twice, not independent repeats")
    dropped = [len(s) - (hi - lo) for s in sequences]
    if any(dropped):
        notes.append(f"compared on the {hi - lo} residues every run has" + "".join(f"; run {k} has {d} more" for k, d in enumerate(dropped, start=1) if d))
    residues = specs[0]["residues"][lo:hi]
    # structure for the 3D views: run 1's first frame, cut to the shared residues
    u = open_universe(specs[0]["topology"], specs[0]["trajectory"])
    u.trajectory[0]
    handle, tmp = tempfile.mkstemp(suffix=".pdb")
    os.close(handle)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u.select_atoms("protein").write(tmp)
        pdb_text = Path(tmp).read_text()
    finally:
        os.unlink(tmp)
    pdb_ca = [a for a in parse_pdb_atoms(pdb_text) if a[0] == "CA"]
    if len(pdb_ca) == len(specs[0]["residues"]):
        keep = {(c, n, ins) for _, _, c, n, ins in pdb_ca[lo:hi]}
        residues = [Residue(c, n, rn, ins) for _, rn, c, n, ins in pdb_ca[lo:hi]]
        pdb_text = "\n".join(l for l in pdb_text.splitlines() if not l.startswith(("ATOM", "HETATM")) or (l[21:22].strip(), int(l[22:26]), l[26:27].strip()) in keep) + "\n"
    for spec in specs:
        spec.pop("residues")
        spec.pop("last_frame")
    engines = sorted({tuple(spec["formats"]) for spec in specs})
    data_files = [Path(p) for spec in specs for p in (spec["topology"], spec["trajectory"]) if p]
    return ReplicateSet(
        name=name or Path(specs[0]["topology"]).stem,
        kind="mixed",
        condition="runs from " + (f"{len(engines)} kinds of file" if len(engines) > 1 else "trajectory files"),
        residues=residues,
        pdb_text=pdb_text,
        replica_ids=[str(i) for i in range(1, len(specs) + 1)],
        data_files=list(dict.fromkeys(data_files)),
        topology=Path(specs[0]["topology"]),
        trajectories=[Path(spec["trajectory"] or spec["topology"]) for spec in specs],
        unwrap=unwrap,
        runs=specs,
        notes={"matching": notes, "sources": [spec["described"] for spec in specs], "labels": labels},
    )


def from_files(topology: str | Path, trajectories: list[str | Path], name: str | None = None, unwrap: bool = False) -> ReplicateSet:
    """Replicates given as ordinary trajectory files sharing one topology."""
    import MDAnalysis as mda

    topology = Path(topology)
    trajectories = [Path(t) for t in trajectories]
    if len(trajectories) < 2:
        raise ValueError("Give at least two replicate trajectories (three or more is better).")
    for path in [topology, *trajectories]:
        if not path.exists():
            raise ValueError(f"File not found: {path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(str(topology), str(trajectories[0]))
        ca = u.select_atoms(FILES_SELECT)
        if len(ca) == 0:
            raise ValueError(f"No protein C-alpha atoms found in {topology.name}.")
        handle, tmp = tempfile.mkstemp(suffix=".pdb")
        os.close(handle)
        try:
            u.trajectory[0]
            u.select_atoms("protein").write(tmp)
            pdb_text = Path(tmp).read_text()
        finally:
            os.unlink(tmp)
        # Residue labels come from the PDB we just wrote, so the 3D views and the
        # agreement PDB use exactly the same chain ids and numbers.
        pdb_ca = [a for a in parse_pdb_atoms(pdb_text) if a[0] == "CA"]
        if len(pdb_ca) == len(ca):
            residues = [Residue(chain, number, resname, insertion) for _, resname, chain, number, insertion in pdb_ca]
        else:
            residues = [Residue(_attr(ca, "chainIDs", i), int(atom.resid), str(atom.resname), _attr(ca, "icodes", i)) for i, atom in enumerate(ca)]

    return ReplicateSet(
        name=name or topology.stem,
        kind="files",
        condition="trajectory files",
        residues=residues,
        pdb_text=pdb_text,
        replica_ids=[str(i) for i in range(1, len(trajectories) + 1)],
        data_files=[topology, *trajectories],
        topology=topology,
        trajectories=trajectories,
        unwrap=unwrap,
    )
