"""Work out what a molecular-dynamics file is from its contents, not its name.

Papers and databases share simulations in many formats, often with unhelpful or missing file
extensions. `sniff()` reads the first bytes of a file and recognises the common formats by their
signatures (GROMACS XTC/TRR/TPR/GRO, CHARMM/NAMD DCD/PSF/CRD/binary coordinates, AMBER
prmtop/NetCDF/mdcrd, PDB and multi-model ensembles, mmCIF, H5MD, mdCATH HDF5, XYZ, LAMMPS) and
per-residue tables (gmx rmsf .xvg, cpptraj .dat, CSV). It returns the format code MDAnalysis
needs, so files can be read even when their extension is wrong. `pair()` matches trajectories
with topologies by atom count, and `suggest()` writes the Concordance command for what it found.
"""

from __future__ import annotations

import bz2
import gzip
import re
import struct
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

HEAD = 8192

# MDAnalysis format codes, what they are and which program writes them.
KINDS = {
    "XTC": ("trajectory", "GROMACS", "GROMACS compressed trajectory"),
    "TRR": ("trajectory", "GROMACS", "GROMACS full-precision trajectory"),
    "TPR": ("topology", "GROMACS", "GROMACS run input (topology with bonds)"),
    "GRO": ("structure", "GROMACS", "GROMACS structure"),
    "DCD": ("trajectory", "CHARMM / NAMD / OpenMM", "DCD trajectory"),
    "PSF": ("topology", "CHARMM / NAMD", "CHARMM/NAMD topology (PSF)"),
    "CRD": ("structure", "CHARMM", "CHARMM coordinate file"),
    "COOR": ("structure", "NAMD", "NAMD binary coordinates"),
    "PARM7": ("topology", "AMBER", "AMBER topology (prmtop)"),
    "NCDF": ("trajectory", "AMBER / OpenMM", "AMBER NetCDF trajectory"),
    "TRJ": ("trajectory", "AMBER", "AMBER ASCII trajectory (mdcrd)"),
    "INPCRD": ("structure", "AMBER", "AMBER coordinates (inpcrd/rst7)"),
    "PDB": ("structure", "any", "PDB structure"),
    "CIF": ("structure", "any", "mmCIF structure"),
    "H5MD": ("trajectory", "H5MD (GROMACS, OpenMM, ...)", "H5MD trajectory"),
    "XYZ": ("trajectory", "any", "XYZ coordinates"),
    "LAMMPSDUMP": ("trajectory", "LAMMPS", "LAMMPS dump trajectory"),
    "DATA": ("topology", "LAMMPS", "LAMMPS data file"),
}


@dataclass
class FileInfo:
    path: str
    kind: str  # trajectory | topology | structure | ensemble | rmsf-table | mdcath | unknown
    format: str | None  # MDAnalysis format code, or "XVG"/"MDCATH"
    engine: str
    description: str
    by: str  # "contents" or "extension"
    atoms: int | None = None
    frames: int | None = None
    compressed: str | None = None
    notes: list[str] = field(default_factory=list)
    sequence: str | None = None  # protein sequence (one letter per C-alpha), when a topology or structure says
    columns: list | None = None  # a table's value-column names

    def as_dict(self) -> dict:
        return asdict(self)


def _read_head(path: Path) -> tuple[bytes, str | None]:
    with open(path, "rb") as f:
        raw = f.read(HEAD)
    if raw[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as f:
            return f.read(HEAD), "gzip"
    if raw[:3] == b"BZh":
        with bz2.open(path, "rb") as f:
            return f.read(HEAD), "bzip2"
    return raw, None


def _is_text(head: bytes) -> bool:
    if not head:
        return False
    if b"\x00" in head[:1024]:
        return False
    printable = sum(32 <= c < 127 or c in (9, 10, 13) for c in head[:2048])
    return printable / min(len(head), 2048) > 0.95


def _floats_fixed(line: str, width: int) -> bool:
    line = line.rstrip("\n")
    if not line.strip() or len(line) % width:
        return False
    try:
        [float(line[i : i + width]) for i in range(0, len(line), width)]
    except ValueError:
        return False
    return True


def _hdf5_kind(path: Path) -> tuple[str | None, str]:
    try:
        import h5py

        with h5py.File(path, "r") as f:
            conventions = str(f.attrs.get("Conventions", b"")) if "Conventions" in f.attrs else ""
            if "AMBER" in conventions:
                return "NCDF", "AMBER NetCDF trajectory (NetCDF4/HDF5)"
            if "h5md" in f:
                return "H5MD", "H5MD trajectory"
            for key in f.keys():
                group = f[key]
                if isinstance(group, h5py.Group) and ("pdbProteinAtoms" in group or "pdbProteinAtoms" in group.attrs):
                    return "MDCATH", f"mdCATH dataset file (domain {key})"
    except Exception as exc:  # unreadable or not ours
        return None, f"HDF5 file that could not be read ({exc.__class__.__name__})"
    return None, "HDF5 file of an unknown layout"


def _count_models(path: Path, compressed: str | None) -> int:
    opener = gzip.open if compressed == "gzip" else bz2.open if compressed == "bzip2" else open
    n = 0
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("MODEL"):
                n += 1
    return n


def _by_contents(path: Path, head: bytes, compressed: str | None) -> tuple[str | None, str | None]:
    """(format code, extra description) from the file's first bytes; (None, None) if not recognised."""
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return _hdf5_kind(path)
    if head[:4] in (b"CDF\x01", b"CDF\x02"):
        return "NCDF", None
    if len(head) >= 4:
        magic = struct.unpack(">i", head[:4])[0]
        if magic == 1995:
            return "XTC", None
        if magic == 1993:
            return "TRR", None
    for endian in ("<", ">"):
        if len(head) >= 8 and struct.unpack(endian + "i", head[:4])[0] == 84 and head[4:8] in (b"CORD", b"VELD"):
            return "DCD", None
        if len(head) >= 12 and struct.unpack(endian + "q", head[:8])[0] == 84 and head[8:12] in (b"CORD", b"VELD"):
            return "DCD", None
    if b"VERSION" in head[:40] and not _is_text(head):
        return "TPR", None
    if not _is_text(head):
        if len(head) >= 4 and not compressed:
            natoms = struct.unpack("<i", head[:4])[0]
            if natoms > 0 and path.stat().st_size == 4 + 24 * natoms:
                return "COOR", None
        return None, None

    text = head.decode("latin-1")
    lines = text.splitlines()
    stripped = [l.strip() for l in lines if l.strip()]
    first = stripped[0] if stripped else ""
    if "%VERSION" in text[:200] and "%FLAG" in text:
        return "PARM7", None
    if first.startswith("PSF"):
        return "PSF", None
    if any(l.startswith("ITEM: TIMESTEP") for l in lines[:3]):
        return "LAMMPSDUMP", None
    if first.startswith("data_") and "_atom_site" in text:
        return "CIF", None
    if any(l.startswith(("ATOM  ", "HETATM")) for l in lines) or first.startswith(("HEADER", "CRYST1", "MODEL", "REMARK", "TITLE", "COMPND")):
        return "PDB", None
    if len(lines) > 3 and lines[1].strip().isdigit() and len(lines[2]) >= 44 and lines[2][10:15].strip() and _looks_number(lines[2][20:28]):
        return "GRO", None
    if first.startswith("*"):
        body = [l for l in lines if not l.startswith("*")]
        if body and body[0].split() and body[0].split()[0].isdigit():
            return "CRD", None
    if len(lines) > 2 and lines[0].strip().isdigit() and len(lines[2].split()) >= 4 and _looks_number(lines[2].split()[1]):
        return "XYZ", None
    if re.search(r"^\s*\d+\s+atoms\s*$", text, re.M) and "Atoms" in text:
        return "DATA", None
    if len(lines) > 2 and _floats_fixed(lines[1], 8) and _floats_fixed(lines[2], 8) and len(lines[1].rstrip("\n")) == 80:
        return "TRJ", None
    # tables of per-residue values (gmx rmsf .xvg, cpptraj .dat, CSV)
    if first.startswith(("@", "#")) or re.match(r"^[A-Za-z_\"']", first):
        if _table_ok(text):
            return "XVG", None
    if len(lines) > 2 and lines[1].split() and lines[1].split()[0].isdigit() and _floats_fixed(lines[2], 12):
        return "INPCRD", None
    if _table_ok(text):
        return "XVG", None
    return None, None


def _looks_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _table_ok(text: str) -> bool:
    from .xvg import parse_rmsf_text

    try:
        table = parse_rmsf_text(text.rsplit("\n", 1)[0] if len(text) >= HEAD else text)
    except ValueError:
        return False
    if len(table.numbers) < 3:
        return False
    values = [v for _, col in table.columns for v in col]
    return min(values) >= 0 and max(values) < 100


EXTENSIONS = {
    ".xtc": "XTC", ".trr": "TRR", ".tpr": "TPR", ".gro": "GRO", ".dcd": "DCD", ".psf": "PSF", ".crd": "CRD", ".cor": "CRD",
    ".coor": "COOR", ".prmtop": "PARM7", ".parm7": "PARM7", ".top": "PARM7", ".nc": "NCDF", ".ncdf": "NCDF", ".mdcrd": "TRJ",
    ".crdbox": "TRJ", ".inpcrd": "INPCRD", ".rst7": "INPCRD", ".pdb": "PDB", ".ent": "PDB", ".cif": "CIF", ".h5md": "H5MD",
    ".xyz": "XYZ", ".lammpstrj": "LAMMPSDUMP", ".data": "DATA", ".xvg": "XVG", ".dat": "XVG", ".csv": "XVG", ".tsv": "XVG",
    ".h5": "MDCATH",
}


def sniff(path: str | Path) -> FileInfo:
    """What a file is, judged by its contents (falling back to its extension)."""
    path = Path(path)
    if not path.exists():
        return FileInfo(str(path), "unknown", None, "", "File not found.", "contents")
    if path.is_dir():
        return FileInfo(str(path), "unknown", None, "", "A folder, not a file.", "contents")
    head, compressed = _read_head(path)
    fmt, extra = _by_contents(path, head, compressed)
    by = "contents"
    if fmt is None:
        suffixes = [s.lower() for s in path.suffixes if s.lower() not in (".gz", ".bz2")]
        fmt = EXTENSIONS.get(suffixes[-1]) if suffixes else None
        by = "extension"
    if fmt is None:
        return FileInfo(str(path), "unknown", None, "", extra or "Not a format Concordance recognises.", "contents", compressed=compressed)
    if fmt == "XVG":
        info = FileInfo(str(path), "rmsf-table", "XVG", "GROMACS / AMBER / spreadsheet", "Per-residue values (gmx rmsf .xvg, cpptraj .dat or a table)", by, compressed=compressed)
        try:
            from .xvg import read_rmsf_table

            table = read_rmsf_table(path)
            info.atoms = len(table.numbers)
            info.columns = [name for name, _ in table.columns]
            if not looks_like_rmsf(info):
                info.description = "A table of numbers (its name and columns don't say RMSF)"
            else:
                info.description = "Per-residue RMSF table (gmx rmsf .xvg, cpptraj .dat, CSV or TSV)"
            info.notes.append(f"{len(table.numbers)} {'residues' if looks_like_rmsf(info) else 'rows'}, {len(table.columns)} value column(s)" + (f", units {'nm' if table.unit == 'nm' else 'ångström'}" if table.unit else ""))
        except Exception:
            # a text table of some other shape (contacts, annotations...): not something Concordance compares
            info.kind, info.format, info.engine, info.by = "unknown", None, "", "contents"
            info.description = "A text table that isn't one value per residue (left out)"
        return info
    if fmt == "MDCATH":
        return FileInfo(str(path), "mdcath", "MDCATH", "mdCATH (GPUGRID / ACEMD)", extra or "mdCATH dataset file", by)
    kind, engine, description = KINDS[fmt]
    info = FileInfo(str(path), kind, fmt, engine, extra or description, by, compressed=compressed)
    if by == "extension":
        info.notes.append("recognised by its file name only; the contents didn't match a known signature")
    if fmt == "PDB":
        models = _count_models(path, compressed)
        if models > 1:
            info.kind = "ensemble"
            info.frames = models
            info.description = f"PDB ensemble ({models} models, e.g. an NMR bundle or a deposited ensemble)"
    _measure(info)
    return info


def _measure(info: FileInfo) -> None:
    """Atom (and for self-contained files, frame) counts where MDAnalysis can tell on its own."""
    if info.format == "COOR":
        info.atoms = int(struct.unpack("<i", open(info.path, "rb").read(4))[0])
        return
    if info.kind not in ("topology", "structure", "ensemble"):
        if info.format in ("XTC", "TRR", "DCD", "NCDF", "H5MD"):
            try:
                import MDAnalysis as mda

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    reader = mda.coordinates.core.get_reader_for(info.path, format=info.format)(info.path)
                    info.atoms, info.frames = int(reader.n_atoms), int(reader.n_frames)
                    reader.close()
            except Exception:
                pass
        return
    try:
        import MDAnalysis as mda

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = mda.Universe(info.path, format=info.format if info.format not in ("PARM7", "PSF", "TPR", "DATA") else None,
                             topology_format=info.format)
            info.atoms = int(len(u.atoms))
            if info.kind == "ensemble":
                info.frames = int(u.trajectory.n_frames)
            ca = u.select_atoms("protein and name CA")
            if len(ca):
                from .sources import one_letter

                info.sequence = "".join(one_letter(n) for n in ca.resnames)
                info.notes.append(f"{len(ca)} protein residues")
    except Exception as exc:
        info.notes.append(f"MDAnalysis could not open it on its own ({exc.__class__.__name__})")


def open_universe(topology: str | Path, trajectory: str | Path | None = None):
    """An MDAnalysis Universe, with formats taken from the files' contents (so wrong extensions don't matter)."""
    import MDAnalysis as mda

    top = sniff(topology)
    kwargs = {}
    if top.format and top.format not in ("XVG", "MDCATH"):
        kwargs["topology_format"] = top.format
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if trajectory is None:
            if top.format and top.kind in ("ensemble", "structure", "trajectory"):
                kwargs["format"] = top.format
            return mda.Universe(str(topology), **kwargs)
        traj = sniff(trajectory)
        if traj.format and traj.format not in ("XVG", "MDCATH"):
            kwargs["format"] = traj.format
        return mda.Universe(str(topology), str(trajectory), **kwargs)


def pair(infos: list[FileInfo]) -> tuple[list[tuple[FileInfo, FileInfo]], list[FileInfo]]:
    """Match each trajectory with a topology or structure that has the same number of atoms."""
    tops = [i for i in infos if i.kind in ("topology", "structure") and i.atoms]
    runs, unmatched = [], []
    for info in infos:
        if info.kind != "trajectory":
            continue
        if info.atoms is None and info.format in ("TRJ",):
            # ASCII AMBER trajectories don't record the atom count: try each AMBER topology
            for top in tops:
                if top.format == "PARM7":
                    runs.append((top, info))
                    break
            else:
                unmatched.append(info)
            continue
        match = [t for t in tops if t.atoms == info.atoms]
        # prefer a real topology (bonds, names) over a bare structure
        match.sort(key=lambda t: 0 if t.kind == "topology" else 1)
        if match:
            runs.append((match[0], info))
        else:
            unmatched.append(info)
    return runs, unmatched


def _short(path: str) -> str:
    import os

    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return rel if len(rel) < len(path) else path


def groups(infos: list[FileInfo]) -> list[list[tuple[FileInfo, FileInfo | None]]]:
    """Runs (topology + trajectory, or one ensemble) grouped by protein: same sequence, give or take a few residues."""
    from .sources import best_offset

    runs, _ = pair(infos)
    items = [(t, r) for t, r in runs] + [(e, None) for e in infos if e.kind == "ensemble"]
    out: list[list] = []
    for top, traj in items:
        seq = top.sequence or ""
        for group in out:
            ref = group[0][0].sequence or ""
            if seq and ref:
                offset, identity, overlap = best_offset(ref, seq)
                if identity >= 0.9 and overlap >= max(10, 0.5 * min(len(ref), len(seq))):
                    group.append((top, traj))
                    break
        else:
            out.append([(top, traj)])
    return out


def describe(info: FileInfo) -> str:
    bits = [info.description]
    if info.atoms:
        bits.append(f"{info.atoms} atoms" if info.kind != "rmsf-table" else "")
    if info.frames:
        bits.append(f"{info.frames} frames" if info.kind != "ensemble" else "")
    bits = [b for b in bits if b]
    text = ", ".join(bits)
    if info.compressed:
        text += f" ({info.compressed}-compressed)"
    return text


def looks_like_rmsf(info: FileInfo) -> bool:
    """Whether a table says it holds RMSF (or fluctuations), by its file name or column names."""
    words = [Path(info.path).name] + list(info.columns or [])
    return any(k in w.lower() for w in words for k in ("rmsf", "fluct"))


def suggest(infos: list[FileInfo]) -> list[str]:
    """Concordance commands for what was found."""
    import shlex

    out = []
    tables = [i for i in infos if i.kind == "rmsf-table"]
    if any(looks_like_rmsf(t) for t in tables):
        tables = [t for t in tables if looks_like_rmsf(t)]  # leave out B-factor, pLDDT, radius-of-gyration tables and the like
    mdcath = [i for i in infos if i.kind == "mdcath"]
    found = groups(infos)
    for group in found:
        if len(group) < 2:
            continue
        run_words = []
        for top, traj in group:
            run_words += ["--run", _short(top.path)] + ([_short(traj.path)] if traj is not None and traj.path != top.path else [])
        unwrap = ["--unwrap"] if any(t.format in ("TPR", "PSF", "PARM7") for t, _ in group) else []
        out.append(shlex.join(["uv", "run", "concordance", "report", *run_words, *unwrap, "--open"]))
        n = len((group[0][0].sequence or ""))
        for table in tables:
            if not n or table.atoms is None or abs(table.atoms - n) <= max(5, 0.1 * n):
                out.append(shlex.join(["uv", "run", "concordance", "reproduce", *run_words, *unwrap, "--published", _short(table.path)]))
    if found and all(len(g) < 2 for g in found):
        out.append("Each protein here has only one run: Concordance compares two or more runs of the same protein.")
    if tables and not found:
        out.append(shlex.join(["uv", "run", "concordance", "report", "--rmsf", *[_short(t.path) for t in tables], "--open"]))
    for m in mdcath:
        out.append(f"uv run concordance report --data-dir {shlex.quote(str(Path(m.path).parent))} --domain <id from 'concordance domains'> --open")
    return out
