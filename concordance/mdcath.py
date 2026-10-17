"""Read mdCATH domain files.

Layout, as documented in Mirarchi et al., Scientific Data 11, 1299 (2024), Table 1:

    <domain_id>/                         root group (attribute: numResidues)
        pdbProteinAtoms                  PDB text for the N solute atoms used for analysis
        resid, resname, chain, element, z
        <temperature>/                   "320", "348", "379", "413", "450" (kelvin)
            <replica>/                   "0" ... "4"
                coords           F x N x 3   angstrom
                dssp             F x R       8-letter DSSP codes
                rmsf             R           nm (C-alpha)
                rmsd             F           nm
                gyrationRadius   F           nm
                (attribute: numFrames)

One frame is 1 ns. Replicas can have different numbers of frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

TEMPERATURES = ("320", "348", "379", "413", "450")
FILE_PREFIX = "mdcath_dataset_"


@dataclass(frozen=True)
class Residue:
    chain: str
    number: int
    name: str
    insertion: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.number, self.insertion)

    @property
    def label(self) -> str:
        return f"{self.name or 'res'}{self.number}{self.insertion}"


@dataclass
class DomainInfo:
    path: Path
    domain_id: str
    residues: list[Residue]
    ca_index: np.ndarray  # positions of C-alpha atoms among the N solute atoms
    n_atoms: int
    replicas: dict[str, list[str]]  # temperature -> replica ids
    pdb_text: str

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def residue_positions(self, start: int, end: int) -> np.ndarray:
        """Positions (0-based) of residues numbered start..end inclusive."""
        positions = [i for i, r in enumerate(self.residues) if start <= r.number <= end]
        if not positions:
            first, last = self.residues[0].number, self.residues[-1].number
            raise ValueError(
                f"No residues numbered {start}-{end} in {self.domain_id} "
                f"(this domain runs from {first} to {last})."
            )
        return np.asarray(positions, dtype=int)

    def replicas_at(self, temperature: str) -> list[str]:
        temperature = str(temperature)
        if temperature not in self.replicas:
            available = ", ".join(sorted(self.replicas, key=int))
            raise ValueError(
                f"Temperature {temperature} K is not in {self.domain_id} (available: {available})."
            )
        return self.replicas[temperature]


def _to_text(value) -> str:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError("Expected a single string value.")
        value = value.reshape(-1)[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_pdb_atoms(pdb_text: str) -> list[tuple[str, str, str, int, str]]:
    """Return (atom name, residue name, chain, residue number, insertion code) per atom."""
    atoms = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atoms.append(
                (
                    line[12:16].strip(),
                    line[17:20].strip(),
                    line[21:22].strip(),
                    int(line[22:26]),
                    line[26:27].strip(),
                )
            )
    return atoms


def domain_id_from_path(path: str | Path) -> str:
    stem = Path(path).stem
    return stem[len(FILE_PREFIX):] if stem.startswith(FILE_PREFIX) else stem


def open_domain(path: str | Path) -> DomainInfo:
    """Read a domain's metadata (structure, residues, available replicas) without loading trajectories."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        wanted = domain_id_from_path(path)
        domain_id = wanted if wanted in f else next(iter(f.keys()))
        group = f[domain_id]
        pdb_text = _to_text(group["pdbProteinAtoms"][()])
        atoms = parse_pdb_atoms(pdb_text)

        ca_index, residues = [], []
        for i, (name, resname, chain, number, insertion) in enumerate(atoms):
            if name == "CA":
                ca_index.append(i)
                residues.append(Residue(chain, number, resname, insertion))

        replicas: dict[str, list[str]] = {}
        for key in group.keys():
            if key.isdigit() and isinstance(group[key], h5py.Group):
                ids = sorted((r for r in group[key].keys() if r.isdigit()), key=int)
                if ids:
                    replicas[key] = ids

        if not replicas:
            raise ValueError(f"No temperature/replica groups found in {path.name}.")
        first_t = sorted(replicas, key=int)[0]
        n_coord_atoms = group[first_t][replicas[first_t][0]]["coords"].shape[1]
        if n_coord_atoms != len(atoms):
            raise ValueError(
                f"{path.name}: coordinates have {n_coord_atoms} atoms but pdbProteinAtoms has "
                f"{len(atoms)}; cannot match C-alpha atoms safely."
            )

    return DomainInfo(
        path=path,
        domain_id=domain_id,
        residues=residues,
        ca_index=np.asarray(ca_index, dtype=int),
        n_atoms=len(atoms),
        replicas=replicas,
        pdb_text=pdb_text,
    )


def _replica_group(f: h5py.File, info: DomainInfo, temperature: str, replica: str) -> h5py.Group:
    return f[info.domain_id][str(temperature)][str(replica)]


def read_ca_coords(info: DomainInfo, temperature: str) -> dict[str, np.ndarray]:
    """C-alpha coordinates (frames x residues x 3, angstrom) for every replica at one temperature."""
    out = {}
    with h5py.File(info.path, "r") as f:
        for replica in info.replicas_at(temperature):
            coords = _replica_group(f, info, temperature, replica)["coords"][()]
            out[replica] = np.asarray(coords[:, info.ca_index, :], dtype=np.float64)
    return out


def read_stored(info: DomainInfo, temperature: str, field: str) -> dict[str, np.ndarray]:
    """A stored per-replica array ('rmsf', 'rmsd' or 'gyrationRadius') for every replica."""
    out = {}
    with h5py.File(info.path, "r") as f:
        for replica in info.replicas_at(temperature):
            out[replica] = np.asarray(_replica_group(f, info, temperature, replica)[field][()], dtype=np.float64)
    return out


def _decode_codes(array: np.ndarray) -> np.ndarray:
    """Turn stored DSSP codes into a frames x residues array of one-character strings."""
    array = np.asarray(array)
    if array.dtype.kind in ("S", "O"):
        array = np.vectorize(lambda v: v.decode("ascii", "replace") if isinstance(v, (bytes, np.bytes_)) else str(v))(array)
    array = array.astype(str)
    if array.ndim == 1:  # one string per frame
        array = np.array([list(row) for row in array])
    return array


def read_dssp(info: DomainInfo, temperature: str) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(info.path, "r") as f:
        for replica in info.replicas_at(temperature):
            out[replica] = _decode_codes(_replica_group(f, info, temperature, replica)["dssp"][()])
    return out


def find_domain_files(data_dir: str | Path) -> dict[str, Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob(f"{FILE_PREFIX}*.h5")) if data_dir.exists() else []
    return {domain_id_from_path(p): p for p in files}
