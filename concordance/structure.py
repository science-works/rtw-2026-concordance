"""Write agreement into files that structure viewers already understand.

The PDB's B-factor column holds the number of runs that call each residue flexible
(-1 = not scored, e.g. chain ends). The PyMOL script colours the same five classes
the visual report uses.
"""

from __future__ import annotations

import math
from pathlib import Path

CLASS_COLOURS = {  # same hex values as the report
    "not scored": "#e3e7eb",
    "never flagged": "#c3cbd3",
    "minority": "#f2b48f",
    "replica-dependent": "#eb6834",
    "consistent": "#2a78d6",
}


def pdb_with_values(pdb_text: str, values: dict[tuple[str, int, str], float], default: float = -1.0) -> str:
    """Put one number per residue into the B-factor column."""
    lines = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            key = (line[21:22].strip(), int(line[22:26]), line[26:27].strip())
            value = values.get(key, default)
            padded = line.ljust(80)
            line = f"{padded[:60]}{value:6.2f}{padded[66:]}".rstrip()
        lines.append(line)
    return "\n".join(lines) + "\n"


def agreement_pdb(rset, agreement) -> str:
    values = {res.key: float(n) for res, n in zip(rset.residues, agreement)}
    return pdb_with_values(rset.pdb_text, values)


def class_bounds(n: int, consistency: float) -> dict[str, tuple[float, float]]:
    """B-factor ranges for each class, for n runs."""
    need = math.ceil(consistency * n - 1e-9)
    few = n - need
    bounds = {"not scored": (-1.5, -0.5), "never flagged": (-0.5, 0.5)}
    if few >= 1:
        bounds["minority"] = (0.5, few + 0.5)
    if need - 1 >= few + 1:
        bounds["replica-dependent"] = (few + 0.5, need - 0.5)
    bounds["consistent"] = (need - 0.5, n + 0.5)
    return bounds


def _rgb(hex_colour: str) -> str:
    h = hex_colour.lstrip("#")
    return "[" + ", ".join(f"{int(h[i:i + 2], 16) / 255:.3f}" for i in (0, 2, 4)) + "]"


def pymol_script(pdb_filename: str, n: int, consistency: float) -> str:
    lines = [
        "# Concordance: colour residues by how many replicate runs call them flexible.",
        "# The B-factor column holds that count (-1 = not scored).",
        f"load {pdb_filename}, concordance",
        "hide everything, concordance",
        "show cartoon, concordance",
        "bg_color white",
    ]
    for name, (low, high) in class_bounds(n, consistency).items():
        colour = "conc_" + name.replace(" ", "_").replace("-", "_")
        lines.append(f"set_color {colour}, {_rgb(CLASS_COLOURS[name])}")
        lines.append(f"color {colour}, concordance and b > {low} and b < {high}")
    lines += ["orient concordance", ""]
    return "\n".join(lines)


def write_structure_files(an, out_dir: str | Path) -> dict[str, Path]:
    folder = Path(out_dir) / "structures"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{an.rset.name}_{an.rset.condition.replace(' ', '')}_agreement"
    pdb_path = folder / f"{stem}.pdb"
    pdb_path.write_text(agreement_pdb(an.rset, an.hot.agreement))
    pml_path = folder / f"{stem}.pml"
    pml_path.write_text(pymol_script(pdb_path.name, an.n, an.method.consistency))
    return {"pdb": pdb_path, "pymol": pml_path}
