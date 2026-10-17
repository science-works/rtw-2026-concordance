"""Write results as GROMACS-style .xvg files, so they drop into scripts and plots built for `gmx rmsf`.

`gmx rmsf -res` writes one trajectory's per-residue RMSF (nm) as an .xvg file: comment lines
starting with '#', Grace plotting directives starting with '@', then whitespace-separated columns.
Concordance writes the same layout with one column per replicate run, plus the mean, and a
second file with how many runs call each residue flexible (kept separate, so no file mixes units).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__


NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
SPLIT = re.compile(r"[\s,;]+")


@dataclass
class RmsfTable:
    """Per-residue values from a text table: residue numbers, then one or more value columns."""

    numbers: np.ndarray
    columns: list[tuple[str, np.ndarray]]
    unit: str | None  # "nm", "A" or None when the file does not say
    numbered: bool = False  # True when the file had no residue column and rows were numbered 1, 2, 3, ...
    skipped: list[str] = field(default_factory=list)  # summary columns left out (mean, SD, ...)


# Column names that hold residue numbers, and summary columns that are not runs. Same patterns in logic.js.
RESIDUE_NAME = re.compile(r"^#?\s*(res(idue)?([ _.-]?(number|num|no|nr|id|seq|i))?|resi|resid|resnum|resseq|seq|position|pos)$", re.I)
SUMMARY_NAME = re.compile(r"^(mean|average|avg|median|sd|std|stdev|std[ _.-]?dev|sem|se|stderr|err|error|min|max|range|runs?[ _-]?flagging|flagged|count|agreement|scored)([ _(].*)?$", re.I)
LEGEND = re.compile(r'^@\s*s(\d+)\s+legend\s+"(.*)"\s*$', re.I)


def detect_unit(header_text: str) -> str | None:
    """Units from a file's header: gmx rmsf writes "(nm)", cpptraj atomicfluct writes "AtomicFlx" (angstrom)."""
    text = header_text.lower()
    if "(nm)" in text or re.search(r"\bnm\b", text):
        return "nm"
    if "atomicflx" in text or "angstrom" in text or "(å)" in text or "(a)" in text or "å" in text:
        return "A"
    return None


def parse_rmsf_text(text: str, name: str = "file") -> RmsfTable:
    """Read gmx rmsf .xvg, cpptraj .dat, CSV/TSV or plain text with per-residue values.

    Lines starting with #, @ or & are comments (GROMACS legends name the columns). Columns may be
    separated by spaces, tabs, commas or semicolons. A first line of column names is allowed, and
    text columns (chain, residue name) are skipped. The residue numbers are the column named like
    "resid" or "residue_number", else the first numeric column if it holds whole numbers; a table
    of values only (as ATLAS writes) is numbered 1, 2, 3, ... in order. Summary columns named mean,
    SD, runs_flagging and so on are left out. The browser page (logic.js, parseRmsfText) follows
    exactly the same rules.
    """
    header_bits: list[str] = []
    legends: dict[int, str] = {}
    names: list[str] | None = None
    numeric: list[int] | None = None
    width = 0
    rows: list[list[float]] = []
    for n, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in "#@&":
            header_bits.append(stripped)
            legend = LEGEND.match(stripped)
            if legend:
                legends[int(legend.group(1))] = legend.group(2)
            continue
        tokens = [t.strip("\"'") for t in SPLIT.split(stripped) if t.strip("\"'")]
        if not tokens:
            continue
        if numeric is None and not any(NUMBER.match(t) for t in tokens):
            names = tokens
            header_bits.append(stripped)
            continue
        if numeric is None:
            numeric = [i for i, t in enumerate(tokens) if NUMBER.match(t)]
            width = len(tokens)
        if len(tokens) != width or not all(NUMBER.match(tokens[i]) for i in numeric):
            raise ValueError(f"{name}, line {n}: expected {width} columns like the first data row, found {stripped[:60]!r}")
        rows.append([float(tokens[i]) for i in numeric])
    if not rows:
        raise ValueError(f"{name}: no data rows found. Each row should hold one residue's values.")
    values = np.asarray(rows, dtype=float)
    aligned = names is not None and len(names) == width
    residue_col = None
    if aligned:
        for k, i in enumerate(numeric):
            if RESIDUE_NAME.match(names[i]):
                residue_col = k
                break
    if residue_col is None and len(numeric) >= 2 and np.allclose(values[:, 0], np.round(values[:, 0]), rtol=0, atol=1e-8):
        residue_col = 0
    numbered = residue_col is None
    numbers = np.arange(1, len(rows) + 1) if numbered else np.round(values[:, residue_col]).astype(int)
    columns, skipped = [], []
    for k, i in enumerate(numeric):
        if k == residue_col:
            continue
        if aligned:
            label = names[i]
        elif residue_col == 0 and (k - 1) in legends:
            label = legends[k - 1]
        else:
            label = f"column {len(columns) + len(skipped) + 1}"
        if SUMMARY_NAME.match(label):
            skipped.append(label)
            continue
        columns.append((label, values[:, k]))
    if not columns:
        raise ValueError(f"{name}: found residue numbers but no value columns.")
    return RmsfTable(numbers, columns, detect_unit(" ".join(header_bits)), numbered, skipped)


def read_rmsf_table(path: str | Path) -> RmsfTable:
    import bz2
    import gzip

    path = Path(path)
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    elif raw[:3] == b"BZh":
        raw = bz2.decompress(raw)
    return parse_rmsf_text(raw.decode("utf-8", errors="replace"), path.name)


def read_xvg(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Residue numbers and the first value column of a gmx rmsf .xvg (or any table this module reads)."""
    table = read_rmsf_table(path)
    return table.numbers, table.columns[0][1]


def _header(title: str, ylabel: str, legends: list[str], an) -> list[str]:
    lines = [
        f"# Written by Concordance {__version__} (replicate-aware RMSF, built on MDAnalysis)",
        (f"# System: {an.rset.name}; RMSF read from gmx rmsf output (frames chosen in GROMACS)" if an.rset.kind == "xvg"
         else f"# System: {an.rset.name}, {an.rset.condition}; frames {an.window.start}-{an.window.stop - 1} of each run"),
        f"# Flexible means: {an.method.rule.describe()}",
        f'@    title "{title}"',
        '@    xaxis  label "Residue"',
        f'@    yaxis  label "{ylabel}"',
        "@TYPE xy",
        "@ legend on",
    ]
    lines += [f'@ s{i} legend "{name}"' for i, name in enumerate(legends)]
    return lines


def write_xvg(an, folder: str | Path) -> dict[str, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{an.rset.name}_{an.rset.condition.replace(' ', '')}"
    runs = list(an.rmsf_nm)

    rmsf_path = folder / f"{stem}_rmsf.xvg"
    lines = _header("RMSF fluctuation, every replicate run", "(nm)", [f"run {i}" for i in range(1, len(runs) + 1)] + ["mean"], an)
    for i, res in enumerate(an.rset.residues):
        values = [an.rmsf_nm[r][i] for r in runs]
        lines.append(f"{res.number:5d} " + " ".join(f"{v:8.4f}" for v in values) + f" {sum(values) / len(values):8.4f}")
    rmsf_path.write_text("\n".join(lines) + "\n")

    agree_path = folder / f"{stem}_agreement.xvg"
    lines = _header("Runs that call each residue flexible", "runs (-1 = not scored)", ["runs flagging"], an)
    for i, res in enumerate(an.rset.residues):
        lines.append(f"{res.number:5d} {int(an.hot.agreement[i]):3d}")
    agree_path.write_text("\n".join(lines) + "\n")
    return {"rmsf": rmsf_path, "agreement": agree_path}
