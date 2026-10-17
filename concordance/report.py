"""A single-file, offline HTML report: the five runs, the consensus, the numbers and the receipt.

Everything (3D viewer, typeface, data) is embedded, so the file opens without a network
connection and can be emailed or attached to a paper. The page adapts to who is reading
it: people new to simulations, people who run simulations, and people checking the work.
"""

from __future__ import annotations

import base64
import html
import json
import math
import shlex
from pathlib import Path

from . import __version__
from .analysis import disorder_hint
from .verdict import contested_stretches

ASSETS = Path(__file__).parent / "assets"
AUDIENCES = ("new", "expert", "check")

MDA_CODE = """import MDAnalysis as mda
from MDAnalysis.analysis import align, rms

u = mda.Universe("topology", "run.xtc")   # one replicate
average = align.AverageStructure(u, u, select="protein and name CA", ref_frame=0).run()
align.AlignTraj(u, average.results.universe, select="protein and name CA", in_memory=True).run()
rmsf = rms.RMSF(u.select_atoms("protein and name CA")).run().results.rmsf   # angstrom

# Concordance runs this for every replicate on the same frame window,
# then scores each residue run by run and counts how many runs agree."""


def _data_and_method_words(replay: str) -> list[str]:
    """The data and method flags of a replay command, without the command, the question or the region."""
    words = shlex.split(replay)[1:]
    if words and words[0] == "report":
        words = words[1:]
    elif words and words[0] == "ask":
        words = words[2:]
    kept, skip = [], False
    for w in words:
        if skip:
            skip = False
            continue
        if w in ("--region", "--min-fraction"):
            skip = True
            continue
        kept.append(w)
    return kept


def region_command(replay: str, min_fraction: float) -> str:
    """The `concordance ask flexible` command for any region, keeping this report's data and method flags."""
    kept = _data_and_method_words(replay)
    return shlex.join(["concordance", "ask", "flexible", *kept]) + f" --region {{region}} --min-fraction {min_fraction:g}"


def hotspots_command(replay: str) -> str:
    """The `concordance ask hotspots` command for the same data and method."""
    return shlex.join(["concordance", "ask", "hotspots", *_data_and_method_words(replay)])


def stretches(mask, residues) -> list[list[int]]:
    """Runs of neighbouring residues (same chain, consecutive numbers) where mask is true, as 0-based [first, last]."""
    out: list[list[int]] = []
    for i, flag in enumerate(mask):
        if not flag:
            continue
        joins = (
            out
            and out[-1][1] == i - 1
            and residues[i].chain == residues[i - 1].chain
            and residues[i].number == residues[i - 1].number + 1
        )
        if joins:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def run_source(rset, k: int, replica: str) -> str:
    """Where a run's numbers came from, in words."""
    if rset.kind == "mdcath":
        return f"mdCATH replica {replica}"
    if rset.kind == "xvg":
        columns = rset.notes.get("columns") or []
        label = columns[k] if k < len(columns) else ""
        files = rset.trajectories or []
        if len(files) == 1:
            return f"{Path(files[0]).name}, column {label}"
        return Path(files[k]).name if k < len(files) else label
    files = rset.trajectories or []
    return Path(files[k]).name if k < len(files) else f"run {k + 1}"


def class_codes(agreement, n: int, consistency: float) -> list[str]:
    need = math.ceil(consistency * n - 1e-9)
    few = n - need
    codes = []
    for a in agreement:
        a = int(a)
        if a < 0:
            codes.append("u")
        elif a == 0:
            codes.append("n")
        elif a <= few:
            codes.append("m")
        elif a < need:
            codes.append("d")
        else:
            codes.append("c")
    return codes


def report_data(an, claim, passport: dict, audience: str = "new") -> dict:
    rset = an.rset
    n = an.n
    need = math.ceil(an.method.consistency * n - 1e-9)
    runs = [{"id": r, "label": f"Run {i}", "source": run_source(rset, k, r)} for k, (i, r) in enumerate(enumerate(an.rmsf_nm, start=1))]
    codes = class_codes(an.hot.agreement, n, an.method.consistency)
    data = {
        "version": __version__,
        "title": f"Concordance: {rset.name}, {rset.condition}",
        "name": rset.name,
        "kind": rset.kind,
        "condition": rset.condition,
        "synthetic": rset.synthetic,
        "audience": audience if audience in AUDIENCES else "new",
        "n": n,
        "need": need,
        "runs": runs,
        "residues": [{"l": r.label, "n": r.number, "c": r.chain} for r in rset.residues],
        "rmsf": {r: [round(float(v), 4) for v in values] for r, values in an.rmsf_nm.items()},
        "flags": {r: [int(bool(x)) for x in flags] for r, flags in an.hot.flags.items()},
        "agreement": [int(a) for a in an.hot.agreement],
        "classes": codes,
        "counts": an.hot.counts(),
        "stretches": {
            "consistent": stretches([c == "c" for c in codes], rset.residues),
            "some": stretches([c == "d" for c in codes], rset.residues),
            "few": stretches([c == "m" for c in codes], rset.residues),
            "contested": [[int(a), int(b)] for a, b in contested_stretches(an.hot.agreement, n, threshold=an.method.consistency)],
        },
        "summaries": an.run_summaries(),
        "spearman": an.hot.pairwise_spearman,
        "window": an.window.as_dict(),
        "method": an.method.as_dict(),
        "rule_text": an.method.rule.describe() + an.method.describe_reference(),
        "check": {k: v for k, v in an.check.items() if k != "per_run"},
        "geometry": an.geometry,
        "claim": None,
        "passport": passport,
        "created": passport["created_utc"],
        "pdb": rset.display_pdb(),
        "has_structure": any(line.startswith(("ATOM", "HETATM")) for line in rset.pdb_text.splitlines()),
        "mda_code": MDA_CODE,
        "region_command": region_command(passport["replay"], an.method.min_fraction),
        "hotspots_command": hotspots_command(passport["replay"]),
        "disorder": disorder_hint(an.rmsf_nm, an.core, [r.number for r in rset.residues], bool(an.method.reference)),
    }
    if rset.kind == "mixed":
        data["mixed"] = {
            "sources": rset.notes.get("sources", []),
            "frames_used": [an.window.per_run[r][1] - an.window.per_run[r][0] for r in rset.replica_ids],
            "matching": rset.notes.get("matching", []),
        }
    if rset.kind == "xvg":
        data["units"] = {"unit": rset.notes.get("units", "nm"), "source": rset.notes.get("unit_source", "file header")}
        data["numbering"] = rset.notes.get("numbering")
        data["skipped_columns"] = rset.notes.get("skipped_columns", [])
    if claim is not None:
        result = claim.to_dict()
        positions = result.get("region_positions") or []
        numbers = [rset.residues[p].number for p in positions]
        result["region"] = [min(numbers), max(numbers)] if numbers else None
        data["claim"] = result
    return data


def _b64(name: str) -> str:
    return base64.b64encode((ASSETS / name).read_bytes()).decode("ascii")


def _fill(data_json: str, example_json: str, title: str) -> str:
    viewer_js = (ASSETS / "3Dmol-min.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    logic_js = (ASSETS / "logic.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    template = (ASSETS / "report.html").read_text(encoding="utf-8")
    replacements = {
        "__TITLE__": html.escape(title),
        "__VERSION__": __version__,
        "__FONT400__": _b64("atkinson-hyperlegible-latin-400-normal.woff2"),
        "__FONT700__": _b64("atkinson-hyperlegible-latin-700-normal.woff2"),
        "__CONCORDANCE_DATA__": data_json,
        "__CONCORDANCE_EXAMPLE__": example_json,
        "__CONCORDANCE_LOGIC__": logic_js,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    # inserted last, so nothing inside the third-party library can be mistaken for a placeholder
    return template.replace("__THREEDMOL__", viewer_js, 1)


def render_report(an, claim, passport: dict, audience: str = "new") -> str:
    data = report_data(an, claim, passport, audience)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return _fill(payload, "null", data["title"])


def render_page() -> str:
    """The no-install page: people add their own RMSF files and the report is built in their browser."""
    example = (ASSETS / "example.json").read_text(encoding="utf-8").replace("</", "<\\/")
    return _fill("null", example, "Do your simulation runs agree?")


def write_page(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_page(), encoding="utf-8")
    return path


def write_report(an, claim, passport: dict, out_dir: str | Path, audience: str = "new") -> Path:
    folder = Path(out_dir) / "reports"
    folder.mkdir(parents=True, exist_ok=True)
    region = ""
    if claim is not None and claim.extra.get("region_positions"):
        nums = [an.rset.residues[p].number for p in claim.extra["region_positions"]]
        region = f"_{min(nums)}-{max(nums)}"
    path = folder / f"{an.rset.name}_{an.rset.condition.replace(' ', '')}{region}.html"
    path.write_text(render_report(an, claim, passport, audience), encoding="utf-8")
    return path
