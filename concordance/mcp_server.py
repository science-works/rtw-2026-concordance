"""Concordance as tools for AI agents (Model Context Protocol, stdio).

Register with Claude Code, from the repository folder:

    claude mcp add --transport stdio concordance -- uv run concordance-mcp

Every tool returns computed numbers together with an evidence passport id, so whatever
an agent says can be checked against the passport and replayed.
"""

from __future__ import annotations

import logging
import os
import shlex
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .analysis import Analysis, Method, analyse
from .mdcath import find_domain_files, open_domain
from .passport import build_passport, find_passport, write_passport
from .sources import ReplicateSet, from_files, from_mdcath, from_xvg
from .structure import write_structure_files

INSTRUCTIONS = (
    "Concordance checks whether a conclusion about protein flexibility holds up across replicate "
    "molecular dynamics runs. Report the verdict and the numbers exactly as returned, always quote the "
    "passport_id, and say that agreement between runs is not proof of correctness."
)

mcp = FastMCP("concordance", instructions=INSTRUCTIONS)


def _dirs() -> tuple[Path, Path]:
    return Path(os.environ.get("CONCORDANCE_DATA", "data")), Path(os.environ.get("CONCORDANCE_OUT", "outputs"))


def _load(protein: str | None, temperature: int, topology: str | None = None, trajectories: list[str] | None = None, unwrap: bool = False) -> ReplicateSet:
    data_dir, _ = _dirs()
    if topology:
        return from_files(topology, trajectories or [], unwrap=unwrap)
    files = find_domain_files(data_dir)
    if protein not in files:
        raise ValueError(f"Protein {protein} is not downloaded. Available: {', '.join(sorted(files)) or 'none'}.")
    return from_mdcath(files[protein], str(temperature))


def _replay(rset: ReplicateSet, question: list[str], extra: list[str]) -> str:
    if rset.kind == "xvg":
        source = ["--rmsf", *[str(t) for t in rset.trajectories]] + (["-s", str(rset.topology)] if rset.topology else [])
    elif rset.kind == "mdcath":
        source = ["--domain", rset.name, "--temp", str(rset.temperature)]
    else:
        source = ["-s", str(rset.topology), "-f", *[str(t) for t in rset.trajectories]] + (["--unwrap"] if rset.unwrap else [])
    m = Method()
    method = ["--rule", m.rule.rule, "--mad-k", f"{m.rule.k:g}", "--min-ratio", f"{m.rule.min_ratio:g}",
              "--trim-termini", str(m.rule.trim_termini), "--discard", f"{m.discard:g}", "--consistency", f"{m.consistency:g}"]
    return shlex.join(["concordance", *question, *source, *method, *extra])


def _finish(an: Analysis, question: dict, result: dict, replay: str) -> dict:
    data_dir, out_dir = _dirs()
    passport = build_passport(an, question=question, result=result, replay=replay, data_dir=data_dir)
    path = write_passport(passport, out_dir)
    return {"passport_id": passport["passport_id"], "passport_file": str(path), "replay": replay}


@mcp.tool()
def list_proteins(temperature: int = 320) -> dict:
    """List the mdCATH protein domains available locally, with their size and number of replicate runs."""
    data_dir, _ = _dirs()
    out = []
    for domain, path in find_domain_files(data_dir).items():
        info = open_domain(path)
        out.append({"protein": domain, "residues": info.n_residues, "first_residue": info.residues[0].number,
                    "last_residue": info.residues[-1].number, "runs": len(info.replicas.get(str(temperature), []))})
    return {"temperature_K": temperature, "proteins": out}


@mcp.tool()
def check_region_flexible(protein: str, start: int, end: int, temperature: int = 320) -> dict:
    """Is the region start-end of an mdCATH protein flexible? Answered separately in every replicate run, with a verdict
    (consistent across replicas / replica-dependent / not seen in most replicas) and how many runs alone would mislead."""
    an = analyse(_load(protein, temperature))
    claim = an.flexible_region(start, end)
    result = claim.to_dict()
    out = _finish(an, {"type": "flexible", "region": f"{start}-{end}", "text": claim.claim}, result,
                  _replay(an.rset, ["ask", "flexible"], ["--region", f"{start}-{end}", "--min-fraction", f"{an.method.min_fraction:g}"]))
    return {"answer": claim.plain_sentence(), "verdict": claim.verdict, "runs_saying_yes": claim.n_yes, "runs_total": claim.n_replicas,
            "per_run": {f"run {i}": a for i, a in enumerate(claim.answers.values(), start=1)}, "rule": claim.rule,
            "engine_cross_check_max_diff_nm": an.check["max_abs_diff_nm"], "chain_check_ok": an.geometry.get("ok", True), **out}


@mcp.tool()
def check_region_flexible_in_files(topology: str, trajectories: list[str], start: int, end: int, unwrap: bool = False) -> dict:
    """Same question for your own replicate runs: a topology file (.tpr, .gro, .pdb, ...) and one trajectory file per run.
    Set unwrap=True if the protein may be split across the periodic box (needs a topology with bonds, e.g. .tpr)."""
    an = analyse(_load(None, 0, topology, trajectories, unwrap))
    claim = an.flexible_region(start, end)
    out = _finish(an, {"type": "flexible", "region": f"{start}-{end}", "text": claim.claim}, claim.to_dict(),
                  _replay(an.rset, ["ask", "flexible"], ["--region", f"{start}-{end}", "--min-fraction", f"{an.method.min_fraction:g}"]))
    return {"answer": claim.plain_sentence(), "verdict": claim.verdict, "runs_saying_yes": claim.n_yes, "runs_total": claim.n_replicas,
            "chain_check": an.geometry, **out}


@mcp.tool()
def check_region_flexible_in_gmx_rmsf(xvg_files: list[str], start: int, end: int, structure: str | None = None) -> dict:
    """Same question using output you already have: one `gmx rmsf -res` .xvg file per replicate run.
    Nothing is re-simulated or recomputed; the runs are compared residue by residue. structure (optional): .pdb, .gro or .tpr."""
    an = analyse(from_xvg(xvg_files, structure))
    claim = an.flexible_region(start, end)
    out = _finish(an, {"type": "flexible", "region": f"{start}-{end}", "text": claim.claim}, claim.to_dict(),
                  _replay(an.rset, ["ask", "flexible"], ["--region", f"{start}-{end}", "--min-fraction", f"{an.method.min_fraction:g}"]))
    return {"answer": claim.plain_sentence(), "verdict": claim.verdict, "runs_saying_yes": claim.n_yes, "runs_total": claim.n_replicas,
            "note": "RMSF values were read from gmx rmsf output, not recomputed.", **out}


@mcp.tool()
def identify_files(paths: list[str]) -> dict:
    """What molecular-dynamics files are, judged by their contents (extensions can be wrong or missing):
    GROMACS, CHARMM/NAMD, AMBER, PDB ensembles, H5MD, mdCATH, per-residue tables. Folders are searched.
    Returns each file's format and role, trajectories paired with topologies and grouped by protein,
    and the Concordance commands to run next. Use this first when given unfamiliar files."""
    from .formats import groups, pair, sniff, suggest

    files = []
    for item in paths:
        item = Path(item).expanduser()
        files += sorted(p for p in item.rglob("*") if p.is_file() and not p.name.startswith(".")) if item.is_dir() else [item]
    infos = [sniff(p) for p in files]
    _, unmatched = pair(infos)
    return {"files": [i.as_dict() for i in infos],
            "runs_by_protein": [[{"topology": t.path, "trajectory": r.path if r is not None else None} for t, r in g] for g in groups(infos)],
            "unmatched_trajectories": [u.path for u in unmatched], "suggested_commands": suggest(infos)}


@mcp.tool()
def check_region_flexible_in_runs(runs: list[list[str]], start: int, end: int, unwrap: bool = False) -> dict:
    """Same flexibility question for runs that may come from different programs or formats. Each run is
    [topology, trajectory] or [ensemble_file]. Formats are read from contents; runs are matched by sequence."""
    from .sources import from_runs

    an = analyse(from_runs([(r[0], r[1] if len(r) > 1 else None) for r in runs], unwrap=unwrap))
    claim = an.flexible_region(start, end)
    replay = ["concordance", "ask", "flexible"]
    for r in runs:
        replay += ["--run", *r]
    replay += (["--unwrap"] if unwrap else []) + ["--region", f"{start}-{end}"]
    out = _finish(an, {"type": "flexible", "region": f"{start}-{end}", "text": claim.claim}, claim.to_dict(), shlex.join(replay))
    return {"answer": claim.plain_sentence(), "verdict": claim.verdict, "runs_saying_yes": claim.n_yes, "runs_total": claim.n_replicas,
            "how_runs_were_matched": an.rset.notes.get("matching", []), "sources": an.rset.notes.get("sources", []), **out}


@mcp.tool()
def reproduce_published_rmsf(runs: list[list[str]], published: str, unwrap: bool = False) -> dict:
    """Check a paper's published per-residue RMSF table against its own trajectories. Each run is [topology, trajectory].
    Works out the table's units, residue numbering and column order, tries four common recipes, and says whether the
    published numbers are reproduced (Pearson r and differences per run, under stated thresholds)."""
    from .reproduce import reproduce
    from .sources import from_runs

    rset = from_runs([(r[0], r[1] if len(r) > 1 else None) for r in runs], unwrap=unwrap)
    result = reproduce(rset, published)
    best = next(r for r in result["recipes"] if r["recipe"] == result["best_recipe"])
    return {"verdict": result["verdict"], "best_recipe": result["best_recipe"], "units": best["units"], "numbering": best["numbering"],
            "matching": best["matching"], "comparisons": best["comparisons"], "flexible_residues": best["flexible_residues"],
            "rule": result["rule"], "caveats": result["caveats"]}


@mcp.tool()
def check_dataset(path: str, name: str | None = None, reference: str | None = None) -> dict:
    """One call for a folder or a downloaded .zip of simulation files (data shared with a paper, a database entry such as
    ATLAS, or someone's own runs). Identifies every file by its contents, finds the replicate runs of the same protein,
    says which residues move a lot in all runs and which in only some (with a passport id and an HTML report), checks
    any published per-residue RMSF table in the folder against the trajectories, and charts the other tables.
    `reference` (e.g. "20-60") measures each run against a still region, for very disordered proteins."""
    import contextlib
    import io
    import json as _json
    import re as _re

    from .cli import main as cli_main

    _, out_dir = _dirs()
    argv = ["--out-dir", str(out_dir), "check", path]
    if name:
        argv += ["--name", name]
    if reference:
        argv += ["--reference", reference]
    log, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(err):  # stdout carries the MCP protocol: keep it clean
        code = cli_main(argv)
    text = log.getvalue()
    if code and "=== In short ===" not in text:
        raise ValueError((err.getvalue().strip() or text.strip()[-800:]) or "The check could not run on that path.")
    match = _re.search(r"=== 2\. Do the \d+ runs agree on which parts of (.+) move\? ===", text)
    found = name or (match.group(1) if match else None)
    checks = sorted((out_dir / "checks").glob("*_check.json"), key=lambda p: p.stat().st_mtime)
    if not checks:
        raise ValueError("The check ran but wrote no summary: " + text.strip()[-800:])
    result = _json.loads(checks[-1].read_text())
    result["summary_text"] = text.split("=== In short ===", 1)[-1].strip()
    result["note"] = ("Agreement between runs is not proof an answer is right. Quote numbers exactly as given here, and give the "
                      "passport_id so the answer can be replayed.")
    if found and result.get("name") != found:
        result["warning"] = "the newest summary on disk may not belong to this call"
    return result


@mcp.tool()
def find_flexible_residues(protein: str, temperature: int = 320) -> dict:
    """Which residues move a lot, and in how many of the replicate runs? Also reports what each run alone would have missed or added."""
    an = analyse(_load(protein, temperature))
    hot = an.hot
    classes = hot.classes()
    result = {"counts": hot.counts(), "runs_alone": an.run_summaries(),
              "per_residue_agreement": {an.rset.residues[i].label: int(a) for i, a in enumerate(hot.agreement)}}
    out = _finish(an, {"type": "hotspots", "region": None, "text": f"which residues of {protein} are flexible"}, result,
                  _replay(an.rset, ["ask", "hotspots"], []))
    pick = lambda name: [an.rset.residues[i].label for i, c in enumerate(classes) if c == name]
    return {"consistent_across_replicas": pick("consistent across replicas"), "replica_dependent": pick("replica-dependent"),
            "runs_alone": {f"run {i}": s for i, s in enumerate(an.run_summaries().values(), start=1)},
            "engine_cross_check_max_diff_nm": an.check["max_abs_diff_nm"], **out}


@mcp.tool()
def make_report(protein: str, temperature: int = 320, start: int | None = None, end: int | None = None, audience: str = "new") -> dict:
    """Write the one-file visual report (five runs in 3D, the consensus, the numbers and the receipt). audience: new, expert or check."""
    from .report import write_report

    an = analyse(_load(protein, temperature))
    claim = an.flexible_region(start, end) if start is not None and end is not None else None
    result = claim.to_dict() if claim else {"counts": an.hot.counts()}
    extra = ["--region", f"{start}-{end}", "--min-fraction", f"{an.method.min_fraction:g}"] if claim else []
    question = {"type": "flexible" if claim else "hotspots", "region": f"{start}-{end}" if claim else None,
                "text": claim.claim if claim else f"which residues of {protein} are flexible"}
    data_dir, out_dir = _dirs()
    replay = _replay(an.rset, ["report"], extra)
    passport = build_passport(an, question=question, result=result, replay=replay, data_dir=data_dir)
    write_passport(passport, out_dir)
    write_structure_files(an, out_dir)
    path = write_report(an, claim, passport, out_dir, audience)
    return {"report_file": str(path), "passport_id": passport["passport_id"], "replay": replay}


@mcp.tool()
def get_passport(passport_id: str) -> dict:
    """Return a stored evidence passport (data fingerprints, method, numbers, replay command) by its id."""
    _, out_dir = _dirs()
    passport = find_passport(passport_id, out_dir)
    if passport is None:
        raise ValueError(f"No passport {passport_id} in {out_dir / 'passports'}.")
    return passport


def main() -> None:
    logging.getLogger("MDAnalysis").setLevel(logging.WARNING)  # keep progress chatter out of agent logs
    mcp.run()


if __name__ == "__main__":
    main()
