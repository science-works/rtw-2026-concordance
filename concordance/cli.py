"""Command line interface: `concordance <command>`.

Your own GROMACS-style files work with the flags GROMACS users already know:
`-s` for the topology and `-f` for the trajectories (one per replicate run).

    concordance selftest                                   check the tool on data with a known answer
    concordance domains                                    mdCATH proteins you have downloaded
    concordance ask hotspots --domain 1abcA00              which residues move, and do the runs agree?
    concordance ask flexible --domain 1abcA00 --region 40-55
    concordance ask flexible -s topol.tpr -f run1.xtc run2.xtc run3.xtc --region 40-55
    concordance report --domain 1abcA00 --region 40-55     one-file visual report (opens offline)
    concordance rmsf --domain 1abcA00                      every run's RMSF as a table
    concordance validate                                   cross-checks on every downloaded protein
    concordance scan                                       agreement summary over every downloaded protein
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import Analysis, Method, analyse, validate
from .mdcath import find_domain_files, open_domain
from .passport import build_passport, load_manifest, write_passport
from .sources import ReplicateSet, from_files, from_mdcath, from_runs, from_xvg
from .structure import write_structure_files
from .verdict import CONSISTENT, DEPENDENT, NOT_SEEN, FlexRule, contested_stretches

REGION_PATTERN = re.compile(r"^\s*(-?\d+)\s*(?:-|:|\.\.)\s*(-?\d+)\s*$")
AUDIENCES = ("new", "expert", "check")


def parse_region(text: str) -> tuple[int, int]:
    match = REGION_PATTERN.match(text or "")
    if not match:
        raise ValueError(f"A region should look like 40-55, not {text!r}.")
    a, b = int(match.group(1)), int(match.group(2))
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------- shared arguments


def add_source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("where the runs come from (an mdCATH domain, or your own files)")
    g.add_argument("--domain", "-d", help="mdCATH domain id, e.g. 1abcA00 (see `concordance domains`)")
    g.add_argument("--temp", default="320", help="mdCATH temperature in kelvin (default 320)")
    g.add_argument("-s", "--topology", help="topology/structure file for your own runs (.tpr, .gro, .pdb, .psf, ...)")
    g.add_argument("-f", "--traj", nargs="+", help="one trajectory file per replicate run (.xtc, .trr, .dcd, ...)")
    g.add_argument("--name", help="a short name for your system (default: the topology file name)")
    g.add_argument("--unwrap", action="store_true", help="make the protein whole in every frame first (needs bonds, e.g. a .tpr)")
    g.add_argument("--run", action="append", nargs="+", metavar="FILE",
                   help="one run: TOPOLOGY TRAJECTORY (or a single multi-model ensemble). Repeat for each run. Runs may come from "
                        "different programs and formats; formats are read from the files' contents")
    g.add_argument("--xvg", "--rmsf", dest="xvg", nargs="+", metavar="FILE",
                   help="RMSF you already have: gmx rmsf -res .xvg, cpptraj .dat, CSV or text, one file per run "
                        "(or one table with a column per run); optionally -s STRUCTURE for 3D")
    g.add_argument("--units", choices=["nm", "A"], help="units of --rmsf files that don't say (gmx rmsf: nm; cpptraj: A)")


def add_method_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("method (every choice is written into the passport)")
    g.add_argument("--rule", choices=["mad", "topk"], default="mad", help="how a residue is called flexible within a run")
    g.add_argument("--mad-k", type=float, default=3.0, help="mad rule: median absolute deviations above the run's median")
    g.add_argument("--min-ratio", type=float, default=1.5, help="mad rule: and at least this many times the run's median")
    g.add_argument("--top-fraction", type=float, default=0.15, help="topk rule: share of residues called flexible per run")
    g.add_argument("--trim-termini", type=int, default=3, help="residues at each chain end left unscored")
    g.add_argument("--discard", type=float, default=0.1, help="share of each run skipped as equilibration")
    g.add_argument("--consistency", type=float, default=0.8, help="share of runs needed to call a finding consistent")
    g.add_argument("--min-fraction", type=float, default=0.5, help="share of a region's residues that must be flexible")
    g.add_argument("--reference", help="measure each run's typical movement over these residues only, e.g. 1-60 "
                   "(useful when a long disordered tail would otherwise set the baseline)")


def method_from(args) -> Method:
    rule = FlexRule(rule=args.rule, k=args.mad_k, min_ratio=args.min_ratio, top=args.top_fraction, trim_termini=args.trim_termini)
    reference = parse_region(args.reference) if getattr(args, "reference", None) else None
    return Method(rule=rule, discard=args.discard, consistency=args.consistency, min_fraction=args.min_fraction, reference=reference)


def method_flags(args) -> list[str]:
    flags = ["--rule", args.rule]
    if args.rule == "mad":
        flags += ["--mad-k", f"{args.mad_k:g}", "--min-ratio", f"{args.min_ratio:g}"]
    else:
        flags += ["--top-fraction", f"{args.top_fraction:g}"]
    flags += ["--trim-termini", str(args.trim_termini), "--discard", f"{args.discard:g}", "--consistency", f"{args.consistency:g}"]
    if getattr(args, "reference", None):
        a, b = parse_region(args.reference)
        flags += ["--reference", f"{a}-{b}"]
    return flags


def source_flags(args, rset: ReplicateSet) -> list[str]:
    if rset.kind == "mixed":
        flags = []
        for run in rset.runs:
            flags += ["--run", run["topology"]] + ([run["trajectory"]] if run["trajectory"] else [])
        return flags + (["--name", args.name] if args.name else []) + (["--unwrap"] if args.unwrap else [])
    if rset.kind == "xvg":
        flags = ["--rmsf", *[str(p) for p in args.xvg]] + (["-s", str(args.topology)] if args.topology else [])
        return flags + (["--name", args.name] if args.name else []) + (["--units", args.units] if getattr(args, "units", None) else [])
    if rset.kind == "mdcath":
        return ["--domain", rset.name, "--temp", str(rset.temperature)]
    flags = ["-s", str(args.topology), "-f", *[str(t) for t in args.traj]]
    return flags + (["--name", args.name] if args.name else []) + (["--unwrap"] if args.unwrap else [])


def load_source(args) -> ReplicateSet:
    if getattr(args, "run", None):
        if args.traj or args.topology or args.domain or getattr(args, "xvg", None):
            raise ValueError("Use --run for every run, without -s/-f, --rmsf or --domain.")
        bad = [r for r in args.run if len(r) not in (1, 2)]
        if bad:
            raise ValueError("Each --run takes a topology and a trajectory, or one ensemble file: e.g. --run topol.tpr run1.xtc")
        return from_runs([(r[0], r[1] if len(r) == 2 else None) for r in args.run], args.name, unwrap=args.unwrap)
    if getattr(args, "xvg", None):
        if args.traj:
            raise ValueError("Use either --rmsf (RMSF files you already have) or -f (trajectories), not both.")
        return from_xvg(args.xvg, args.topology, args.name, getattr(args, "units", None))
    if args.topology or args.traj:
        if not (args.topology and args.traj):
            raise ValueError("For your own runs give both -s TOPOLOGY and -f RUN1 RUN2 ... (one file per run).")
        return from_files(args.topology, args.traj, args.name, unwrap=args.unwrap)
    if not args.domain:
        raise ValueError("Say which runs to use: --domain ID for an mdCATH protein, or -s TOPOLOGY -f RUN1 RUN2 ...")
    files = find_domain_files(args.data_dir)
    if args.domain not in files:
        known = ", ".join(sorted(files)) or "none yet (run scripts/download_demo_data.py)"
        raise ValueError(f"Domain {args.domain} is not in {args.data_dir}. Downloaded: {known}.")
    return from_mdcath(files[args.domain], args.temp)


# ---------------------------------------------------------------- answering


def answer(args, an: Analysis):
    """Run the question; return (claim or None, result dict, question dict, extra replay flags)."""
    q = args.question
    extra: list[str] = []
    if q == "hotspots":
        hot = an.hot
        result = {
            "per_residue_agreement": {an.rset.residues[i].label: int(a) for i, a in enumerate(hot.agreement)},
            "counts": hot.counts(),
            "runs_alone": an.run_summaries(),
            "pairwise_spearman": hot.pairwise_spearman,
            "contested_stretches": [
                f"{an.rset.residues[a].number}-{an.rset.residues[b].number}"
                for a, b in contested_stretches(hot.agreement, hot.n_replicas, threshold=an.method.consistency)
            ],
            "replicas_total": hot.n_replicas,
        }
        text = f"which residues of {an.rset.name} are flexible ({an.rset.condition})"
        return None, result, {"type": q, "region": None, "text": text}, extra
    if q in ("flexible", "folded"):
        if not args.region:
            raise ValueError("This question needs a region, for example --region 40-55.")
        start, end = parse_region(args.region)
        extra += ["--region", f"{start}-{end}"]
        if q == "flexible":
            claim = an.flexible_region(start, end)
            extra += ["--min-fraction", f"{an.method.min_fraction:g}"]
        else:
            claim = an.folded_region(start, end, args.kind)
            extra += ["--kind", args.kind]
    else:
        claim = an.stays_near_start(args.max_rmsd)
        extra += ["--max-rmsd", f"{args.max_rmsd:g}"]
    result = claim.to_dict()
    return claim, result, {"type": q, "region": args.region, "text": claim.claim}, extra


def residue_list(an: Analysis, indices, limit: int = 30) -> str:
    labels = [an.rset.residues[i].label for i in indices]
    if not labels:
        return "none"
    return " ".join(labels[:limit]) + (f" (+{len(labels) - limit} more)" if len(labels) > limit else "")


def print_answer(args, an: Analysis, claim, result: dict, passport: dict, paths: dict) -> None:
    audience = args.audience
    n = an.n
    need = math.ceil(an.method.consistency * n - 1e-9)
    if an.rset.kind == "xvg":
        notes = an.rset.notes
        print(f"{an.rset.name}: RMSF for {n} runs read from your files (frames, fitting and atoms were chosen when they were made). "
              f"Units: {'angstrom' if notes.get('units') == 'A' else 'nm'}, {notes.get('unit_source', 'file header')}.")
        if notes.get("numbering"):
            print(f"Note: {notes['numbering']}.")
        if notes.get("skipped_columns"):
            print(f"Note: summary columns left out (not runs): {', '.join(notes['skipped_columns'])}.")
    elif an.rset.kind == "mixed":
        print(f"{an.rset.name}: {n} runs from different files, each read by its contents and matched residue by residue.")
        for k, (replica, source) in enumerate(zip(an.rset.replica_ids, an.rset.notes.get("sources", [])), start=1):
            a, b = an.window.per_run[replica]
            print(f"  Run {k}: {source}; frames {a}-{b - 1} used ({b - a})")
        for note in an.rset.notes.get("matching", []):
            print(f"  Note: {note}")
    else:
        print(f"{an.rset.name}, {an.rset.condition}, {n} runs, {an.window.used} frames per run analysed.")
    if an.rset.synthetic:
        print("Synthetic test data: the answer was designed in to test the tool.")
    from .analysis import disorder_hint

    hint = disorder_hint(an.rmsf_nm, an.core, [r.number for r in an.rset.residues], bool(an.method.reference))
    if hint:
        msg = (f"NOTE: {hint['share']:.0%} of residues move far more than the stillest ones (a long disordered region?). "
               "Each run's threshold is relative to its typical residue, so a mostly disordered chain can hide real flexible regions.")
        if hint["reference"]:
            msg += f" Try measuring against the still part: --reference {hint['reference'][0]}-{hint['reference'][1]}"
        print(msg, file=sys.stderr)
    warn_geometry(an)
    print()
    if claim is None:
        classes = an.hot.classes()
        consistent = [i for i, c in enumerate(classes) if c == CONSISTENT]
        dependent = [i for i, c in enumerate(classes) if c == DEPENDENT]
        minority = [i for i, (a, c) in enumerate(zip(an.hot.agreement, classes)) if a > 0 and c == NOT_SEEN]
        print(f"Move a lot in at least {need} of {n} runs: {residue_list(an, consistent)}")
        print(f"Move a lot in only some runs:          {residue_list(an, dependent)}")
        if audience != "new":
            if n - need >= 1:
                print(f"Move a lot in only {n - need} of {n} run(s):     {residue_list(an, minority)}")
            stretches = result["contested_stretches"]
            print(f"Contested stretches (3+ residues in a row): {', '.join(stretches) or 'none'}")
        print()
        for i, (replica, sm) in enumerate(an.run_summaries().items(), start=1):
            line = (f"  Run {i}: {sm['flexible_residues']} flexible; alone it would miss {sm['missed_consistent']} of the consistent ones "
                    f"and flag {sm['extra_not_consistent']} that fewer than {need} runs agree on")
            if audience != "new":
                line += f" (threshold {sm['threshold_nm']:.3f} nm, median {sm['median_rmsf_nm']:.3f} nm)"
            print(line)
    else:
        if audience == "new":
            word = {CONSISTENT: "Consistent across runs", DEPENDENT: "Replica-dependent", NOT_SEEN: "Not seen in most runs"}[claim.verdict]
            print(f"{word}: {claim.n_yes} of {n} runs say yes.")
            print(claim.plain_sentence())
            print("  " + "   ".join(f"Run {i}: {'yes' if a else 'no'}" for i, a in enumerate(claim.answers.values(), start=1)))
        else:
            print(claim.summary())
    if audience != "new":
        print()
        print(f"Rule: {an.method.rule.describe()}{an.method.describe_reference()}")
        check = an.check
        if check["agree"] is None:
            print("Cross-check: not applicable (RMSF was read from your files, not computed here).")
        else:
            print(
            f"Cross-check: MDAnalysis and an independent NumPy implementation "
            f"{'agree' if check['agree'] else 'DIFFER'} (largest difference {check['max_abs_diff_nm']:.1e} nm)."
        )
    print()
    print(f"Evidence passport {passport['passport_id']}: {paths['passport']}")
    if audience == "check":
        print(f"Replay: {passport['replay']}")
        for item in passport["source"]["files"]:
            print(f"  {item['file']}  sha256 {item['sha256']}")
    if paths.get("pdb"):
        print(f"Structure coloured by agreement: {paths['pdb']} (PyMOL: {paths['pymol']})")
    if paths.get("report"):
        print(f"Visual report: {paths['report']}")


def warn_geometry(an: Analysis) -> None:
    if an.rset.notes.get("looks_like_atom_numbers"):
        print("WARNING: the first column of the .xvg files looks like atom numbers, not residue numbers. "
              "Run gmx rmsf with -res so each row is one residue.", file=sys.stderr)
    g = an.geometry
    if g and not g.get("ok", True):
        w = g["where"]
        print(f"WARNING: neighbouring residues {w['between']} are {g['largest_A']:.0f} angstrom apart (run {w['run']}, frame {w['frame']}); "
              f"normal is about 3.8. {g['advice']}", file=sys.stderr)


def run_question(args, *, make_report: bool) -> tuple[Analysis, object, dict, dict]:
    rset = load_source(args)
    an = analyse(rset, method_from(args))
    claim, result, question, extra = answer(args, an)
    command = "report" if make_report else "ask"
    replay = ["concordance", command] + ([args.question] if command == "ask" else []) + source_flags(args, rset) + method_flags(args) + extra
    passport = build_passport(an, question=question, result=result, replay=shlex.join(replay), data_dir=args.data_dir)
    paths = {"passport": write_passport(passport, args.out_dir)}
    if args.question in ("hotspots", "flexible"):
        paths.update(write_structure_files(an, args.out_dir))
    if make_report:
        from .report import write_report

        paths["report"] = write_report(an, claim, passport, args.out_dir, args.audience)
    return an, claim, passport, paths


def cmd_ask(args) -> int:
    if args.report and args.question not in ("hotspots", "flexible"):
        raise ValueError("The visual report covers the 'hotspots' and 'flexible' questions.")
    an, claim, passport, paths = run_question(args, make_report=args.report)
    if args.json:
        print(json.dumps(passport, indent=2, default=str))
    else:
        print_answer(args, an, claim, passport["result"], passport, paths)
    return 0


def cmd_report(args) -> int:
    args.question = "flexible" if args.region else "hotspots"
    an, claim, passport, paths = run_question(args, make_report=True)
    print_answer(args, an, claim, passport["result"], passport, paths)
    if args.open:
        import webbrowser

        webbrowser.open(Path(paths["report"]).resolve().as_uri())
    return 0


# ---------------------------------------------------------------- tables and checks


def _collect_paths(items, out_dir) -> list[Path]:
    """Files from files, folders and downloaded .zip archives (unpacked under OUT_DIR/unzipped). Missing paths are explained."""
    import zipfile

    paths, missing = [], []
    for item in items:
        item = Path(item).expanduser()
        if not item.exists():
            missing.append(item)
            continue
        if item.is_file() and zipfile.is_zipfile(item):
            # a downloaded archive: unpack it next to the outputs and look inside
            target = Path(out_dir) / "unzipped" / item.stem
            with zipfile.ZipFile(item) as z:
                for member in z.namelist():
                    if member.startswith("/") or ".." in Path(member).parts:
                        raise ValueError(f"{item.name} contains an unsafe path ({member}); unzip it yourself and point at the folder.")
                z.extractall(target)
            print(f"Unzipped {item.name} into {target}")
            item = target
        if item.is_dir():
            paths += sorted(p for p in item.rglob("*") if p.is_file() and not p.name.startswith(".") and "__MACOSX" not in p.parts)
        else:
            paths.append(item)
    for m in missing:
        print(f"Can't find {m}.", file=sys.stderr)
        parent = m.parent if m.parent.exists() else None
        if parent is not None:
            near = sorted(p.name for p in parent.iterdir() if m.name.split("_")[0].lower() in p.name.lower())[:8]
            if near:
                print(f"  In {parent} there is: " + ", ".join(near), file=sys.stderr)
        print("  Check the name (Terminal is case-sensitive, and a downloaded .zip can be given as it is).", file=sys.stderr)
    return paths


def _print_inventory(infos) -> None:
    from .formats import describe, groups, pair

    width = min(max(len(Path(i.path).name) for i in infos), 40)
    for info in infos:
        name = Path(info.path).name
        how = "" if info.by == "contents" else "  [by file name only]"
        what = describe(info) if info.kind != "unknown" else info.description
        engine = f" ({info.engine})" if info.engine and info.engine != "any" and info.kind not in ("unknown",) else ""
        print(f"  {name:<{width}}  {what}{engine}{how}")
        for note in info.notes:
            if not note.startswith("MDAnalysis could not open it on its own"):
                print(f"  {'':<{width}}  {note}")
    _, unmatched = pair(infos)
    found = groups(infos)
    print()
    if found:
        print("Runs found, grouped by protein:")
        for g, group in enumerate(found, start=1):
            seq = group[0][0].sequence or ""
            print(f"  Protein {g}" + (f" ({len(seq)} residues, starts {seq[:12]}...)" if seq else "") + f": {len(group)} run(s)")
            for top, traj in group:
                if traj is None:
                    print(f"    - {Path(top.path).name} (an ensemble: each model counts as a frame)")
                else:
                    print(f"    - {Path(top.path).name} + {Path(traj.path).name}")
    for u in unmatched:
        print(f"  No topology with {u.atoms or 'a matching number of'} atoms for {Path(u.path).name}; add its .tpr/.psf/.prmtop/.pdb.")


def cmd_identify(args) -> int:
    """What each file is, judged by its contents; which files make runs; what to run next."""
    from .formats import pair, sniff, suggest

    paths = _collect_paths(args.files, args.out_dir)
    if not paths:
        return 1
    infos = [sniff(p) for p in paths]
    if args.json:
        runs, unmatched = pair(infos)
        print(json.dumps({"files": [i.as_dict() for i in infos], "runs": [[t.path, r.path] for t, r in runs],
                          "unmatched_trajectories": [u.path for u in unmatched], "suggested": suggest(infos)}, indent=2))
        return 0
    _print_inventory(infos)
    commands = suggest(infos)
    if commands:
        print()
        print("Next:")
        if any(c.startswith("uv run concordance report --run") for c in commands):
            print(f"  {shlex.join(['uv', 'run', 'concordance', 'check', *args.files, '--open'])}     (everything below, in one go)")
        for c in commands:
            print(f"  {c}")
    return 0


def _safe_name(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text)


def cmd_check(args) -> int:
    """One command for a folder or zip of simulation files: what's here, do the runs agree, do published numbers follow."""
    from .formats import groups, looks_like_rmsf, sniff

    paths = _collect_paths(args.files, args.out_dir)
    if not paths:
        return 1
    infos = [sniff(p) for p in paths]
    print("\n=== 1. What's here (read from each file's contents) ===")
    _print_inventory(infos)
    found = [g for g in groups(infos) if len(g) >= 2]
    tables = [i for i in infos if i.kind == "rmsf-table"]
    base = ["--out-dir", str(args.out_dir)]
    outcome = {}
    if not found:
        if tables and any(looks_like_rmsf(t) for t in tables):
            print("\n=== 2. Do the runs agree? (from the RMSF tables here; no trajectories found) ===")
            rmsf_tables = [t.path for t in tables if looks_like_rmsf(t)]
            return main([*base, "report", "--rmsf", *rmsf_tables, *(["--open"] if args.open else [])])
        print("\nNo set of two or more runs of the same protein here, so there is nothing to compare.")
        return 1
    group = max(found, key=len)
    run_words = []
    for top, traj in group:
        run_words += ["--run", top.path] + ([traj.path] if traj is not None and traj.path != top.path else [])
    unwrap = ["--unwrap"] if any(t.format in ("TPR", "PSF", "PARM7") for t, _ in group) else []
    name = args.name or Path(group[0][0].path).stem
    method = (["--reference", args.reference] if args.reference else [])
    print(f"\n=== 2. Do the {len(group)} runs agree on which parts of {name} move? ===")
    outcome["report"] = main([*base, "report", "--name", name, *run_words, *unwrap, *method, *(["--open"] if args.open else [])])
    n = len(group[0][0].sequence or "")
    published = [t for t in tables if looks_like_rmsf(t) and (not n or t.atoms is None or abs(t.atoms - n) <= max(5, 0.1 * n))]
    if published:
        for t in published:
            print(f"\n=== 3. Do the published numbers in {Path(t.path).name} follow from these trajectories? ===")
            outcome[f"reproduce {Path(t.path).name}"] = main([*base, "reproduce", "--name", name, *run_words, *unwrap, *method, "--published", t.path])
    else:
        print("\n=== 3. Published numbers ===\nNo per-residue RMSF table here to check against; give one with: concordance reproduce ... --published TABLE")
    series = [t for t in tables if not looks_like_rmsf(t) and t.atoms and n and abs(t.atoms - n) > max(5, 0.1 * n)]
    if series and not args.no_plots:
        print("\n=== Also: the other tables here, as charts (in place of xmgrace) ===")
        for t in series:
            main([*base, "plot", t.path])
    failed = [k for k, v in outcome.items() if v]
    import glob as _glob

    out = Path(args.out_dir)
    summary = {
        "name": name,
        "files": [{"file": Path(i.path).name, "kind": i.kind, "format": i.format, "what": i.description} for i in infos],
        "runs": [[top.path, traj.path if traj is not None else None] for top, traj in group],
        "steps_not_finished": failed,
        "agreement": None,
        "published": [],
        "charts": [],
    }
    print("\n=== In short ===")
    try:
        passports = sorted(Path(p) for p in _glob.glob(str(out / "passports" / f"{_glob.escape(name)}_*.json")))
        passports.sort(key=lambda p: p.stat().st_mtime)
        if passports and not outcome.get("report"):
            passport = json.loads(passports[-1].read_text())
            per = passport["result"]["per_residue_agreement"]
            need = math.ceil(0.8 * len(group) - 1e-9)
            consistent = [r for r, a in per.items() if a >= need]
            some = [r for r, a in per.items() if 0 < a < need]
            reports = sorted((out / "reports").glob("*.html"), key=lambda p: p.stat().st_mtime)
            summary["agreement"] = {"runs": len(group), "need": need, "consistent_residues": consistent, "only_some_runs": some,
                                    "passport_id": passport["passport_id"], "report": str(reports[-1].resolve()) if reports else None}
            print(f"  {len(group)} runs of {name}: {len(consistent)} residues move a lot in at least {need} of {len(group)} runs; "
                  f"{len(some)} move a lot in only some runs. (Passport {passport['passport_id']}.)")
        for t in published:
            rpath = out / "reproductions" / f"{name.replace(' ', '_')}_reproduction.json"
            if rpath.exists() and not outcome.get(f"reproduce {Path(t.path).name}"):
                r = json.loads(rpath.read_text())
                best = next(x for x in r["recipes"] if x["recipe"] == r["best_recipe"])
                rs = [c["pearson"] for c in best["comparisons"] if c["pearson"] is not None]
                diffs = [c["max_abs_diff_nm"] for c in best["comparisons"]]
                summary["published"].append({"table": Path(t.path).name, "verdict": r["verdict"], "best_recipe": r["best_recipe"],
                                             "pearson_min": min(rs), "pearson_max": max(rs), "max_abs_diff_nm": max(diffs),
                                             "flexible_residues": best["flexible_residues"], "units": best["units"], "file": str(rpath.resolve())})
                print(f"  Published RMSF ({Path(t.path).name}): {r['verdict']}, best with '{r['best_recipe']}' "
                      f"(Pearson r {min(rs):.3f} to {max(rs):.3f}; largest difference {max(diffs):.3f} nm).")
        summary["charts"] = [str((out / "plots" / (_safe_name(Path(t.path).stem) + ".html")).resolve()) for t in series] if series and not args.no_plots else []
    except Exception as exc:  # the summary is a convenience; the steps above already printed everything
        print(f"  (couldn't summarise: {exc})")
    folder = out / "checks"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / (_safe_name(name) + "_check.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print("  Done." + (f" These steps didn't finish: {', '.join(failed)}." if failed else " Every step finished."))
    return 1 if failed else 0


def cmd_plot(args) -> int:
    """Any .xvg / .dat / .csv / .tsv as an offline chart, in place of xmgrace."""
    from .plot import write

    files = []
    for item in args.files:
        item = Path(item).expanduser()
        if not item.exists():
            raise ValueError(f"Can't find {item}.")
        files += sorted(p for p in item.iterdir() if p.suffix.lower() in (".xvg", ".dat", ".csv", ".tsv")) if item.is_dir() else [item]
    if not files:
        raise ValueError("No .xvg, .dat, .csv or .tsv files found.")
    stem = _safe_name(args.title or (files[0].stem if len(files) == 1 else f"{files[0].stem}_and_{len(files) - 1}_more"))
    out = Path(args.out) if args.out else Path(args.out_dir) / "plots" / f"{stem}.html"
    path, data = write(files, out, args.title)
    print(f"{data['title']}: {len(data['series'])} series from {len(files)} file(s)" + (", treated as runs of the same thing: the range across them is shaded" if data["same_x"] and data["runs_like"] else ""))
    for row in data["summary"]:
        print(f"  {row['label']}: {row['points']} points, mean {row['mean']:.4g}, SD {row['sd']:.3g}, second-half mean {row['second_half_mean']:.4g}")
    print(f"Chart: {path}")
    if args.open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    return 0


def cmd_reproduce(args) -> int:
    """Check a paper's published per-residue RMSF against its own trajectories."""
    from .passport import sha256_file
    from .reproduce import reproduce, summary_lines

    rset = load_source(args)
    if rset.kind == "xvg":
        raise ValueError("reproduce needs the trajectories (--run TOP TRAJ ..., or -s/-f, or --domain), not RMSF files.")
    result = reproduce(rset, args.published, method_from(args), args.published_units)
    result["source"] = {"name": rset.name, "kind": rset.kind,
                        "files": [{"file": Path(p).name, "sha256": sha256_file(p)} for p in [*rset.data_files, Path(args.published)]]}
    result["replay"] = shlex.join(["concordance", "reproduce", *source_flags(args, rset), "--published", str(args.published)]
                                  + (["--published-units", args.published_units] if args.published_units else []))
    folder = Path(args.out_dir) / "reproductions"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / f"{rset.name.replace(' ', '_')}_reproduction.json"
    out.write_text(json.dumps(result, indent=2, default=str))
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    for line in summary_lines(result, rset.name):
        print(line)
    print(f"Saved: {out}")
    return 0


def cmd_page(args) -> int:
    from .report import write_page

    path = write_page(args.out or Path(args.out_dir) / "concordance.html")
    print(f"No-install page: {path}")
    print("Open it in any browser and add RMSF files (gmx rmsf .xvg, cpptraj .dat, CSV), one per run. Nothing is uploaded.")
    if args.open:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    return 0


def cmd_domains(args) -> int:
    files = find_domain_files(args.data_dir)
    if not files:
        print(f"No mdCATH files in {args.data_dir} yet. Try: python scripts/download_demo_data.py --n 6 --dry-run")
        return 0
    for domain, path in files.items():
        info = open_domain(path)
        temps = ", ".join(f"{t} K x{len(r)}" for t, r in sorted(info.replicas.items(), key=lambda kv: int(kv[0])))
        size = path.stat().st_size / 1e6
        print(f"{domain}: {info.n_residues} residues ({info.residues[0].number}-{info.residues[-1].number}); runs: {temps}; {size:.0f} MB")
    return 0


def cmd_export(args) -> int:
    """One gmx rmsf-style .xvg per run plus a structure, ready to drop into the no-install page."""
    rset = load_source(args)
    an = analyse(rset, method_from(args))
    folder = Path(args.folder) if args.folder else Path(args.out_dir) / "export" / f"{rset.name}_{rset.condition.replace(' ', '')}"
    folder.mkdir(parents=True, exist_ok=True)
    source = ("RMSF read from your files" if rset.kind == "xvg" else
              f"C-alpha RMSF from MDAnalysis, frames {an.window.start}-{an.window.stop - 1}")
    written = []
    for k, (replica, values) in enumerate(an.rmsf_nm.items(), start=1):
        lines = [
            f"# Written by Concordance {__version__}: {rset.name}, {rset.condition}, run {k} (replica {replica})",
            f"# {source}; same layout as gmx rmsf -res",
            '@    title "RMS fluctuation"',
            '@    xaxis  label "Residue"',
            '@    yaxis  label "(nm)"',
            "@TYPE xy",
        ]
        lines += [f"{res.number:5d}   {v:8.4f}" for res, v in zip(rset.residues, values)]
        path = folder / f"run{k}_rmsf.xvg"
        path.write_text("\n".join(lines) + "\n")
        written.append(path)
    structure = rset.display_pdb()
    if structure.strip():
        (folder / "structure.pdb").write_text(structure)
        written.append(folder / "structure.pdb")
    print(f"Wrote {len(written)} files to {folder}:")
    for path in written:
        print(f"  {path.name}")
    print("Drop them all into the no-install page (concordance page --open), or run:")
    print(f"  concordance report --rmsf {folder}/run*_rmsf.xvg" + (f" -s {folder}/structure.pdb" if structure.strip() else "") + " --open")
    return 0


def cmd_rmsf(args) -> int:
    rset = load_source(args)
    an = analyse(rset, method_from(args))
    out = Path(args.csv) if args.csv else Path(args.out_dir) / "rmsf" / f"{rset.name}_{rset.condition.replace(' ', '')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    runs = list(an.rmsf_nm)
    with open(out, "w") as handle:
        handle.write(",".join(["chain", "residue_number", "residue"] + [f"rmsf_nm_run{i}" for i in range(1, len(runs) + 1)] + ["mean_nm", "sd_nm", "runs_flagging"]) + "\n")
        for i, res in enumerate(rset.residues):
            values = [an.rmsf_nm[r][i] for r in runs]
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            handle.write(",".join([res.chain, str(res.number), res.label] + [f"{v:.4f}" for v in values] + [f"{np.mean(values):.4f}", f"{sd:.4f}", str(int(an.hot.agreement[i]))]) + "\n")
    warn_geometry(an)
    if rset.kind == "xvg":
        print(f"RMSF for {rset.name}: {len(runs)} runs read from gmx rmsf output and combined into one table.")
    else:
        print(f"C-alpha RMSF for {rset.name}, {rset.condition}: {len(runs)} runs, frames {an.window.start}-{an.window.stop - 1}, computed with MDAnalysis.")
        print(f"MDAnalysis vs independent NumPy implementation: largest difference {an.check['max_abs_diff_nm']:.1e} nm.")
    print(f"Table written to {out}")
    from .xvg import write_xvg

    xvg = write_xvg(an, out.parent)
    print(f"GROMACS-style .xvg (same layout as gmx rmsf -res, one column per run): {xvg['rmsf']} and {xvg['agreement']}")
    return 0


def _markdown_validation(reports: list[dict]) -> str:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Validation",
        "",
        f"Generated by `concordance validate` (Concordance {__version__}) on {stamp}. Every number below was computed, not typed.",
        "",
        "## 1. Two independent implementations agree",
        "",
        "MDAnalysis's documented RMSF recipe (AverageStructure, AlignTraj, RMSF) against Concordance's own NumPy Kabsch implementation, on exactly the same frames.",
        "",
        "| Protein | Condition | Largest difference (nm) | Agree |",
        "|---|---|---|---|",
    ]
    for r in reports:
        c = r["engine_check"]
        tag = " (synthetic)" if r["synthetic"] else ""
        lines.append(f"| {r['name']}{tag} | {r['condition']} | {c['max_abs_diff_nm']:.1e} | {'yes' if c['agree'] else 'NO'} |")
    stored = [r for r in reports if "stored_check" in r]
    if stored:
        lines += [
            "",
            "## 2. Agreement with the values published in mdCATH",
            "",
            "MDAnalysis RMSF over each full run against the C-alpha RMSF stored in the dataset file. The dataset authors computed theirs with HTMD and did not document their alignment choices, so small differences are expected; this checks that we are measuring the same thing.",
            "",
            "| Protein | Condition | Run | Pearson r | Spearman rho | Mean abs. difference (nm) |",
            "|---|---|---|---|---|---|",
        ]
        for r in stored:
            tag = " (synthetic)" if r["synthetic"] else ""
            for i, (replica, row) in enumerate(r["stored_check"]["per_run"].items(), start=1):
                if "error" in row:
                    lines.append(f"| {r['name']}{tag} | {r['condition']} | {i} | {row['error']} | | |")
                else:
                    lines.append(f"| {r['name']}{tag} | {r['condition']} | {i} | {row['pearson']:.3f} | {row['spearman']:.3f} | {row['mean_abs_diff_nm']:.4f} |")
    lines += ["", "Rows marked synthetic come from generated test data with a known answer, not from real simulations.", ""]
    return "\n".join(lines)


def cmd_validate(args) -> int:
    sources: list[ReplicateSet] = []
    if getattr(args, "xvg", None):
        raise ValueError("validate needs trajectories or mdCATH files; gmx rmsf output has no coordinates to re-check.")
    if args.topology or args.traj or args.domain:
        sources.append(load_source(args))
    else:
        files = find_domain_files(args.data_dir)
        if not files:
            raise ValueError(f"Nothing to validate in {args.data_dir}; download data or pass --domain / -s and -f.")
        sources = [from_mdcath(path, args.temp) for path in files.values()]
    reports = []
    for rset in sources:
        report = validate(rset, method_from(args))
        reports.append(report)
        line = f"{rset.name} {rset.condition}: MDAnalysis vs NumPy largest difference {report['engine_check']['max_abs_diff_nm']:.1e} nm"
        if "stored_check" in report:
            rs = [row["pearson"] for row in report["stored_check"]["per_run"].values() if "pearson" in row]
            if rs:
                line += f"; vs stored mdCATH RMSF Pearson r {min(rs):.3f}-{max(rs):.3f}"
        print(line)
    out = Path(args.results_dir) / "validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2, default=float) + "\n")
    Path(args.markdown).write_text(_markdown_validation(reports))
    print(f"Saved {out} and {args.markdown}")
    return 0


def cmd_scan(args) -> int:
    files = find_domain_files(args.data_dir)
    if not files:
        raise ValueError(f"No mdCATH files in {args.data_dir}; run scripts/download_demo_data.py first.")
    method = method_from(args)
    rows = []
    for domain, path in files.items():
        info = open_domain(path)
        if str(args.temp) not in info.replicas:
            rows.append({"domain": domain, "skipped": f"no {args.temp} K runs"})
            continue
        an = analyse(from_mdcath(path, args.temp), method)
        counts = an.hot.counts()
        stretches = contested_stretches(an.hot.agreement, an.n, threshold=method.consistency)
        flagged = counts["flagged_in_any_run"]
        rows.append(
            {
                "domain": domain,
                "residues": info.n_residues,
                "runs": an.n,
                **counts,
                "share_of_flagged_not_consistent": round((flagged - counts["consistent"]) / flagged, 3) if flagged else None,
                "contested_stretches": [f"{an.rset.residues[a].number}-{an.rset.residues[b].number}" for a, b in stretches],
                "synthetic": an.rset.synthetic,
            }
        )
    scanned = [r for r in rows if "skipped" not in r]
    flagged_total = sum(r["flagged_in_any_run"] for r in scanned)
    consistent_total = sum(r["consistent"] for r in scanned)
    with_stretch = sum(1 for r in scanned if r["contested_stretches"])
    share = (flagged_total - consistent_total) / flagged_total if flagged_total else 0.0
    manifest = load_manifest(args.data_dir) or {}
    n_runs = scanned[0]["runs"] if scanned else 5
    statement = (
        f"Across {len(scanned)} protein domains at {args.temp} K, {share:.0%} of the residues that moved a lot in at least one run "
        f"did so in fewer than {math.ceil(method.consistency * n_runs - 1e-9)} of {n_runs} runs; {with_stretch} of {len(scanned)} domains had a stretch of "
        f"3 or more residues in a row that only some runs flagged."
    )
    report = {
        "concordance_version": __version__,
        "temperature_K": int(args.temp),
        "method": method.as_dict(),
        "selection_rule": manifest.get("selection_rule", "unknown (no data/manifest.json)"),
        "dataset_revision": manifest.get("revision"),
        "statement": statement,
        "domains": rows,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    out = Path(args.out) if args.out else Path(args.results_dir) / f"scan_{args.temp}K.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'domain':<10} {'res':>4} {'consistent':>10} {'some runs':>9} {'one run':>7}  contested stretches")
    for r in rows:
        if "skipped" in r:
            print(f"{r['domain']:<10} skipped: {r['skipped']}")
            continue
        print(f"{r['domain']:<10} {r['residues']:>4} {r['consistent']:>10} {r['replica_dependent']:>9} {r['flagged_by_a_minority']:>7}  {', '.join(r['contested_stretches']) or '-'}")
    print(f"\n{statement}\nSelection rule: {report['selection_rule']}\nSaved to {out}")
    if scanned:
        pick = max(scanned, key=lambda r: (len(r["contested_stretches"]), r["replica_dependent"]))
        print(f"Most contested domain in this sample: {pick['domain']} (pick it for a demo only if you also show this scan).")
    return 0


# ---------------------------------------------------------------- selftest


def cmd_selftest(args) -> int:
    from .synthetic import write_synthetic_domain, write_trajectory_files

    checks: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        h5 = write_synthetic_domain(Path(tmp) / "data" / "mdcath_dataset_1abcA00.h5")
        rset = from_mdcath(h5, "320")
        an = analyse(rset)
        checks.append(("MDAnalysis and the independent NumPy implementation agree", an.check["agree"], f"largest difference {an.check['max_abs_diff_nm']:.1e} nm"))
        agreement = {rset.residues[i].number: int(a) for i, a in enumerate(an.hot.agreement)}
        ok = all(agreement[r] == 5 for r in (21, 22, 23))
        checks.append(("Residues designed to move in every run are found in 5 of 5 runs", ok, f"21-23: {[agreement[r] for r in (21, 22, 23)]}"))
        ok = all(agreement[r] == 3 for r in (9, 10, 11))
        checks.append(("Residues designed to move in 3 runs are found in exactly 3", ok, f"9-11: {[agreement[r] for r in (9, 10, 11)]}"))
        c = an.flexible_region(9, 11)
        checks.append(("Region 9-11 is called replica-dependent, with 2 runs that would mislead", c.verdict == DEPENDENT and c.risk == 2, f"{c.verdict}, risk {c.risk}"))
        c = an.flexible_region(21, 23)
        checks.append(("Region 21-23 is called consistent", c.verdict == CONSISTENT, c.verdict))
        c = an.flexible_region(4, 7)
        checks.append(("A rigid region (4-7) is not called flexible", c.verdict == NOT_SEEN, c.verdict))
        c = an.folded_region(14, 19)
        checks.append(("A helix present in only 2 runs is called replica-dependent", c.verdict == DEPENDENT and c.n_yes == 2, f"{c.verdict}, {c.n_yes}/5"))
        topk = analyse(rset, Method(rule=FlexRule(rule="topk", top=0.2)))
        designed = {9, 10, 11, 21, 22, 23}
        topk_agree = {rset.residues[i].number: int(a) for i, a in enumerate(topk.hot.agreement)}
        invented = sorted(r for r, a in topk_agree.items() if a >= 2 and r not in designed)
        demoted = sorted(r for r in (21, 22, 23) if topk_agree[r] < 5)
        checks.append(("The older top-k% rule shows the artefact the default rule avoids", bool(invented or demoted), f"top-20% rule: invented {invented}, demoted {demoted}"))
        topology, trajectories = write_trajectory_files(h5, Path(tmp) / "files")
        files_an = analyse(from_files(topology, trajectories))
        same = np.array_equal(files_an.hot.agreement, an.hot.agreement)
        diff = max(float(np.abs(files_an.rmsf_nm[f] - an.rmsf_nm[m]).max()) for f, m in zip(files_an.rmsf_nm, an.rmsf_nm))
        checks.append(("The same runs written as GROMACS .xtc files give the same answer", same and diff < 0.002, f"largest RMSF difference {diff:.1e} nm (xtc stores 0.001 nm precision)"))
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = sum(1 for _, ok, _ in checks if not ok)
    print(f"\n{len(checks) - failed} of {len(checks)} checks passed on synthetic data with a designed answer.")
    return 1 if failed else 0


# ---------------------------------------------------------------- entry point


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="concordance", description="Does what your protein simulation tells you hold up when it is run again?")
    parser.add_argument("--version", action="version", version=f"concordance {__version__}")
    parser.add_argument("--data-dir", default=os.environ.get("CONCORDANCE_DATA", "data"), help="where mdCATH files live")
    parser.add_argument("--out-dir", default=os.environ.get("CONCORDANCE_OUT", "outputs"), help="where passports, structures and reports go")
    parser.add_argument("--results-dir", default="results", help="where scan and validation summaries go")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("selftest", help="check the tool on synthetic data with a known answer (no download needed)")
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("domains", help="list downloaded mdCATH proteins")
    p.set_defaults(func=cmd_domains)

    p = sub.add_parser("ask", help="check a claim across replicate runs")
    p.add_argument("question", choices=["hotspots", "flexible", "folded", "stable"])
    add_source_args(p)
    add_method_args(p)
    p.add_argument("--region", help="residue numbers, e.g. 40-55")
    p.add_argument("--kind", choices=["helix", "strand"], default="helix", help="for 'folded'")
    p.add_argument("--max-rmsd", type=float, default=0.4, help="nm, for 'stable'")
    p.add_argument("--audience", choices=AUDIENCES, default="new", help="how much detail to print")
    p.add_argument("--json", action="store_true", help="print the full evidence passport as JSON (for scripts and agents)")
    p.add_argument("--report", action="store_true", help="also write the visual report")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("report", help="write the one-file visual report")
    add_source_args(p)
    add_method_args(p)
    p.add_argument("--region", help="optional: ask whether this region is flexible, e.g. 40-55")
    p.add_argument("--audience", choices=AUDIENCES, default="new", help="who the report opens for (readers can switch)")
    p.add_argument("--open", action="store_true", help="open the report in your browser")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("page", help="write the no-install web page: anyone adds their own RMSF files in a browser")
    p.add_argument("--out", help="where to write it (default OUT_DIR/concordance.html)")
    p.add_argument("--open", action="store_true", help="open it in your browser")
    p.set_defaults(func=cmd_page)

    p = sub.add_parser("reproduce", help="check a paper's published per-residue RMSF against its own trajectories")
    add_source_args(p)
    add_method_args(p)
    p.add_argument("--published", required=True, help="the published per-residue RMSF table (.xvg, .dat, .tsv, .csv; one column per run or one mean)")
    p.add_argument("--published-units", choices=["nm", "A"], help="units of the published table if it doesn't say (otherwise inferred)")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("check", help="one command for a folder or .zip of simulation files: what's here, do the runs agree, do published numbers follow")
    p.add_argument("files", nargs="+", help="a folder, a downloaded .zip, or files")
    p.add_argument("--name", help="what to call the protein (default: from the file names)")
    p.add_argument("--reference", help="measure each run against this still region, e.g. 20-60 (for very disordered proteins)")
    p.add_argument("--no-plots", action="store_true", help="don't chart the other tables")
    p.add_argument("--open", action="store_true", help="open the report in your browser")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("plot", help="open any .xvg (or CSV/TSV) as a modern, offline chart, in place of xmgrace")
    p.add_argument("files", nargs="+", help=".xvg, .dat, .csv or .tsv files, or a folder of them; one file per run overlays them")
    p.add_argument("--title", help="chart title (default: from the file)")
    p.add_argument("--out", help="where to write the chart (default OUT_DIR/plots/<name>.html)")
    p.add_argument("--open", action="store_true", help="open it in your browser")
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("identify", help="what MD files are, from their contents (any extension), and how to analyse them")
    p.add_argument("files", nargs="+", help="files or folders")
    p.add_argument("--json", action="store_true", help="machine-readable output (for scripts and agents)")
    p.set_defaults(func=cmd_identify)

    p = sub.add_parser("export", help="one gmx rmsf-style .xvg per run plus a structure, for the no-install page")
    add_source_args(p)
    add_method_args(p)
    p.add_argument("--folder", help="where to write the files (default outputs/export/<name>)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("rmsf", help="every run's per-residue RMSF as a CSV table")
    add_source_args(p)
    add_method_args(p)
    p.add_argument("--csv", help="where to write the table")
    p.set_defaults(func=cmd_rmsf)

    p = sub.add_parser("validate", help="MDAnalysis vs NumPy vs stored mdCATH values")
    add_source_args(p)
    add_method_args(p)
    p.add_argument("--markdown", default="VALIDATION.md", help="where to write the human-readable summary")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("scan", help="agreement summary over every downloaded mdCATH protein")
    p.add_argument("--temp", default="320")
    add_method_args(p)
    p.add_argument("--out", help="where to save the scan (default results/scan_<T>K.json)")
    p.set_defaults(func=cmd_scan)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except ValueError as err:
        print(f"concordance: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
