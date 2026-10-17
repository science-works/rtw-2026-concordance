"""Check a paper's published per-residue RMSF against its own trajectories.

Papers and databases often publish RMSF for each simulation, in a table whose units, residue
numbering and column order aren't documented. `reproduce()` recomputes RMSF from the trajectories
with a few common recipes, lines the published table up with the protein (units, numbering offset,
which column is which run), and reports how closely each recipe reproduces the published numbers,
and whether the same residues come out as flexible. The thresholds for calling a result reproduced
are stated conventions, written into the result.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from .analysis import Method, make_window, make_window_mixed, rmsf_mdanalysis
from .rmsf import superimpose
from .verdict import core_mask, flag_residues
from .xvg import read_rmsf_table

RECIPES = [
    {"name": "fit to the average structure, skip the first 10%", "fit": "average", "discard": 0.1},
    {"name": "fit to the average structure, all frames", "fit": "average", "discard": 0.0},
    {"name": "fit to the first frame, skip the first 10%", "fit": "first", "discard": 0.1},
    {"name": "fit to the first frame, all frames", "fit": "first", "discard": 0.0},
]
REPRODUCED_R, LARGELY_R, CLOSE_FRACTION = 0.95, 0.85, 0.15


def _rmsf_first_frame(coords: np.ndarray) -> np.ndarray:
    """C-alpha RMSF (angstrom) after fitting every frame onto the first one."""
    aligned = superimpose(coords, coords[0])
    return np.sqrt(((aligned - aligned.mean(axis=0)) ** 2).sum(axis=2).mean(axis=0))


def compute(rset, recipe: dict) -> dict[str, np.ndarray]:
    """Per-run RMSF (nm) under one recipe."""
    if rset.kind == "mixed":
        window = make_window_mixed(rset, recipe["discard"])
        coords = rset.ca_coords(windows=window.per_run)
    else:
        window = make_window(rset, recipe["discard"])
        coords = rset.ca_coords(window.start, window.stop)
    if recipe["fit"] == "average":
        return rmsf_mdanalysis(rset, coords)
    return {r: _rmsf_first_frame(c) / 10.0 for r, c in coords.items()}


def _align_numbers(ours: list[int], theirs: np.ndarray, numbered: bool, our_mean: np.ndarray, their_mean: np.ndarray) -> tuple[int, str]:
    """Shift to add to the published residue numbers so they match ours, and how it was decided."""
    if numbered and len(theirs) == len(ours):
        return ours[0] - 1, "the table has no residue numbers; rows taken in order"
    index = {n: i for i, n in enumerate(ours)}
    best = (None, -2.0, 0)
    for shift in range(-500, 501):
        pairs = [(index[n + shift], k) for k, n in enumerate(theirs) if (n + shift) in index]
        if len(pairs) < max(5, 0.6 * min(len(ours), len(theirs))):
            continue
        a = our_mean[[p for p, _ in pairs]]
        b = their_mean[[k for _, k in pairs]]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        r = float(pearsonr(a, b).statistic)
        if r > best[1] + 1e-9 or (abs(r - best[1]) <= 1e-9 and abs(shift) < abs(best[0] or 10**9)):
            best = (shift, r, len(pairs))
    if best[0] is None:
        raise ValueError("Couldn't line the published table up with the protein's residues.")
    how = "residue numbers match" if best[0] == 0 else f"the table's residue numbers are offset by {best[0]:+d} from the trajectories' (found by best match)"
    return best[0], how


def reproduce(rset, published_path, method: Method | None = None, published_units: str | None = None) -> dict:
    method = method or Method()
    table = read_rmsf_table(published_path)
    theirs_raw = np.column_stack([c for _, c in table.columns])  # rows x columns
    labels = [name for name, _ in table.columns]
    ours_numbers = [r.number for r in rset.residues]
    results = []
    first_ours = None
    for recipe in RECIPES:
        ours = compute(rset, recipe)
        runs = list(ours)
        ours_matrix = np.column_stack([ours[r] for r in runs])  # residues x runs (nm)
        if first_ours is None:
            first_ours = ours_matrix
        shift, numbering = _align_numbers(ours_numbers, table.numbers, table.numbered, ours_matrix.mean(axis=1), theirs_raw.mean(axis=1))
        index = {n: i for i, n in enumerate(ours_numbers)}
        rows = [(index[n + shift], k) for k, n in enumerate(table.numbers) if (n + shift) in index]
        our_rows = np.array([p for p, _ in rows])
        their_rows = np.array([k for _, k in rows])
        a = ours_matrix[our_rows]
        b = theirs_raw[their_rows]
        # units: stated in the file, or inferred from the scale (angstrom tables are ~10x nm)
        ratio = float(np.median(b) / np.median(a)) if np.median(a) > 0 else 1.0
        if published_units:
            unit, unit_how = published_units, "given"
        elif table.unit:
            unit, unit_how = table.unit, "from the file header"
        else:
            unit = "A" if 5 <= ratio <= 20 else "nm"
            unit_how = f"inferred: the published values are {ratio:.1f}x ours"
        b_nm = b / 10.0 if unit == "A" else b
        # which published column is which run
        n_runs, n_cols = a.shape[1], b_nm.shape[1]
        if n_cols > 1 or n_runs == 1:
            score = np.array([[pearsonr(a[:, i], b_nm[:, j]).statistic if np.std(a[:, i]) > 0 and np.std(b_nm[:, j]) > 0 else -1
                               for j in range(n_cols)] for i in range(n_runs)])
            ri, cj = linear_sum_assignment(-score)  # works when the counts differ: the extra runs or columns stay unpaired
            pairs = [(int(i), int(j)) for i, j in zip(ri, cj)]
            compare = [(f"run {i + 1}", labels[j], a[:, i], b_nm[:, j]) for i, j in pairs]
            if n_cols == n_runs:
                matching = "one published column per run, paired by best match" + ("" if all(i == j for i, j in pairs) else " (not in the same order as the files)")
            elif n_cols < n_runs:
                left = [f"run {i + 1}" for i in range(n_runs) if i not in ri]
                matching = (f"{n_cols} published columns for {n_runs} runs: each column paired with its best-matching run; "
                            f"{', '.join(left)} matched no published column")
            else:
                left = [labels[j] for j in range(n_cols) if j not in cj]
                matching = (f"{n_cols} published columns for {n_runs} run(s): each run paired with its best-matching column; "
                            f"published {', '.join(left)} matched no run here")
        else:
            compare = [("mean of runs", labels[0], a.mean(axis=1), b_nm[:, 0])]
            matching = f"one published column for {n_runs} runs, taken as their mean"
        counts_differ = n_cols > 1 and n_cols != n_runs
        rows_out = []
        for ours_label, their_label, x, y in compare:
            diff = np.abs(x - y)
            rows_out.append({
                "ours": ours_label, "published": their_label,
                "pearson": round(float(pearsonr(x, y).statistic), 4) if np.std(x) > 0 and np.std(y) > 0 else None,
                "spearman": round(float(spearmanr(x, y).statistic), 4) if np.std(x) > 0 and np.std(y) > 0 else None,
                "rmse_nm": round(float(np.sqrt((diff ** 2).mean())), 4),
                "max_abs_diff_nm": round(float(diff.max()), 4),
                "median_relative_diff": round(float(np.median(diff / np.maximum(y, 1e-6))), 4),
            })
        # does the finding hold: the same residues flagged as flexible from their numbers and from ours?
        core = core_mask(len(our_rows), method.rule.trim_termini)
        same_flags = []
        for _, _, x, y in compare:
            fx, fy = flag_residues(x, core, method.rule), flag_residues(y, core, method.rule)
            union = int((fx | fy).sum())
            same_flags.append({"flagged_from_ours": int(fx.sum()), "flagged_from_published": int(fy.sum()), "in_both": int((fx & fy).sum()),
                               "jaccard": round(float((fx & fy).sum() / union), 3) if union else 1.0})
        rs = [r["pearson"] for r in rows_out if r["pearson"] is not None]
        results.append({
            "recipe": recipe["name"], "fit": recipe["fit"], "discard": recipe["discard"],
            "units": {"published": "angstrom" if unit == "A" else "nm", "how": unit_how},
            "numbering": numbering, "residues_compared": int(len(rows)), "matching": matching, "counts_differ": counts_differ,
            "comparisons": rows_out, "flexible_residues": same_flags,
            "mean_pearson": round(float(np.mean(rs)), 4) if rs else None,
            "mean_rmse_nm": round(float(np.mean([r["rmse_nm"] for r in rows_out])), 4),
        })
    best = max(results, key=lambda r: ((r["mean_pearson"] or -1), -r["mean_rmse_nm"]))
    worst_r = min(r["pearson"] for r in best["comparisons"] if r["pearson"] is not None)
    close = all(r["median_relative_diff"] <= CLOSE_FRACTION for r in best["comparisons"])
    if worst_r >= REPRODUCED_R and close:
        verdict = "reproduced"
    elif worst_r >= LARGELY_R:
        verdict = "largely reproduced"
    else:
        verdict = "not reproduced"
    caveats = [
        "C-alpha RMSF only; a paper that averaged over all atoms of each residue will differ in scale but should correlate.",
        "Frames, fitting and selection in the paper may differ from all four recipes tried here.",
        "Agreement shows the published numbers follow from these trajectories, not that the simulation is right.",
    ]
    if best["counts_differ"]:
        caveats.insert(0, "The table and the trajectories have different numbers of runs, so only the paired ones were compared: "
                          "check these are the runs the paper used.")
        if verdict == "reproduced":
            verdict = "largely reproduced"  # the runs compared aren't the full set on either side
    return {
        "published_file": str(published_path),
        "verdict": verdict,
        "best_recipe": best["recipe"],
        "rule": (f"'reproduced': every compared profile has Pearson r >= {REPRODUCED_R} and a median difference within "
                 f"{CLOSE_FRACTION:.0%} of the published value; 'largely reproduced': every r >= {LARGELY_R}, or the table and the trajectories have different numbers of runs; otherwise 'not reproduced'. "
                 "These are stated conventions, not statistical tests."),
        "recipes": results,
        "caveats": caveats,
    }


def summary_lines(result: dict, name: str) -> list[str]:
    best = next(r for r in result["recipes"] if r["recipe"] == result["best_recipe"])
    word = {"reproduced": "Reproduced", "largely reproduced": "Largely reproduced", "not reproduced": "Not reproduced"}[result["verdict"]]
    lines = [f"{word}: the published RMSF for {name} " + ("follows from its trajectories." if result["verdict"] != "not reproduced" else "doesn't follow from its trajectories with the recipes tried.")]
    lines.append(f"Best match: {best['recipe']}.")
    lines.append(f"Published units: {best['units']['published']} ({best['units']['how']}). Numbering: {best['numbering']}. {best['matching'][0].upper() + best['matching'][1:]}.")
    if best["counts_differ"]:
        lines.append("Note: " + result["caveats"][0])
    for comp, flags in zip(best["comparisons"], best["flexible_residues"]):
        lines.append(f"  {comp['ours']} vs published {comp['published']}: r = {comp['pearson']}, RMSE {comp['rmse_nm']:.3f} nm, "
                     f"max difference {comp['max_abs_diff_nm']:.3f} nm; flexible residues: {flags['in_both']} in both, "
                     f"{flags['flagged_from_ours']} from ours, {flags['flagged_from_published']} from theirs")
    lines.append("All recipes tried:")
    for r in result["recipes"]:
        lines.append(f"  {'*' if r is best else ' '} {r['recipe']}: mean r = {r['mean_pearson']}, mean RMSE {r['mean_rmse_nm']:.3f} nm")
    lines.append(result["rule"])
    return lines
