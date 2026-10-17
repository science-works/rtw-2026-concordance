"""The analysis core shared by the command line, the visual report and the agent endpoint."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .rmsf import ca_rmsf, mdanalysis_rmsf
from .sources import ReplicateSet
from .verdict import (
    ClaimResult,
    FlexRule,
    HotspotResult,
    core_mask,
    hotspots,
    region_flexible,
    region_keeps_structure,
    stays_near_start,
    threshold_for,
)

ENGINE = "MDAnalysis: AverageStructure, then AlignTraj onto it, then RMSF (C-alpha)"


@dataclass(frozen=True)
class Method:
    """Every analysis choice, in one place, so it can be written into the passport."""

    rule: FlexRule = field(default_factory=FlexRule)
    discard: float = 0.1  # share of each run treated as equilibration and skipped
    consistency: float = 0.8  # share of runs needed for "consistent across replicas"
    min_fraction: float = 0.5  # share of a region's residues that must be flexible
    reference: tuple[int, int] | None = None  # residue numbers each run's typical movement is measured from

    def describe_reference(self) -> str:
        if not self.reference:
            return ""
        return (f" Each run's typical movement is measured over residues {self.reference[0]}-{self.reference[1]} only "
                "(for example a folded domain, so a long disordered tail doesn't set the baseline).")

    def as_dict(self) -> dict:
        return {
            "flexibility_rule": self.rule.as_dict(),
            "reference_residues": list(self.reference) if self.reference else None,
            "discard_fraction": self.discard,
            "consistency_threshold": self.consistency,
            "region_min_fraction": self.min_fraction,
            "engine": ENGINE,
            "atoms": "C-alpha",
        }


@dataclass(frozen=True)
class Window:
    frames_available: dict[str, int]
    start: int
    stop: int
    per_run: dict | None = None  # runs from different sources: each run's own (start, stop)

    @property
    def used(self) -> int:
        if self.per_run:
            return min(b - a for a, b in self.per_run.values())
        return self.stop - self.start

    def as_dict(self) -> dict:
        out = {
            "frames_available_per_run": self.frames_available,
            "common_length": self.stop,
            "frames_discarded": self.start,
            "frames_used_per_run": self.used,
            "frame_range": [self.start, self.stop],
        }
        if self.per_run:
            out["per_run"] = {r: {"frames": [a, b], "used": b - a} for r, (a, b) in self.per_run.items()}
            out["note"] = "Runs from different sources keep their own lengths: each run skips its own first share as equilibration (none for ensembles)."
        return out


def make_window(rset: ReplicateSet, discard: float) -> Window:
    """Same frames for every run: cut to the shortest run, then skip the first `discard` share."""
    if not 0 <= discard < 1:
        raise ValueError("--discard must be at least 0 and below 1.")
    counts = rset.frame_counts()
    common = min(counts.values())
    start = int(math.floor(discard * common))
    if common - start < 2:
        raise ValueError("Too few frames left to measure fluctuation; lower --discard.")
    return Window(counts, start, common)


def make_window_mixed(rset: ReplicateSet, discard: float) -> Window:
    """Runs from different sources save frames at different rates, so each run keeps its own length:
    skip each run's first `discard` share (nothing for an ensemble, which has no equilibration)."""
    if not 0 <= discard < 1:
        raise ValueError("--discard must be at least 0 and below 1.")
    counts = rset.frame_counts()
    per_run = {}
    for replica, run in zip(rset.replica_ids, rset.runs):
        n = counts[replica]
        start = 0 if run.get("ensemble") else int(math.floor(discard * n))
        if n - start < 2:
            raise ValueError(f"Run {replica} has only {n} frame(s); at least 2 are needed to measure fluctuation.")
        per_run[replica] = (start, n)
    first = per_run[rset.replica_ids[0]]
    return Window(counts, first[0], first[1], per_run)


def ca_universe(rset: ReplicateSet, coords: np.ndarray):
    """An in-memory MDAnalysis Universe holding only the C-alpha atoms of one run."""
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    n = rset.n_residues
    u = mda.Universe.empty(n, n_residues=n, atom_resindex=np.arange(n), trajectory=False)
    u.add_TopologyAttr("name", ["CA"] * n)
    u.add_TopologyAttr("type", ["C"] * n)
    u.add_TopologyAttr("resid", [r.number for r in rset.residues])
    u.add_TopologyAttr("resname", [r.name for r in rset.residues])
    u.load_new(np.asarray(coords, dtype=np.float32), format=MemoryReader)
    return u


def rmsf_mdanalysis(rset: ReplicateSet, coords: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-residue RMSF in nm for each run, computed by MDAnalysis."""
    return {r: mdanalysis_rmsf(ca_universe(rset, c), "name CA") / 10.0 for r, c in coords.items()}


def engine_check(mda_nm: dict[str, np.ndarray], coords: dict[str, np.ndarray]) -> dict:
    """Compare MDAnalysis with an independent NumPy implementation on the same frames."""
    per_run = {}
    for replica, values in mda_nm.items():
        ours = ca_rmsf(coords[replica]) / 10.0
        diff = np.abs(values - ours)
        per_run[replica] = {
            "max_abs_diff_nm": float(diff.max()),
            "pearson": float(pearsonr(values, ours).statistic) if np.std(ours) > 0 else 1.0,
        }
    worst = max(v["max_abs_diff_nm"] for v in per_run.values())
    return {
        "description": "Every RMSF value was computed twice: by MDAnalysis and by an independent NumPy implementation.",
        "max_abs_diff_nm": worst,
        "agree": bool(worst < 1e-3),
        "per_run": per_run,
    }


def chain_check(rset: ReplicateSet, coords: dict[str, np.ndarray], limit: float = 15.0) -> dict:
    """Are neighbouring residues still neighbours in every frame? A C-alpha to C-alpha distance is about
    3.8 angstrom; far larger jumps mean the protein is split across the periodic box (or the files are broken),
    which makes any RMSF meaningless."""
    res = rset.residues
    pairs = np.array([i for i in range(len(res) - 1) if res[i].chain == res[i + 1].chain and res[i + 1].number == res[i].number + 1])
    out = {"description": "Largest distance between C-alpha atoms of neighbouring residues, over every analysed frame.", "limit_A": limit}
    if pairs.size == 0:
        return {**out, "ok": True, "checked": False}
    worst, where, broken, total = 0.0, None, 0, 0
    for replica, c in coords.items():
        d = np.linalg.norm(c[:, pairs + 1] - c[:, pairs], axis=2)
        broken += int((d > limit).any(axis=1).sum())
        total += d.shape[0]
        if d.max() > worst:
            frame, k = np.unravel_index(int(d.argmax()), d.shape)
            worst = float(d.max())
            where = {"run": replica, "frame": int(frame), "between": f"{res[pairs[k]].label}-{res[pairs[k] + 1].label}"}
    ok = worst <= limit
    record = {**out, "ok": ok, "checked": True, "largest_A": round(worst, 2), "frames_with_a_break": broken, "frames_checked": total}
    if not ok:
        record["where"] = where
        record["advice"] = ("The protein looks split across the periodic box in some frames. Make molecules whole first "
                            "(GROMACS: gmx trjconv -pbc mol), or let Concordance do it with --unwrap (needs a topology with bonds, e.g. .tpr).")
    return record


@dataclass
class Analysis:
    rset: ReplicateSet
    method: Method
    window: Window
    rmsf_nm: dict[str, np.ndarray]
    core: np.ndarray
    hot: HotspotResult
    check: dict
    geometry: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.rmsf_nm)

    @property
    def ref(self) -> np.ndarray | None:
        return reference_mask(self.rset, self.method)

    def thresholds(self) -> dict[str, float]:
        return {r: threshold_for(p, self.core, self.method.rule, self.ref) for r, p in self.rmsf_nm.items()}

    def flexible_region(self, start: int, end: int) -> ClaimResult:
        positions = self.rset.residue_positions(start, end)
        claim = f"residues {start}-{end} of {self.rset.name} are flexible ({self.rset.condition})"
        return region_flexible(
            self.rmsf_nm, positions, claim, self.core, self.method.rule, self.method.min_fraction, self.method.consistency, self.ref
        )

    def folded_region(self, start: int, end: int, kind: str = "helix", min_fraction: float = 0.7) -> ClaimResult:
        dssp = {r: v[self.window.start : self.window.stop] for r, v in self.rset.dssp().items()}
        positions = self.rset.residue_positions(start, end)
        claim = f"residues {start}-{end} of {self.rset.name} stay {kind} ({self.rset.condition})"
        return region_keeps_structure(dssp, positions, claim, kind, min_fraction, self.method.consistency)

    def stays_near_start(self, max_nm: float = 0.4) -> ClaimResult:
        rmsd = {r: v[self.window.start : self.window.stop] for r, v in self.rset.stored("rmsd").items()}
        claim = f"{self.rset.name} stays close to its starting structure ({self.rset.condition})"
        return stays_near_start(rmsd, claim, max_nm, threshold=self.method.consistency)

    def run_summaries(self) -> dict[str, dict]:
        """What each run, analysed on its own, would have told you about the whole protein."""
        consistent = self.hot.agreement >= math.ceil(self.method.consistency * self.n - 1e-9)
        out = {}
        for replica, flags in self.hot.flags.items():
            out[replica] = {
                "flexible_residues": int(flags.sum()),
                "missed_consistent": int((consistent & ~flags).sum()),
                "extra_not_consistent": int((flags & ~consistent).sum()),
                "threshold_nm": round(threshold_for(self.rmsf_nm[replica], self.core, self.method.rule, self.ref), 4),
                "median_rmsf_nm": round(float(np.median(self.rmsf_nm[replica][self.core])), 4),
            }
        return out


def disorder_hint(rmsf_nm: dict, core: np.ndarray, numbers: list[int], has_reference: bool) -> dict | None:
    """Spot proteins where many residues move far more than the stillest ones (e.g. a long disordered tail).

    Each run's threshold is relative to its typical residue, so when most of the chain is disordered the
    baseline itself is disordered and real flexible regions can be missed. This suggests measuring against
    the still part instead (--reference). Same rule in logic.js (disorderHint).
    """
    if has_reference:
        return None
    mean = np.mean(np.vstack(list(rmsf_nm.values())), axis=0)
    scored = np.flatnonzero(core)
    if scored.size < 20:
        return None
    floor = float(np.percentile(mean[scored], 10))
    high = mean[scored] > 3 * floor
    share = float(high.mean())
    if share <= 0.4:
        return None
    still = [int(i) for i in scored if mean[i] <= 3 * floor]
    groups: list[list[int]] = []
    for i in still:
        if groups and i - groups[-1][-1] <= 10:
            groups[-1].append(i)
        else:
            groups.append([i])
    best = max(groups, key=lambda g: g[-1] - g[0]) if groups else None
    if not best or best[-1] - best[0] < 9:
        return {"share": round(share, 3), "reference": None}
    return {"share": round(share, 3), "reference": [int(numbers[best[0]]), int(numbers[best[-1]])]}


def reference_mask(rset: ReplicateSet, method: Method) -> np.ndarray | None:
    """Residues each run's typical movement is measured from (None: all scored residues)."""
    if not method.reference:
        return None
    a, b = method.reference
    mask = np.array([a <= r.number <= b for r in rset.residues])
    core = core_mask(rset.n_residues, method.rule.trim_termini)
    if (mask & core).sum() < 5:
        raise ValueError(f"The reference residues {a}-{b} include fewer than 5 scored residues; choose a longer stretch.")
    return mask


def analyse(rset: ReplicateSet, method: Method | None = None) -> Analysis:
    method = method or Method()
    ref = reference_mask(rset, method)
    if rset.supplied_rmsf is not None:
        rmsf_nm = {r: np.asarray(v, dtype=float) for r, v in rset.supplied_rmsf.items()}
        core = core_mask(rset.n_residues, method.rule.trim_termini)
        hot = hotspots(rmsf_nm, core, method.rule, method.consistency, ref)
        check = {
            "description": "Not applicable: the RMSF values were read from your files (gmx rmsf, cpptraj or a table), not computed here. "
            "Frames, fitting and atom selection are whatever was used when they were made.",
            "max_abs_diff_nm": None,
            "agree": None,
        }
        geometry = {"ok": True, "checked": False, "description": "Not applicable: no coordinates were supplied."}
        return Analysis(rset, method, Window({}, 0, 0), rmsf_nm, core, hot, check, geometry)
    if rset.kind == "mixed":
        window = make_window_mixed(rset, method.discard)
        coords = rset.ca_coords(windows=window.per_run)
    else:
        window = make_window(rset, method.discard)
        coords = rset.ca_coords(window.start, window.stop)
    rmsf_nm = rmsf_mdanalysis(rset, coords)
    core = core_mask(rset.n_residues, method.rule.trim_termini)
    hot = hotspots(rmsf_nm, core, method.rule, method.consistency, ref)
    return Analysis(rset, method, window, rmsf_nm, core, hot, engine_check(rmsf_nm, coords), chain_check(rset, coords))


def validate(rset: ReplicateSet, method: Method | None = None) -> dict:
    """Three-way check: MDAnalysis vs NumPy (same frames) and, for mdCATH, vs the dataset's stored RMSF."""
    method = method or Method()
    an = analyse(rset, method)
    out = {"name": rset.name, "condition": rset.condition, "synthetic": rset.synthetic, "engine_check": an.check}
    if rset.kind == "mdcath":
        full = rset.ca_coords(0, None)  # each run's full length, as the dataset's own RMSF most likely used
        ours = rmsf_mdanalysis(rset, full)
        stored = rset.stored("rmsf")
        rows = {}
        for replica, values in ours.items():
            theirs = stored[replica]
            if values.shape != theirs.shape:
                rows[replica] = {"error": f"length mismatch: ours {values.shape[0]}, stored {theirs.shape[0]}"}
                continue
            rows[replica] = {
                "pearson": float(pearsonr(values, theirs).statistic),
                "spearman": float(spearmanr(values, theirs).statistic),
                "mean_abs_diff_nm": float(np.abs(values - theirs).mean()),
                "max_abs_diff_nm": float(np.abs(values - theirs).max()),
                "frames": int(full[replica].shape[0]),
            }
        out["stored_check"] = {
            "description": "MDAnalysis RMSF over each full run vs the RMSF stored in the mdCATH file "
            "(computed by the dataset authors with HTMD; their alignment choices are not documented).",
            "per_run": rows,
        }
    return out
