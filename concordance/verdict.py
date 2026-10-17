"""Turn per-replicate numbers into verdicts about a scientific claim.

Every rule here is explicit and adjustable, and is written into the evidence passport.
A verdict says how far independent runs agree with each other. Agreement is not proof
that an answer is right (replicates started from one structure can agree and all be
wrong), but disagreement is a clear warning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

CONSISTENT = "consistent across replicas"
DEPENDENT = "replica-dependent"
NOT_SEEN = "not seen in most replicas"

HELIX_CODES = frozenset("HGI")
STRAND_CODES = frozenset("EB")


# ---------------------------------------------------------------- rules


@dataclass(frozen=True)
class FlexRule:
    """How a residue is called 'flexible' within one run."""

    rule: str = "mad"  # "mad": median + k x MAD (with a floor); "topk": the run's top fraction
    k: float = 3.0
    min_ratio: float = 1.5  # floor for the mad rule: also at least this many times the run's median
    top: float = 0.15
    trim_termini: int = 3

    def describe(self) -> str:
        trim = (
            f" The first and last {self.trim_termini} residues are not scored, because chain ends are always floppy."
            if self.trim_termini
            else ""
        )
        if self.rule == "topk":
            core = f"a residue counts as flexible in a run if its C-alpha RMSF is in that run's top {self.top:.0%}."
        else:
            core = (
                "a residue counts as flexible in a run if its C-alpha RMSF is more than "
                f"{self.k:g} median absolute deviations above that run's median, and at least "
                f"{self.min_ratio:g} times that median."
            )
        return core + trim

    def as_dict(self) -> dict:
        if self.rule == "topk":
            return {"rule": "topk", "top_fraction": self.top, "trim_termini": self.trim_termini}
        return {"rule": "mad", "k": self.k, "min_ratio": self.min_ratio, "trim_termini": self.trim_termini}


def core_mask(n_residues: int, trim: int) -> np.ndarray:
    """True for residues that are scored (chain ends trimmed when the chain is long enough)."""
    mask = np.ones(n_residues, dtype=bool)
    if trim > 0 and n_residues > 2 * trim + 3:
        mask[:trim] = False
        mask[-trim:] = False
    return mask


def flag_residues(rmsf: np.ndarray, core: np.ndarray, rule: FlexRule, ref: np.ndarray | None = None) -> np.ndarray:
    """Boolean flags (flexible in this run) for scored residues; unscored residues are False.

    `ref` (optional) marks the residues the run's typical movement is measured from, for example
    a folded domain when a long disordered tail would otherwise set the baseline. Default: all scored.
    """
    rmsf = np.asarray(rmsf, dtype=float)
    values = rmsf[core if ref is None else (core & ref)]
    if rule.rule == "topk":
        k = max(1, int(round(rule.top * len(values))))
        threshold = np.sort(values)[-k]
        return core & (rmsf >= threshold)
    return core & (rmsf > threshold_for(rmsf, core, rule, ref))


def threshold_for(rmsf: np.ndarray, core: np.ndarray, rule: FlexRule, ref: np.ndarray | None = None) -> float:
    values = np.asarray(rmsf, dtype=float)[core if ref is None else (core & ref)]
    if rule.rule == "topk":
        k = max(1, int(round(rule.top * len(values))))
        return float(np.sort(values)[-k])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(median + rule.k * mad, rule.min_ratio * median)


# ---------------------------------------------------------------- agreement maths


def classify(n_yes: int, n_total: int, threshold: float = 0.8) -> str:
    """Consistent if at least `threshold` of runs say yes; not seen if at most the mirror image."""
    if n_total <= 0:
        raise ValueError("Need at least one replicate.")
    need = math.ceil(threshold * n_total - 1e-9)
    if n_yes >= need:
        return CONSISTENT
    if n_yes <= n_total - need:
        return NOT_SEEN
    return DEPENDENT


def majority(answers: list[bool]) -> bool | None:
    yes = sum(bool(a) for a in answers)
    no = len(answers) - yes
    if yes == no:
        return None
    return yes > no


def single_run_risk(answers: dict[str, bool]) -> int | None:
    """How many runs, analysed alone, give the opposite answer to the majority of all runs."""
    consensus = majority(list(answers.values()))
    if consensus is None:
        return None
    return sum(1 for a in answers.values() if bool(a) != consensus)


def agreement_curve(answers: dict[str, bool]) -> dict[int, dict[str, float]]:
    """For every study size k, the share of all k-run subsets whose majority matches the all-runs answer.

    Exact enumeration (no random resampling), so it is reproducible. Even k can tie;
    ties are reported as 'undecided' rather than hidden.
    """
    full = majority(list(answers.values()))
    replicas = list(answers)
    curve: dict[int, dict[str, float]] = {}
    if full is None:
        return curve
    for k in range(1, len(replicas) + 1):
        subsets = list(combinations(replicas, k))
        same = opposite = undecided = 0
        for subset in subsets:
            m = majority([answers[r] for r in subset])
            if m is None:
                undecided += 1
            elif m == full:
                same += 1
            else:
                opposite += 1
        n = len(subsets)
        curve[k] = {"same": same / n, "opposite": opposite / n, "undecided": undecided / n}
    return curve


def pairwise_spearman(profiles: dict[str, np.ndarray], core: np.ndarray | None = None) -> dict[str, float]:
    out = {}
    for a, b in combinations(list(profiles), 2):
        x, y = profiles[a], profiles[b]
        if core is not None:
            x, y = x[core], y[core]
        out[f"{a}-{b}"] = float(spearmanr(x, y).statistic)
    return out


# ---------------------------------------------------------------- claims


@dataclass
class ClaimResult:
    claim: str
    rule: str
    answers: dict[str, bool]
    details: dict[str, dict]
    threshold: float = 0.8
    extra: dict = field(default_factory=dict)

    @property
    def n_replicas(self) -> int:
        return len(self.answers)

    @property
    def n_yes(self) -> int:
        return sum(bool(a) for a in self.answers.values())

    @property
    def verdict(self) -> str:
        return classify(self.n_yes, self.n_replicas, self.threshold)

    @property
    def consensus(self) -> bool | None:
        return majority(list(self.answers.values()))

    @property
    def risk(self) -> int | None:
        return single_run_risk(self.answers)

    @property
    def curve(self) -> dict[int, dict[str, float]]:
        return agreement_curve(self.answers)

    def plain_sentence(self) -> str:
        n, yes = self.n_replicas, self.n_yes
        if self.risk is None:
            return f"The runs split {yes} to {n - yes}, so there is no majority answer. More runs are needed."
        if self.risk == 0:
            return f"All {n} runs give the same answer when analysed on their own."
        word = "yes" if not self.consensus else "no"
        return (
            f"{self.risk} of the {n} runs, analysed alone, would have told you {word}. "
            + ("Treat this as unresolved until you have more runs." if self.verdict == DEPENDENT else "")
        ).strip()

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "rule": self.rule,
            "verdict": self.verdict,
            "replicas_supporting": self.n_yes,
            "replicas_total": self.n_replicas,
            "consensus": self.consensus,
            "single_run_risk": self.risk,
            "agreement_curve": {str(k): v for k, v in self.curve.items()},
            "answers": {r: bool(a) for r, a in self.answers.items()},
            "per_replica": self.details,
            "consistency_threshold": self.threshold,
            "plain_sentence": self.plain_sentence(),
            **self.extra,
        }

    def summary(self) -> str:
        lines = [f"Claim: {self.claim}", f"Rule:  {self.rule}", ""]
        for i, (replica, detail) in enumerate(self.details.items(), start=1):
            answer = "yes" if self.answers[replica] else "no "
            lines.append(f"  run {i}: {answer}  ({detail.get('note', '')})")
        lines.append("")
        lines.append(f"Verdict: {self.verdict.upper()}. {self.n_yes} of {self.n_replicas} runs support this claim.")
        lines.append(self.plain_sentence())
        if self.curve:
            parts = []
            for k, v in self.curve.items():
                text = f"{k} run{'s' if k > 1 else ''}: {v['same']:.0%}"
                if v["undecided"]:
                    text += f" ({v['undecided']:.0%} tied)"
                parts.append(text)
            lines.append(f"With fewer runs, how often you'd reach the same answer as all {self.n_replicas}: " + " | ".join(parts))
        return "\n".join(lines)


def region_flexible(
    rmsf_nm: dict[str, np.ndarray],
    positions: np.ndarray,
    claim: str,
    core: np.ndarray,
    rule: FlexRule,
    min_fraction: float = 0.5,
    threshold: float = 0.8,
    ref: np.ndarray | None = None,
) -> ClaimResult:
    """Is a region flexible? Checked separately in each replicate."""
    scored = np.asarray([p for p in positions if core[p]], dtype=int)
    if scored.size == 0:
        raise ValueError("That region lies entirely in the unscored chain ends; use --trim-termini 0 to include them.")
    answers, details = {}, {}
    for replica, profile in rmsf_nm.items():
        flags = flag_residues(profile, core, rule, ref)
        flagged = int(flags[scored].sum())
        yes = flagged / scored.size >= min_fraction
        answers[replica] = bool(yes)
        details[replica] = {
            "flexible_residues_in_region": flagged,
            "region_residues_scored": int(scored.size),
            "region_mean_rmsf_nm": round(float(profile[scored].mean()), 4),
            "run_median_rmsf_nm": round(float(np.median(profile[core])), 4),
            "run_threshold_nm": round(threshold_for(profile, core, rule, ref), 4),
            "note": f"{flagged} of {scored.size} residues flexible; region mean RMSF {profile[scored].mean():.3f} nm",
        }
    text = rule.describe() + f" The region counts as flexible in a run if at least {min_fraction:.0%} of its scored residues do."
    return ClaimResult(
        claim,
        text,
        answers,
        details,
        threshold=threshold,
        extra={"pairwise_spearman": pairwise_spearman(rmsf_nm, core), "region_positions": [int(p) for p in scored]},
    )


def region_keeps_structure(
    dssp: dict[str, np.ndarray],
    positions: np.ndarray,
    claim: str,
    kind: str = "helix",
    min_fraction: float = 0.7,
    threshold: float = 0.8,
) -> ClaimResult:
    """Does a region stay helical (or strand) through the run?"""
    codes = HELIX_CODES if kind == "helix" else STRAND_CODES
    answers, details = {}, {}
    for replica, frames in dssp.items():
        fraction = float(np.isin(frames[:, positions], list(codes)).mean())
        answers[replica] = fraction >= min_fraction
        details[replica] = {
            "fraction_in_state": round(fraction, 4),
            "frames": int(frames.shape[0]),
            "note": f"{fraction:.0%} of residue-frames are {kind} ({frames.shape[0]} frames)",
        }
    rule = (
        f"the region counts as {kind} in a run if at least {min_fraction:.0%} of its residue-frames "
        f"have DSSP codes {'/'.join(sorted(codes))} (as stored in mdCATH)."
    )
    return ClaimResult(claim, rule, answers, details, threshold=threshold)


def stays_near_start(
    rmsd_nm: dict[str, np.ndarray],
    claim: str,
    max_nm: float = 0.4,
    tail_fraction: float = 0.2,
    threshold: float = 0.8,
) -> ClaimResult:
    """Does the domain stay close to its starting structure (no large drift or unfolding)?"""
    answers, details = {}, {}
    for replica, series in rmsd_nm.items():
        n_tail = max(1, int(round(tail_fraction * len(series))))
        tail_mean = float(np.mean(series[-n_tail:]))
        answers[replica] = tail_mean <= max_nm
        details[replica] = {
            "tail_mean_rmsd_nm": round(tail_mean, 4),
            "frames": int(len(series)),
            "note": f"mean RMSD over the last {tail_fraction:.0%} of the run: {tail_mean:.2f} nm",
        }
    rule = (
        f"the domain counts as staying near its start in a run if its mean RMSD over the last "
        f"{tail_fraction:.0%} of frames is at most {max_nm} nm (RMSD as stored in mdCATH)."
    )
    return ClaimResult(claim, rule, answers, details, threshold=threshold)


# ---------------------------------------------------------------- whole-protein view


@dataclass
class HotspotResult:
    """Which residues are flexible, and in how many runs (-1 = not scored)."""

    agreement: np.ndarray
    flags: dict[str, np.ndarray]
    core: np.ndarray
    n_replicas: int
    jaccard_vs_consensus: dict[str, float]
    pairwise_spearman: dict[str, float]
    rule: FlexRule
    threshold: float = 0.8

    def classes(self) -> list[str]:
        out = []
        for n in self.agreement:
            if n < 0:
                out.append("not scored")
            elif n == 0:
                out.append("never flagged")
            else:
                out.append(classify(int(n), self.n_replicas, self.threshold))
        return out

    def counts(self) -> dict[str, int]:
        classes = self.classes()
        return {
            "scored_residues": int(self.core.sum()),
            "flagged_in_any_run": int((self.agreement > 0).sum()),
            "consistent": classes.count(CONSISTENT),
            "replica_dependent": classes.count(DEPENDENT),
            "flagged_by_a_minority": int(sum(1 for n, c in zip(self.agreement, classes) if n > 0 and c == NOT_SEEN)),
        }


def hotspots(rmsf_nm: dict[str, np.ndarray], core: np.ndarray, rule: FlexRule, threshold: float = 0.8, ref: np.ndarray | None = None) -> HotspotResult:
    flags = {r: flag_residues(p, core, rule, ref) for r, p in rmsf_nm.items()}
    stacked = np.vstack(list(flags.values()))
    agreement = stacked.sum(axis=0).astype(int)
    agreement[~core] = -1
    n = stacked.shape[0]
    consensus = (stacked.sum(axis=0) * 2 > n) & core
    jaccard = {}
    for replica, mask in flags.items():
        union = (mask | consensus).sum()
        jaccard[replica] = float((mask & consensus).sum() / union) if union else 1.0
    return HotspotResult(
        agreement=agreement,
        flags=flags,
        core=core,
        n_replicas=n,
        jaccard_vs_consensus=jaccard,
        pairwise_spearman=pairwise_spearman(rmsf_nm, core),
        rule=rule,
        threshold=threshold,
    )


def contested_stretches(agreement: np.ndarray, n_replicas: int, min_length: int = 3, threshold: float = 0.8) -> list[tuple[int, int]]:
    """Runs of consecutive scored residues that some, but not most, replicas call flexible (0-based, inclusive)."""
    contested = np.array([a > 0 and classify(int(a), n_replicas, threshold) == DEPENDENT for a in agreement])
    stretches, start = [], None
    for i, flag in enumerate(contested):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            if i - start >= min_length:
                stretches.append((start, i - 1))
            start = None
    if start is not None and len(contested) - start >= min_length:
        stretches.append((start, len(contested) - 1))
    return stretches
