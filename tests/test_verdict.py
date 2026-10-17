"""Verdicts: the agreement maths and the designed answers."""

from concordance.analysis import Method, analyse
from concordance.sources import from_mdcath
from concordance.verdict import CONSISTENT, DEPENDENT, NOT_SEEN, FlexRule, agreement_curve, classify, single_run_risk


def test_classify_five_runs():
    assert [classify(n, 5) for n in range(6)] == [NOT_SEEN, NOT_SEEN, DEPENDENT, DEPENDENT, CONSISTENT, CONSISTENT]


def test_classify_three_runs_has_no_minority_band():
    assert [classify(n, 3) for n in range(4)] == [NOT_SEEN, DEPENDENT, DEPENDENT, CONSISTENT]


def test_risk_and_exact_agreement_curve_with_ties():
    answers = {"0": True, "1": True, "2": True, "3": False, "4": False}
    assert single_run_risk(answers) == 2
    curve = agreement_curve(answers)
    assert curve[1] == {"same": 0.6, "opposite": 0.4, "undecided": 0.0}
    assert curve[2] == {"same": 0.3, "opposite": 0.1, "undecided": 0.6}
    assert curve[3]["same"] == 0.7 and curve[5]["same"] == 1.0
    assert single_run_risk({"0": True, "1": False}) is None


def test_designed_answers_are_recovered(domain_path):
    an = analyse(from_mdcath(domain_path, "320"))
    by_number = {an.rset.residues[i].number: int(a) for i, a in enumerate(an.hot.agreement)}
    assert [by_number[r] for r in (21, 22, 23)] == [5, 5, 5]
    assert [by_number[r] for r in (9, 10, 11)] == [3, 3, 3]
    assert by_number[1] == -1 and by_number[30] == -1  # chain ends are not scored
    loop = an.flexible_region(9, 11)
    assert loop.verdict == DEPENDENT and loop.risk == 2
    assert loop.answers == {"0": True, "1": True, "2": True, "3": False, "4": False}
    assert "would have told you no" in loop.plain_sentence()
    assert an.flexible_region(21, 23).verdict == CONSISTENT
    assert an.flexible_region(4, 7).verdict == NOT_SEEN
    assert an.folded_region(1, 7).verdict == CONSISTENT
    assert an.folded_region(14, 19).n_yes == 2
    assert an.stays_near_start().n_yes == 3


def test_default_rule_avoids_the_top_k_artefact(domain_path):
    rset = from_mdcath(domain_path, "320")
    designed = {9, 10, 11, 21, 22, 23}
    topk = analyse(rset, Method(rule=FlexRule(rule="topk", top=0.2)))
    topk_by = {rset.residues[i].number: int(a) for i, a in enumerate(topk.hot.agreement)}
    assert any(a >= 2 for r, a in topk_by.items() if r not in designed) or any(topk_by[r] < 5 for r in (21, 22, 23))
    mad = analyse(rset)
    mad_by = {rset.residues[i].number: int(a) for i, a in enumerate(mad.hot.agreement)}
    assert all(a <= 1 for r, a in mad_by.items() if r not in designed)


def test_region_in_unscored_ends_is_explained(domain_path):
    an = analyse(from_mdcath(domain_path, "320"))
    try:
        an.flexible_region(1, 3)
    except ValueError as err:
        assert "--trim-termini 0" in str(err)
    else:
        raise AssertionError("expected a helpful error")
