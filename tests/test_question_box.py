"""The report's question box: the browser logic gives the same answers as the Python engine.

The page's logic lives in concordance/assets/logic.js. These tests run that exact file in Node
on the data embedded in a real report, so they need Node.js (skipped when it is not installed).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from concordance.analysis import analyse
from concordance.cli import main
from concordance.report import ASSETS, hotspots_command, region_command, stretches
from concordance.sources import from_mdcath

from conftest import DOMAIN

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the report's JavaScript")

HARNESS = """
const Logic = require(process.argv[2]);
const input = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const logic = Logic.create(input.data);
const out = {
  regions: input.regions.map(([a, b]) => {
    const v = logic.verdictFor(logic.positions(a, b));
    return v && { yes: v.yes, verdict: v.verdict, flagged: v.flagged, risk: v.risk };
  }),
  parsed: input.questions.map((q) => {
    const p = logic.parse(q);
    return { intent: p.intent, lo: p.lo, hi: p.hi, run: p.run, runCount: p.runCount, top: p.top, still: p.still };
  }),
  answers: input.questions.map((q) => logic.ask(q, input.audience)),
  suggestions: logic.suggestions(),
};
process.stdout.write(JSON.stringify(out));
"""


def report_data(capsys, data_dir, out_dir, *extra) -> tuple[dict, str]:
    code = main(["--data-dir", str(data_dir), "--out-dir", str(out_dir), "report", "--domain", DOMAIN, *extra])
    printed = capsys.readouterr().out
    assert code == 0
    path = re.search(r"Visual report: (\S+\.html)", printed).group(1)
    page = Path(path).read_text(encoding="utf-8")
    match = re.search(r'<script type="application/json" id="concordance-data">(.*?)</script>', page, re.S)
    return json.loads(match.group(1)), page


def run_logic(tmp_path, data, questions=(), regions=(), audience="expert") -> dict:
    harness = tmp_path / "harness.js"
    harness.write_text(HARNESS)
    payload = tmp_path / "input.json"
    payload.write_text(json.dumps({"data": data, "questions": list(questions), "regions": list(regions), "audience": audience}))
    done = subprocess.run([NODE, str(harness), str(ASSETS / "logic.js"), str(payload)], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def test_stretches_and_commands():
    class Res:
        def __init__(self, number, chain="A"):
            self.number, self.chain = number, chain

    residues = [Res(1), Res(2), Res(3), Res(5), Res(6), Res(7, "B")]
    assert stretches([True, True, False, True, True, True], residues) == [[0, 1], [3, 4], [5, 5]]
    replay = "concordance report --domain 1abcA00 --temp 320 --rule mad --region 9-11 --min-fraction 0.5"
    assert hotspots_command(replay) == "concordance ask hotspots --domain 1abcA00 --temp 320 --rule mad"
    assert region_command(replay, 0.5).endswith("--rule mad --region {region} --min-fraction 0.5")


def test_report_embeds_the_logic(capsys, data_dir, tmp_path):
    data, page = report_data(capsys, data_dir, tmp_path)
    assert "var ConcordanceLogic" in page and "__CONCORDANCE_LOGIC__" not in page
    assert 'id="ask-input"' in page
    assert data["stretches"]["consistent"] and data["stretches"]["some"]
    assert data["hotspots_command"].startswith("concordance ask hotspots --domain " + DOMAIN)


@needs_node
def test_every_stretch_matches_the_python_engine(capsys, data_dir, domain_path, tmp_path):
    data, _ = report_data(capsys, data_dir, tmp_path)
    an = analyse(from_mdcath(domain_path, "320"))
    numbers = [r.number for r in an.rset.residues]
    regions = [(a, b) for i, a in enumerate(numbers) for b in numbers[i:]]
    got = run_logic(tmp_path, data, regions=regions)["regions"]
    compared = 0
    for (a, b), js in zip(regions, got):
        try:
            claim = an.flexible_region(a, b)
        except ValueError:  # only chain ends: the page says so too
            assert js is None
            continue
        assert js is not None
        assert js["yes"] == claim.n_yes and js["verdict"] == claim.verdict and js["risk"] == claim.risk
        assert js["flagged"] == {r: d["flexible_residues_in_region"] for r, d in claim.details.items()}
        compared += 1
    assert compared > 300


QUESTIONS = {
    "Which parts move a lot?": {"intent": "overview"},
    "Is 9-11 flexible?": {"intent": "region", "lo": 9, "hi": 11},
    "is residues 9 to 11 flexible": {"intent": "region", "lo": 9, "hi": 11},
    "Does GLY9 to ALA11 move?": {"intent": "region", "lo": 9, "hi": 11},
    "what about between 21 and 23": {"intent": "region", "lo": 21, "hi": 23},
    "residue 22": {"intent": "region", "lo": 22, "hi": 22},
    "Is 9–11 flexible in run 2?": {"intent": "run-region", "lo": 9, "hi": 11, "run": 2},
    "what did the third run say about 9-11": {"intent": "run-region", "run": 3},
    "run 4": {"intent": "run", "run": 4},
    "Where do the runs disagree?": {"intent": "disagree"},
    "What would one run have told me?": {"intent": "fewer-runs", "runCount": 1},
    "what if I only had 3 runs": {"intent": "fewer-runs", "runCount": 3},
    "What does replica-dependent mean?": {"intent": "define:dependent"},
    "what is RMSF": {"intent": "define:rmsf"},
    "what is flexible?": {"intent": "overview"},
    "Can I trust this?": {"intent": "trust"},
    "why five runs?": {"intent": "why-repeat"},
    "which residue moves the most": {"intent": "most"},
    "top 5 most flexible residues": {"intent": "most", "top": 5},
    "what is the most stable part": {"intent": "least"},
    "is the protein stable?": {"intent": "still"},
    "does 21-23 stay still": {"intent": "region", "still": True},
    "is it still flexible at 9-11": {"intent": "region", "still": False},
    "how many runs are there": {"intent": "count-runs"},
    "how many residues": {"intent": "count-residues"},
    "how was this calculated": {"intent": "method"},
    "hello": {"intent": "hello"},
    "thanks!": {"intent": "thanks"},
    "ok which parts move": {"intent": "overview"},
    "asdfgh": {"intent": "unknown"},
    "is 999 flexible": {"intent": "region", "lo": 999},
    "is residue 10 flexible in 4 of 5 runs": {"intent": "region", "lo": 10, "hi": 10},
    "flexibility of 9-11 at 320 K": {"intent": "region", "lo": 9, "hi": 11},
    "what does the orange mean": {"intent": "define:colours"},
    "what am I looking at?": {"intent": "define:page"},
    "": {"intent": "help"},
}


@needs_node
def test_questions_are_read_as_intended(capsys, data_dir, tmp_path):
    data, _ = report_data(capsys, data_dir, tmp_path)
    got = run_logic(tmp_path, data, questions=QUESTIONS)["parsed"]
    for (question, expected), parsed in zip(QUESTIONS.items(), got):
        for key, value in expected.items():
            assert parsed[key] == value, (question, key, parsed)


@needs_node
def test_answers_use_the_reports_numbers(capsys, data_dir, tmp_path):
    data, _ = report_data(capsys, data_dir, tmp_path)
    asks = ["Is 9-11 flexible?", "Which parts move a lot?", "Where do the runs disagree?", "is 999 flexible", "run 4", "asdfgh", "residue 1"]
    out = run_logic(tmp_path, data, questions=asks)
    region, overview, disagree, missing, run4, unknown, end = out["answers"]
    assert region["verdict"]["label"] == "Replica-dependent"
    assert region["lines"][0] == "3 of 5 runs say residues 9–11 move a lot."
    assert region["command"].endswith("--region 9-11 --min-fraction 0.5") and region["explore"]["lo"] == 8
    assert f"{data['counts']['consistent']} residues move a lot in at least 4 of 5 runs: 21–23" in overview["lines"][0]
    assert "9–11" in " ".join(disagree["lines"]) and disagree["command"] == data["hotspots_command"]
    assert missing["unknown"] is True and "numbered 1 to" in missing["lines"][0]
    summary = data["summaries"][data["runs"][3]["id"]]
    assert f"finds {summary['flexible_residues']} residues" in run4["lines"][0]
    assert unknown["unknown"] is True and unknown["lines"][0].startswith("Sorry, I didn't understand")
    assert end["verdict"] is None and "chain end" in end["lines"][0]
    assert any("9–11" in s for s in out["suggestions"])


@needs_node
def test_one_run_answer_uses_the_exact_agreement_curve(capsys, data_dir, tmp_path):
    data, _ = report_data(capsys, data_dir, tmp_path, "--region", "9-11")
    out = run_logic(tmp_path, data, questions=["what would one run have told me?"], audience="new")
    same = data["claim"]["agreement_curve"]["1"]["same"]
    assert f"{round(same * 100)}% of the time" in out["answers"][0]["lines"][0]


AUDIT_HARNESS = """
const Logic = require(process.argv[2]);
const input = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const logic = Logic.create(input.data);
const result = logic.engine.checkRegion(9, 11);
const good = "Residues 9-11 move a lot in " + result.runs_saying_it_moves_a_lot + " of " + result.total_runs + " runs (60%), so " + result.runs_that_alone_would_give_the_opposite_answer + " runs alone would say no.";
const bad = good + " Its mean RMSF is 0.913 nm.";
process.stdout.write(JSON.stringify({ result, good: logic.auditNumbers(good, [result], ""), bad: logic.auditNumbers(bad, [result], ""), pdb: logic.auditNumbers("See PDB 4AKE.", [], "") }));
"""


@needs_node
def test_claude_tools_and_number_check(capsys, data_dir, tmp_path):
    data, _ = report_data(capsys, data_dir, tmp_path)
    (tmp_path / "a.js").write_text(AUDIT_HARNESS)
    (tmp_path / "in.json").write_text(json.dumps({"data": data}))
    out = json.loads(subprocess.run([NODE, str(tmp_path / "a.js"), str(ASSETS / "logic.js"), str(tmp_path / "in.json")], capture_output=True, text=True, check=True).stdout)
    assert out["result"]["runs_saying_it_moves_a_lot"] == 3 and out["result"]["verdict"] == "replica-dependent"
    assert out["good"]["checked"] == 6 and out["good"]["unsupported"] == []
    assert out["bad"]["unsupported"] == ["0.913"]
    assert out["pdb"]["checked"] == 0  # identifiers like 4AKE are not numbers
