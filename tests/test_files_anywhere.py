"""RMSF from any tool: gmx rmsf .xvg, cpptraj .dat (angstrom), CSV tables, and the no-install page.

The page builds its report in the browser with concordance/assets/logic.js. These tests run that
file in Node on the same files the Python engine reads, and check that they agree exactly.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from concordance.analysis import Method, analyse
from concordance.cli import main
from concordance.report import ASSETS
from concordance.sources import from_mdcath, from_xvg
from concordance.verdict import FlexRule
from concordance.xvg import parse_rmsf_text

NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is needed to run the page's JavaScript")


@pytest.fixture(scope="module")
def formats(domain_path, tmp_path_factory):
    """The synthetic runs written three ways: gmx rmsf .xvg (nm), cpptraj .dat (angstrom), one CSV table (nm)."""
    an = analyse(from_mdcath(domain_path, "320"))
    folder = tmp_path_factory.mktemp("formats")
    numbers = [r.number for r in an.rset.residues]
    xvg, dat = [], []
    for k, values in enumerate(an.rmsf_nm.values(), start=1):
        path = folder / f"run{k}.xvg"
        path.write_text('@    yaxis  label "(nm)"\n@TYPE xy\n' + "".join(f"{n:5d}   {v:8.4f}\n" for n, v in zip(numbers, values)))
        xvg.append(path)
        path = folder / f"run{k}.dat"
        path.write_text("#Res  AtomicFlx\n" + "".join(f"{n:8d} {v * 10:10.3f}\n" for n, v in zip(numbers, values)))
        dat.append(path)
    rows = ["resid,resname," + ",".join(f"rep{k}" for k in range(1, 6))]
    columns = list(an.rmsf_nm.values())
    for i, n in enumerate(numbers):
        rows.append(f"{n},ALA," + ",".join(f"{c[i]:.4f}" for c in columns))
    table = folder / "table.csv"
    table.write_text("\n".join(rows) + "\n")
    return {"xvg": xvg, "dat": dat, "csv": [table], "folder": folder}


def test_reader_handles_each_format():
    assert parse_rmsf_text('@ yaxis label "(nm)"\n1 0.1\n2 0.2\n').unit == "nm"
    assert parse_rmsf_text("#Res  AtomicFlx\n 1 1.5\n 2 2.5\n").unit == "A"
    table = parse_rmsf_text("resid;resname;a;b\n1;ALA;0.1;0.2\n2;GLY;0.3;0.4\n")
    assert [c[0] for c in table.columns] == ["a", "b"] and list(table.columns[1][1]) == [0.2, 0.4] and table.unit is None
    with pytest.raises(ValueError, match="no data rows"):
        parse_rmsf_text("hello world\nnot data\n")
    values_only = parse_rmsf_text("1.5 0.2\n2.5 0.3\n")  # no residue column: numbered in order
    assert values_only.numbered and list(values_only.numbers) == [1, 2] and len(values_only.columns) == 2


def test_same_verdicts_from_every_format(formats):
    flags = {}
    for kind in ("xvg", "dat", "csv"):
        rset = from_xvg(formats[kind])
        an = analyse(rset)
        flags[kind] = {r: list(f) for r, f in an.hot.flags.items()}
        assert rset.notes["units"] == {"xvg": "nm", "dat": "A", "csv": "nm"}[kind]
    assert flags["xvg"] == flags["dat"] == flags["csv"]
    assert analyse(from_xvg(formats["dat"])).rmsf_nm["1"].max() < 1  # angstrom converted to nm


def test_cli_reads_a_table_and_writes_the_page(capsys, formats, tmp_path):
    base = ["--out-dir", str(tmp_path)]
    assert main([*base, "ask", "flexible", "--rmsf", str(formats["csv"][0]), "--region", "9-11", "--json"]) == 0
    passport = json.loads(capsys.readouterr().out)
    assert passport["result"]["verdict"] == "replica-dependent" and "--rmsf" in passport["replay"]
    assert main([*base, "page"]) == 0
    page = (tmp_path / "concordance.html").read_text()
    assert '<script type="application/json" id="concordance-data">null</script>' in page
    assert '"run1_rmsf.xvg"' in page and 'id="dropzone"' in page and "__CONCORDANCE" not in page


HARNESS = """
const Logic = require(process.argv[2]);
const input = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const D = Logic.buildData(input.build);
const logic = Logic.create(D);
const grid = Logic.settingsGrid(input.build.rule).map((cell) => {
  const core = D.classes.map((c) => c !== "u");
  const flags = {};
  D.runs.forEach((run) => { flags[run.id] = Logic.flagResidues(D.rmsf[run.id], core, cell); });
  return { cell, region: logic.verdictFor(logic.positions(input.region[0], input.region[1]), flags).yes };
});
process.stdout.write(JSON.stringify({ D, grid, protein: logic.proteinSensitivity(), sens: logic.sensitivity(logic.positions(input.region[0], input.region[1])) }));
"""


def build_in_node(tmp_path, paths, region=(9, 11), units=None):
    files = [{"file": p.name, "text": p.read_text()} for p in paths]
    build = {"files": files, "structure": None, "name": "", "units": units, "version": "test",
             "rule": {"rule": "mad", "k": 3, "min_ratio": 1.5, "top_fraction": 0.15, "trim_termini": 3},
             "consistency": 0.8, "min_fraction": 0.5}
    (tmp_path / "h.js").write_text(HARNESS)
    (tmp_path / "in.json").write_text(json.dumps({"build": build, "region": list(region)}))
    done = subprocess.run([NODE, str(tmp_path / "h.js"), str(ASSETS / "logic.js"), str(tmp_path / "in.json")], capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def report_data(capsys, tmp_path, paths) -> dict:
    assert main(["--out-dir", str(tmp_path / "out"), "report", "--rmsf", *map(str, paths)]) == 0
    printed = capsys.readouterr().out
    page = Path(re.search(r"Visual report: (.+\.html)", printed).group(1)).read_text()
    return json.loads(re.search(r'<script type="application/json" id="concordance-data">(.*?)</script>', page, re.S).group(1))


@needs_node
@pytest.mark.parametrize("kind", ["xvg", "dat", "csv"])
def test_page_builds_the_same_report_as_python(capsys, formats, tmp_path, kind):
    py = report_data(capsys, tmp_path, formats[kind])
    js = build_in_node(tmp_path, formats[kind])["D"]
    for key in ("n", "need", "flags", "agreement", "classes", "counts", "stretches"):
        assert js[key] == py[key], key
    for run, summary in py["summaries"].items():
        for field, value in summary.items():
            assert js["summaries"][run][field] == pytest.approx(value, abs=1.5e-4), (run, field)  # display rounding at 4 dp
    assert [r["n"] for r in js["residues"]] == [r["n"] for r in py["residues"]]
    assert js["units"]["unit"] == py["units"]["unit"] and js["rule_text"] == py["rule_text"]
    for pair, rho in py["spearman"].items():
        assert js["spearman"][pair] == pytest.approx(rho, abs=1e-9)


@needs_node
def test_settings_check_matches_python(formats, tmp_path):
    out = build_in_node(tmp_path, formats["xvg"], region=(9, 11))
    rset = from_xvg(formats["xvg"])
    for row, protein in zip(out["grid"], out["protein"]["rows"]):
        cell = row["cell"]
        an = analyse(rset, Method(rule=FlexRule(rule="mad", k=cell["k"], min_ratio=cell["min_ratio"], trim_termini=3)))
        assert row["region"] == an.flexible_region(9, 11).n_yes, cell
        assert protein["consistent"] == an.hot.counts()["consistent"], cell
    assert out["sens"]["total"] == 9


@needs_node
def test_page_explains_bad_files(tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("hello world\nthis is not data\n")
    other = tmp_path / "short.xvg"
    other.write_text("1 0.1\n2 0.2\n")
    with pytest.raises(subprocess.CalledProcessError) as err:
        build_in_node(tmp_path, [bad, other])
    assert "notes.txt: no data rows found" in err.value.stderr


# Layouts people actually have: ATLAS-style tables of values only, Concordance's own CSV and .xvg
# (with mean/SD/runs_flagging columns that are not runs), GROMACS legends naming the columns.
LAYOUTS = {
    "atlas.tsv": "RMSF_R1\tRMSF_R2\tRMSF_R3\n0.81\t0.90\t0.70\n0.52\t0.50\t0.61\n0.33\t0.35\t0.31\n0.40\t0.41\t0.38\n",
    "concordance.csv": "chain,residue_number,residue,rmsf_nm_run1,rmsf_nm_run2,rmsf_nm_run3,mean_nm,sd_nm,runs_flagging\n"
                       "A,10,MET,0.1,0.2,0.3,0.2,0.1,0\nA,11,ALA,0.3,0.2,0.1,0.2,0.1,1\n",
    "legends.xvg": '@    yaxis  label "(nm)"\n@ s0 legend "run 1"\n@ s1 legend "run 2"\n@ s2 legend "mean"\n 5 0.1 0.2 0.15\n 6 0.3 0.2 0.25\n',
    "resid_not_first.csv": "rep1;rep2;resid\n0.11;0.12;40\n0.21;0.22;41\n",
}
EXPECTED = {
    "atlas.tsv": ([1, 2, 3, 4], ["RMSF_R1", "RMSF_R2", "RMSF_R3"], [], True),
    "concordance.csv": ([10, 11], ["rmsf_nm_run1", "rmsf_nm_run2", "rmsf_nm_run3"], ["mean_nm", "sd_nm", "runs_flagging"], False),
    "legends.xvg": ([5, 6], ["run 1", "run 2"], ["mean"], False),
    "resid_not_first.csv": ([40, 41], ["rep1", "rep2"], [], False),
}

PARSE_HARNESS = """
const Logic = require(process.argv[2]);
const input = JSON.parse(require("fs").readFileSync(process.argv[3], "utf8"));
const out = {};
for (const [name, text] of Object.entries(input)) {
  const t = Logic.parseRmsfText(text, name);
  out[name] = { numbers: t.numbers, labels: t.columns.map((c) => c.name), values: t.columns.map((c) => c.values), skipped: t.skipped, numbered: t.numbered, unit: t.unit };
}
process.stdout.write(JSON.stringify(out));
"""


def test_reader_handles_real_world_layouts():
    for name, text in LAYOUTS.items():
        table = parse_rmsf_text(text, name)
        numbers, labels, skipped, numbered = EXPECTED[name]
        assert list(table.numbers) == numbers and [c[0] for c in table.columns] == labels, name
        assert table.skipped == skipped and table.numbered == numbered, name


@needs_node
def test_browser_reader_matches_python_on_real_world_layouts(tmp_path):
    (tmp_path / "h.js").write_text(PARSE_HARNESS)
    (tmp_path / "in.json").write_text(json.dumps(LAYOUTS))
    done = subprocess.run([NODE, str(tmp_path / "h.js"), str(ASSETS / "logic.js"), str(tmp_path / "in.json")], capture_output=True, text=True, check=True)
    js = json.loads(done.stdout)
    for name, text in LAYOUTS.items():
        py = parse_rmsf_text(text, name)
        assert js[name]["numbers"] == [int(x) for x in py.numbers], name
        assert js[name]["labels"] == [c[0] for c in py.columns] and js[name]["skipped"] == py.skipped, name
        assert js[name]["values"] == [list(c[1]) for c in py.columns] and js[name]["numbered"] == py.numbered and js[name]["unit"] == py.unit, name


def test_values_only_table_takes_numbering_from_the_structure(tmp_path):
    from concordance.synthetic import backbone_pdb

    pdb = tmp_path / "p.pdb"
    lines = [line[:22] + f"{int(line[22:26]) + 100:4d}" + line[26:] if line.startswith("ATOM") else line for line in backbone_pdb(12).splitlines()]
    pdb.write_text("\n".join(lines) + "\n")  # residues numbered 101-112
    rows = ["RMSF_R1\tRMSF_R2\tRMSF_R3"] + [f"{0.1 + 0.01 * i:.3f}\t{0.11 + 0.01 * i:.3f}\t{0.12:.3f}" for i in range(12)]
    table = tmp_path / "atlas.tsv"
    table.write_text("\n".join(rows) + "\n")
    rset = from_xvg([table], pdb)
    assert [r.number for r in rset.residues] == list(range(101, 113)) and "as in the structure" in rset.notes["numbering"]
    assert all(r.name == "ALA" for r in rset.residues)


@needs_node
def test_browser_takes_numbering_from_the_structure_too(tmp_path):
    from concordance.synthetic import backbone_pdb

    lines = [line[:22] + f"{int(line[22:26]) + 100:4d}" + line[26:] if line.startswith("ATOM") else line for line in backbone_pdb(12).splitlines()]
    rows = ["RMSF_R1\tRMSF_R2\tRMSF_R3"] + [f"{0.1 + 0.01 * i:.3f}\t{0.11 + 0.01 * i:.3f}\t{0.12:.3f}" for i in range(12)]
    build = {"files": [{"file": "atlas.tsv", "text": "\n".join(rows) + "\n"}], "structure": {"file": "p.pdb", "text": "\n".join(lines) + "\n"},
             "name": "", "units": None, "version": "test", "rule": {"rule": "mad", "k": 3, "min_ratio": 1.5, "top_fraction": 0.15, "trim_termini": 3},
             "consistency": 0.8, "min_fraction": 0.5}
    script = 'const L = require(process.argv[1]); const D = L.buildData(JSON.parse(process.argv[2])); process.stdout.write(JSON.stringify({n: D.residues.map(r => r.n), note: D.numbering}));'
    done = subprocess.run([NODE, "-e", script, str(ASSETS / "logic.js"), json.dumps(build)], capture_output=True, text=True, check=True)
    out = json.loads(done.stdout)
    assert out["n"] == list(range(101, 113)) and "as in the structure" in out["note"]


def test_export_round_trips_to_the_same_answer(capsys, data_dir, tmp_path):
    from conftest import DOMAIN

    assert main(["--data-dir", str(data_dir), "--out-dir", str(tmp_path), "export", "--domain", DOMAIN]) == 0
    folder = tmp_path / "export" / f"{DOMAIN}_320K"
    runs = sorted(folder.glob("run*_rmsf.xvg"))
    assert len(runs) == 5 and (folder / "structure.pdb").exists()
    original = analyse(from_mdcath(data_dir / f"mdcath_dataset_{DOMAIN}.h5", "320"))
    again = analyse(from_xvg(runs, folder / "structure.pdb"))
    assert [r.number for r in again.rset.residues] == [r.number for r in original.rset.residues]
    assert list(again.hot.agreement) == list(original.hot.agreement)


def disordered_runs(folder, n_core=60, n_tail=90, seed=1):
    """Five runs of a protein whose long disordered tail is most of the chain, plus a loop (40-45) in runs 1-3."""
    import numpy as np

    rng = np.random.default_rng(seed)
    x = np.arange(1, n_core + n_tail + 1)
    paths = []
    for k in range(5):
        v = 0.07 * rng.lognormal(0, 0.15, len(x))
        if k < 3:
            v[39:45] += 0.25
        v[n_core:] = np.linspace(0.6, 2.8, n_tail) * rng.lognormal(0, 0.25, n_tail)
        path = folder / f"tail_run{k + 1}.xvg"
        path.write_text("".join(f"{i} {val:.4f}\n" for i, val in zip(x, v)))
        paths.append(path)
    return paths


def test_disordered_tail_is_detected_and_reference_recovers_the_loop(tmp_path):
    from concordance.analysis import disorder_hint

    paths = disordered_runs(tmp_path)
    plain = analyse(from_xvg(paths))
    hint = disorder_hint(plain.rmsf_nm, plain.core, [r.number for r in plain.rset.residues], False)
    assert hint and hint["share"] > 0.5 and hint["reference"] == [4, 60]
    assert plain.flexible_region(40, 45).n_yes == 0  # the failure mode: the baseline is the disordered tail
    fixed = analyse(from_xvg(paths), Method(reference=tuple(hint["reference"])))
    assert fixed.flexible_region(40, 45).n_yes == 3 and fixed.flexible_region(61, 150).n_yes == 5


@needs_node
def test_browser_reference_and_disorder_hint_match_python(tmp_path):
    paths = disordered_runs(tmp_path)
    for reference in (None, [4, 60]):
        build = {"files": [{"file": p.name, "text": p.read_text()} for p in paths], "structure": None, "name": "", "units": None, "version": "t",
                 "rule": {"rule": "mad", "k": 3, "min_ratio": 1.5, "top_fraction": 0.15, "trim_termini": 3}, "consistency": 0.8, "min_fraction": 0.5,
                 "reference": reference}
        script = 'const L = require(process.argv[1]); const D = L.buildData(JSON.parse(process.argv[2])); process.stdout.write(JSON.stringify({flags: D.flags, disorder: D.disorder, rule: D.rule_text}));'
        js = json.loads(subprocess.run([NODE, "-e", script, str(ASSETS / "logic.js"), json.dumps(build)], capture_output=True, text=True, check=True).stdout)
        py = analyse(from_xvg(paths), Method(reference=tuple(reference) if reference else None))
        assert {r: [int(x) for x in f] for r, f in py.hot.flags.items()} == js["flags"]
        if reference is None:
            assert js["disorder"] == {"share": round(js["disorder"]["share"], 3), "reference": [4, 60]}
        else:
            assert js["disorder"] is None and "residues 4-60" in js["rule"]
