"""What users and agents receive: CLI text, passports, structures, reports, the agent endpoint."""

import asyncio
import json
import os
import shlex

import pytest

from concordance.cli import main
from concordance.structure import class_bounds, pdb_with_values
from concordance.synthetic import backbone_pdb

from conftest import DOMAIN


def run(capsys, data_dir, out_dir, *args):
    code = main(["--data-dir", str(data_dir), "--out-dir", str(out_dir), "--results-dir", str(out_dir / "results"), *args])
    return code, capsys.readouterr()


def test_ask_speaks_to_each_audience(capsys, data_dir, tmp_path):
    code, new = run(capsys, data_dir, tmp_path, "ask", "flexible", "--domain", DOMAIN, "--region", "9-11")
    assert code == 0 and "Replica-dependent: 3 of 5 runs say yes." in new.out and "Rule:" not in new.out
    code, expert = run(capsys, data_dir, tmp_path, "ask", "flexible", "--domain", DOMAIN, "--region", "9-11", "--audience", "expert")
    assert "Rule:" in expert.out and "Cross-check" in expert.out
    code, check = run(capsys, data_dir, tmp_path, "ask", "flexible", "--domain", DOMAIN, "--region", "9-11", "--audience", "check")
    assert "Replay: concordance ask flexible" in check.out and "sha256" in check.out
    assert "Synthetic test data" in new.out


def test_passport_is_deterministic_and_replayable(capsys, data_dir, tmp_path):
    code, first = run(capsys, data_dir, tmp_path, "ask", "flexible", "--domain", DOMAIN, "--region", "9-11", "--json")
    passport = json.loads(first.out)
    assert passport["result"]["verdict"] == "replica-dependent" and passport["result"]["single_run_risk"] == 2
    assert passport["source"]["synthetic_test_data"] is True and len(passport["source"]["files"][0]["sha256"]) == 64
    assert passport["engine_check"]["agree"] is True
    replay = shlex.split(passport["replay"])[1:] + ["--json"]
    code, again = run(capsys, data_dir, tmp_path, *replay)
    assert json.loads(again.out)["passport_id"] == passport["passport_id"]
    code, other = run(capsys, data_dir, tmp_path, "ask", "flexible", "--domain", DOMAIN, "--region", "9-11", "--mad-k", "2.5", "--json")
    assert json.loads(other.out)["passport_id"] != passport["passport_id"]


def test_files_mode_uses_gromacs_style_flags(capsys, data_dir, tmp_path, xtc_files):
    topology, trajectories = xtc_files
    code, out = run(capsys, data_dir, tmp_path, "ask", "flexible", "-s", str(topology), "-f", *map(str, trajectories), "--region", "9-11")
    assert code == 0 and "3 of 5 runs say yes" in out.out


def test_hotspots_report_and_structure_files(capsys, data_dir, tmp_path):
    code, out = run(capsys, data_dir, tmp_path, "report", "--domain", DOMAIN, "--region", "9-11", "--audience", "expert")
    assert code == 0
    html = next((tmp_path / "reports").glob("*.html")).read_text()
    assert '"synthetic":true' in html and "Atkinson Hyperlegible" in html and "3dmol" in html.lower()
    assert html.count("</script>") == 5  # data, example slot, viewer library, logic and page script, each closed exactly once
    assert len(html) < 2_000_000
    pdb = next((tmp_path / "structures").glob("*.pdb")).read_text()
    assert "  5.00" in pdb and " -1.00" in pdb
    assert "set_color" in next((tmp_path / "structures").glob("*.pml")).read_text()


def test_rmsf_validate_scan_and_selftest(capsys, data_dir, tmp_path):
    os.chdir(tmp_path)
    assert run(capsys, data_dir, tmp_path, "rmsf", "--domain", DOMAIN)[0] == 0
    code, out = run(capsys, data_dir, tmp_path, "validate", "--markdown", str(tmp_path / "VALIDATION.md"))
    assert code == 0 and "| yes |" in (tmp_path / "VALIDATION.md").read_text()
    code, out = run(capsys, data_dir, tmp_path, "scan")
    assert code == 0 and "Across 1 protein domains" in out.out
    code, out = run(capsys, data_dir, tmp_path, "selftest")
    assert code == 0 and "9 of 9 checks passed" in out.out


def test_errors_are_helpful(capsys, data_dir, tmp_path):
    code, out = run(capsys, data_dir, tmp_path, "ask", "flexible", "--domain", DOMAIN)
    assert code == 2 and "needs a region" in out.err
    code, out = run(capsys, data_dir, tmp_path, "ask", "hotspots", "--domain", "9zzzA00")
    assert code == 2 and "not in" in out.err
    code, out = run(capsys, data_dir, tmp_path, "ask", "hotspots", "-s", "missing.pdb")
    assert code == 2 and "both -s" in out.err


def test_pdb_b_factor_and_pymol_classes():
    text = pdb_with_values(backbone_pdb(2), {("A", 1, ""): 4.0})
    atoms = [line for line in text.splitlines() if line.startswith("ATOM")]
    assert atoms[0][60:66] == "  4.00" and atoms[-1][60:66] == " -1.00"
    assert class_bounds(5, 0.8) == {
        "not scored": (-1.5, -0.5), "never flagged": (-0.5, 0.5), "minority": (0.5, 1.5),
        "replica-dependent": (1.5, 3.5), "consistent": (3.5, 5.5),
    }


def test_pymol_colours_match_the_classes(tmp_path, capsys, data_dir):
    pymol = pytest.importorskip("pymol")
    from pymol import cmd

    run(capsys, data_dir, tmp_path, "ask", "hotspots", "--domain", DOMAIN)
    script = next((tmp_path / "structures").glob("*.pml"))
    cwd = os.getcwd()
    os.chdir(script.parent)
    try:
        cmd.reinitialize()
        cmd.do(f"@{script.name}")
        colours = {}
        cmd.iterate("name CA", "colours[int(resi)] = color", space={"colours": colours})
    finally:
        os.chdir(cwd)
    index = lambda name: cmd.get_color_index(name)
    assert colours[22] == index("conc_consistent") and colours[10] == index("conc_replica_dependent")
    assert colours[1] == index("conc_not_scored") and colours[5] == index("conc_never_flagged")


def test_agent_tools_return_numbers_with_a_passport(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("CONCORDANCE_DATA", str(data_dir))
    monkeypatch.setenv("CONCORDANCE_OUT", str(tmp_path))
    from concordance import mcp_server

    tools = {t.name for t in asyncio.run(mcp_server.mcp.list_tools())}
    assert {"list_proteins", "check_region_flexible", "find_flexible_residues", "make_report", "get_passport"} <= tools
    result = mcp_server.check_region_flexible(DOMAIN, 9, 11)
    assert result["verdict"] == "replica-dependent" and result["runs_saying_yes"] == 3
    assert mcp_server.get_passport(result["passport_id"])["result"]["single_run_risk"] == 2
    hot = mcp_server.find_flexible_residues(DOMAIN)
    assert hot["consistent_across_replicas"] == ["ALA21", "ALA22", "ALA23"]
    report = mcp_server.make_report(DOMAIN, 320, 9, 11)
    assert report["report_file"].endswith(".html")


def test_xvg_files_read_like_gmx_rmsf_output(capsys, data_dir, tmp_path):
    from MDAnalysis.auxiliary.XVG import uncomment

    run(capsys, data_dir, tmp_path, "rmsf", "--domain", DOMAIN)
    rmsf = next((tmp_path / "rmsf").glob("*_rmsf.xvg")).read_text().splitlines()
    assert any('@ s0 legend "run 1"' in l for l in rmsf) and any('@ s5 legend "mean"' in l for l in rmsf)
    rows = [l.split() for l in uncomment(rmsf) if l.strip()]
    assert len(rows) == 30 and all(len(r) == 7 for r in rows) and rows[0][0] == "1"
    agree = [l.split() for l in uncomment(next((tmp_path / "rmsf").glob("*_agreement.xvg")).read_text().splitlines()) if l.strip()]
    assert agree[21] == ["22", "5"] and agree[9] == ["10", "3"] and agree[0] == ["1", "-1"]
