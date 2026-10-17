"""Reading the .xvg files people already have from `gmx rmsf -res`, one per replicate run."""

import json
import shlex

import pytest

from concordance.analysis import analyse
from concordance.cli import main
from concordance.sources import from_mdcath, from_xvg
from concordance.synthetic import backbone_pdb
from concordance.verdict import DEPENDENT

from conftest import DOMAIN

GMX_HEADER = """# This file was created Sat Oct 17 09:00:00 2026
# Created by:
# gmx rmsf -s md.tpr -f run{run}.xtc -res -o rmsf{run}.xvg
@    title "RMS fluctuation"
@    xaxis  label "Residue"
@    yaxis  label "(nm)"
@TYPE xy
"""


def write_runs(folder, an, header=True, numbers=None):
    paths = []
    for i, values in enumerate(an.rmsf_nm.values(), start=1):
        rows = [f"{(numbers[k] if numbers else r.number):5d}   {v:.4f}" for k, (r, v) in enumerate(zip(an.rset.residues, values))]
        text = (GMX_HEADER.format(run=i) if header else "") + "\n".join(rows) + "\n"
        path = folder / f"run{i}" / "rmsf.xvg"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        paths.append(path)
    return paths


@pytest.fixture(scope="module")
def reference(domain_path):
    return analyse(from_mdcath(domain_path, "320"))


def test_gmx_rmsf_files_give_the_same_verdicts(tmp_path, reference):
    for header in (True, False):  # with GROMACS headers, and with -xvg none
        an = analyse(from_xvg(write_runs(tmp_path / str(header), reference, header)))
        assert an.hot.agreement.tolist() == reference.hot.agreement.tolist()
        claim = an.flexible_region(9, 11)
        assert claim.verdict == DEPENDENT and claim.risk == 2
        assert an.check["agree"] is None and an.rset.kind == "xvg"


def test_structure_gives_names_and_3d(tmp_path, reference):
    structure = tmp_path / "structure.pdb"
    structure.write_text(backbone_pdb(30))
    rset = from_xvg(write_runs(tmp_path, reference), structure)
    assert rset.residues[8].label == "ALA9" and "ATOM" in rset.pdb_text
    bare = from_xvg(write_runs(tmp_path / "bare", reference))
    assert bare.residues[8].label == "res9" and bare.pdb_text == ""


def test_mismatched_runs_and_atom_numbers_are_caught(tmp_path, reference):
    paths = write_runs(tmp_path, reference)
    paths[1].write_text("\n".join(paths[1].read_text().splitlines()[:-2]) + "\n")
    with pytest.raises(ValueError, match="different residues"):
        from_xvg(paths)
    atoms = write_runs(tmp_path / "atoms", reference, numbers=[12 * k + 5 for k in range(30)])
    assert from_xvg(atoms).notes["looks_like_atom_numbers"] is True


def test_cli_and_report_from_gmx_rmsf(tmp_path, capsys, data_dir, reference):
    paths = [str(p) for p in write_runs(tmp_path, reference)]
    base = ["--data-dir", str(data_dir), "--out-dir", str(tmp_path / "out")]
    assert main([*base, "ask", "flexible", "--xvg", *paths, "--region", "9-11", "--json"]) == 0
    passport = json.loads(capsys.readouterr().out)
    assert passport["result"]["verdict"] == "replica-dependent" and passport["source"]["kind"] == "xvg"
    assert len(passport["source"]["files"]) == 5
    assert main([*base, *shlex.split(passport["replay"])[1:], "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["passport_id"] == passport["passport_id"]
    assert main([*base, "report", "--xvg", *paths, "--region", "9-11"]) == 0
    html = next((tmp_path / "out" / "reports").glob("*.html")).read_text()
    assert '"kind":"xvg"' in html and '"has_structure":false' in html
    assert main([*base, "validate", "--xvg", *paths]) == 2


def test_agent_tool_reads_gmx_rmsf(tmp_path, monkeypatch, reference):
    monkeypatch.setenv("CONCORDANCE_OUT", str(tmp_path / "out"))
    from concordance import mcp_server

    result = mcp_server.check_region_flexible_in_gmx_rmsf([str(p) for p in write_runs(tmp_path, reference)], 9, 11)
    assert result["verdict"] == "replica-dependent" and "--rmsf" in result["replay"]
