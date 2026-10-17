"""`concordance plot` (an open replacement for opening .xvg files in xmgrace) and `concordance check` (one command for a folder or zip)."""

from __future__ import annotations

import shutil
import zipfile

import pytest

from concordance.cli import main
from concordance.plot import build, parse

XVG = """# Created by:
#   :-) GROMACS - gmx energy (-:
@    title "GROMACS Energies"
@    xaxis  label "Time (ps)"
@    yaxis  label "(kJ/mol)"
@TYPE xy
@ s0 legend "Potential"
@ s1 legend "Kinetic En."
   0.000000  -500000.1  60000.2
   1.000000  -500010.5  60010.0
   2.000000  -499990.0  59990.5
"""


def test_reads_gromacs_xvg_titles_axes_and_legends():
    d = parse(XVG, "energy.xvg")
    assert d["title"] == "GROMACS Energies" and d["xlabel"] == "Time (ps)" and d["ylabel"] == "(kJ/mol)"
    assert [s["label"] for s in d["series"]] == ["Potential", "Kinetic En."]
    assert d["series"][0]["x"] == [0.0, 1.0, 2.0] and d["series"][1]["y"][2] == 59990.5


def test_tables_grace_sets_and_row_numbering():
    d = parse("time_ns\tgyr_R1\tgyr_R2\n0.0\t1.30\t1.31\n0.1\t1.29\t1.30\n", "g.tsv")
    assert d["x_from"] == "column" and [s["label"] for s in d["series"]] == ["gyr_R1", "gyr_R2"]
    d = parse("RMSF_R1\tRMSF_R2\n1.5\t1.4\n0.9\t1.0\n2.2\t2.0\n", "rmsf.tsv")  # values only: rows numbered, every column a series
    assert d["x_from"] == "row" and d["series"][0]["x"] == [1.0, 2.0, 3.0] and "residue" in d["xlabel"]
    d = parse("0 1\n1 2\n&\n0 3\n1 4\n", "sets.xvg")  # Grace sets separated by '&'
    assert len(d["series"]) == 2 and d["series"][1]["y"] == [3.0, 4.0]
    d = parse('@ xaxis label "Residue"\n' + r'@ yaxis label "\xD\f{}G (kJ mol\S-1\N)"' + "\n1 0.5\n2 0.7\n", "dg.xvg")
    assert d["ylabel"] == "ΔG (kJ mol^-1)"


def test_runs_are_shaded_together_but_different_quantities_are_not(tmp_path):
    for k in (1, 2, 3):
        (tmp_path / f"rmsd_run{k}.xvg").write_text("".join(f"{t} {0.1 * k + 0.01 * t}\n" for t in range(10)))
    data = build(sorted(tmp_path.glob("rmsd_run*.xvg")))
    assert data["same_x"] and data["runs_like"] and len(data["series"]) == 3 and data["series"][0]["label"] == "rmsd_run1"
    energy = tmp_path / "energy.xvg"
    energy.write_text(XVG)
    assert not build([energy])["runs_like"]


def test_plot_command_writes_one_offline_page(tmp_path, capsys):
    f = tmp_path / "energy.xvg"
    f.write_text(XVG)
    assert main(["--out-dir", str(tmp_path), "plot", str(f)]) == 0
    page = (tmp_path / "plots" / "energy.html").read_text()
    assert "GROMACS Energies" in page and "http" not in page.split("<script>")[1].split("</script>")[0].replace("http://www.w3.org/2000/svg", "")
    assert "2 series" in capsys.readouterr().out
    assert main(["plot", str(tmp_path / "missing.xvg")]) == 2


def test_check_runs_everything_on_a_downloaded_zip(tmp_path, capsys, monkeypatch):
    D = pytest.importorskip("MDAnalysisTests.datafiles")
    import MDAnalysis as mda
    import numpy as np

    from concordance.rmsf import superimpose

    folder = tmp_path / "entry"
    folder.mkdir()
    shutil.copy(D.PSF, folder / "entry.psf")
    cols = []
    for k, dcd in enumerate((D.DCD, D.DCD2), 1):
        shutil.copy(dcd, folder / f"entry_R{k}.dcd")
        u = mda.Universe(D.PSF, dcd)
        ca = u.select_atoms("protein and name CA")
        x = np.array([ca.positions.copy() for _ in u.trajectory])
        a = superimpose(x, x[0])
        cols.append(np.sqrt(((a - a.mean(0)) ** 2).sum(2).mean(0)))
    (folder / "entry_RMSF.tsv").write_text("RMSF_R1\tRMSF_R2\n" + "".join(f"{a:.3f}\t{b:.3f}\n" for a, b in zip(*cols)))
    (folder / "entry_gyrate.tsv").write_text("time_ns\tgyr_R1\tgyr_R2\n" + "".join(f"{t / 10}\t{1.3 + 0.01 * np.sin(t)}\t{1.31}\n" for t in range(50)))
    archive = tmp_path / "entry_analysis.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in folder.iterdir():
            z.write(p, p.name)
    assert main(["--out-dir", str(tmp_path / "out"), "check", str(archive)]) == 0
    out = capsys.readouterr().out
    assert "=== 1. What's here" in out and "=== 2. Do the 2 runs agree" in out and "=== 3. Do the published numbers" in out
    assert "Published RMSF (entry_RMSF.tsv): reproduced" in out and "Every step finished" in out
    assert (tmp_path / "out" / "plots" / "entry_gyrate.html").exists() and list((tmp_path / "out" / "reports").glob("*.html"))

    # the same in one call for an AI assistant, with nothing printed on stdout (which carries the MCP protocol)
    from concordance import mcp_server

    monkeypatch.setenv("CONCORDANCE_OUT", str(tmp_path / "agent"))
    capsys.readouterr()
    result = mcp_server.check_dataset(str(archive), name="entry")
    assert capsys.readouterr().out == ""
    assert result["agreement"]["runs"] == 2 and result["agreement"]["passport_id"] and result["published"][0]["verdict"] == "reproduced"


def test_website_is_current_and_carries_no_api_key():
    import re
    from pathlib import Path

    from concordance.report import render_page

    site = Path(__file__).resolve().parents[1] / "docs" / "index.html"
    page = site.read_text(encoding="utf-8")
    assert page == render_page(), "docs/index.html is out of date: run  uv run concordance page --out docs/index.html"
    assert not re.search(r"sk-ant-[A-Za-z0-9_\-]{20,}", page)  # the reader brings their own key; none is ever built in
    assert "anthropic-dangerous-direct-browser-access" in page and "sessionStorage" in page
