"""Unknown formats, runs from different programs, and checking a paper's published RMSF.

Uses MDAnalysis's own test files (real simulations of adenylate kinase from CHARMM and GROMACS,
cobrotoxin saved as XTC, TRR and H5MD, an NMR ensemble); skipped when MDAnalysisTests isn't installed.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

datafiles = pytest.importorskip("MDAnalysisTests.datafiles")

from concordance.analysis import analyse  # noqa: E402
from concordance.cli import main  # noqa: E402
from concordance.formats import groups, pair, sniff  # noqa: E402
from concordance.reproduce import reproduce  # noqa: E402
from concordance.rmsf import superimpose  # noqa: E402
from concordance.sources import best_offset, from_runs, one_letter  # noqa: E402

D = datafiles


@pytest.fixture(scope="module")
def misnamed(tmp_path_factory):
    """Real files copied under misleading or missing extensions."""
    folder = tmp_path_factory.mktemp("mystery")
    names = {D.XTC: "run_a.dat", D.DCD: "run_b", D.TPR: "system.bin", D.PSF: "topology.txt", D.NCDF: "traj_c.out",
             D.PRMncdf: "params", D.PDB_multiframe: "ensemble.txt", D.GRO: "start.structure", D.TRR: "run_d.trajectory"}
    for src, dst in names.items():
        shutil.copy(src, folder / dst)
    return folder


EXPECTED = {"run_a.dat": ("trajectory", "XTC"), "run_b": ("trajectory", "DCD"), "system.bin": ("topology", "TPR"),
            "topology.txt": ("topology", "PSF"), "traj_c.out": ("trajectory", "NCDF"), "params": ("topology", "PARM7"),
            "ensemble.txt": ("ensemble", "PDB"), "start.structure": ("structure", "GRO"), "run_d.trajectory": ("trajectory", "TRR")}


def test_formats_are_read_from_contents_not_names(misnamed):
    for path in misnamed.iterdir():
        info = sniff(path)
        assert (info.kind, info.format) == EXPECTED[path.name], path.name
        assert info.by == "contents"
    assert sniff(misnamed / "ensemble.txt").frames == 24
    assert sniff(D.TRJ12_bz2).format == "TRJ" and sniff(D.TRJ12_bz2).compressed == "bzip2"
    assert sniff(D.H5MD_xvf).format == "H5MD" and sniff(D.H5MD_xvf).atoms == 19385


def test_trajectories_pair_with_topologies_and_group_by_protein(misnamed):
    infos = [sniff(p) for p in misnamed.iterdir()]
    runs, unmatched = pair(infos)
    pairs = {(t.path.split("/")[-1], r.path.split("/")[-1]) for t, r in runs}
    assert ("system.bin", "run_a.dat") in pairs and ("topology.txt", "run_b") in pairs and ("params", "traj_c.out") in pairs
    adk = [g for g in groups(infos) if len(g) == 3]
    assert adk, "GROMACS and CHARMM adenylate kinase should group as one protein"


def test_residue_names_from_different_force_fields():
    assert [one_letter(n) for n in ("HSD", "HIE", "HISA", "LYSH", "CYS2", "NMET", "CGLY", "ASPH", "XYZ")] == list("HHHKCMGDX")
    assert best_offset("ABCDEFGHIJKLMNOP", "DEFGHIJKLM")[:2] == (3, 1.0)


def test_runs_from_charmm_and_gromacs_combine(tmp_path):
    rset = from_runs([(D.PSF, D.DCD), (D.PSF, D.DCD2), (D.TPR, D.XTC)], name="adk", unwrap=True)
    assert rset.kind == "mixed" and rset.n_residues == 214
    an = analyse(rset)
    assert an.window.per_run == {"1": (9, 98), "2": (10, 102), "3": (1, 10)}
    assert an.check["agree"] and an.geometry.get("ok", True)  # unwrapped: the GROMACS run isn't split any more
    # both of adenylate kinase's mobile lids move a lot in all three runs (NMP 50-58, LID 127-131)
    assert an.flexible_region(50, 58).n_yes == 3 and an.flexible_region(127, 131).n_yes == 3


def test_same_simulation_in_three_formats_gives_the_same_answer():
    rset = from_runs([(D.TPR_xvf, D.XTC_sub_sol), (D.TPR_xvf, D.TRR_xvf), (D.TPR_xvf, D.H5MD_xvf)], name="cobrotoxin")
    assert any("same simulation saved twice" in note for note in rset.notes["matching"])
    from concordance.analysis import Method

    an = analyse(rset, Method(discard=0.0))
    values = list(an.rmsf_nm.values())
    assert max(np.abs(values[0] - v).max() for v in values[1:]) < 0.002  # xtc stores 0.001 nm


def test_an_ensemble_counts_as_a_run():
    rset = from_runs([(D.PDB_multiframe, None), (D.PDB_multiframe, None)], name="nmr")
    an = analyse(rset)
    assert an.window.per_run["1"] == (0, 24)  # no equilibration to skip in an ensemble


def _published(tmp_path, offset=None, shuffle=False):
    import MDAnalysis as mda

    cols = []
    for dcd in (D.DCD2, D.DCD):  # swapped order
        u = mda.Universe(D.PSF, dcd)
        ca = u.select_atoms("protein and name CA")
        x = np.array([ca.positions.copy() for _ in u.trajectory])
        aligned = superimpose(x, x[0])
        col = np.sqrt(((aligned - aligned.mean(0)) ** 2).sum(2).mean(0))  # angstrom
        if shuffle:
            col = np.random.default_rng(0).permutation(col)
        cols.append(col)
    if offset is None:
        text = "RMSF_R1\tRMSF_R2\n" + "".join(f"{a:.3f}\t{b:.3f}\n" for a, b in zip(*cols))
    else:
        text = "".join(f"{i + offset} {a:.3f} {b:.3f}\n" for i, (a, b) in enumerate(zip(*cols)))
    path = tmp_path / "published.tsv"
    path.write_text(text)
    return path


def test_reproduce_finds_units_numbering_column_order_and_recipe(tmp_path):
    rset = from_runs([(D.PSF, D.DCD), (D.PSF, D.DCD2)], name="adk")
    result = reproduce(rset, _published(tmp_path))
    best = next(r for r in result["recipes"] if r["recipe"] == result["best_recipe"])
    assert result["verdict"] == "reproduced" and best["fit"] == "first" and best["discard"] == 0.0
    assert best["units"]["published"] == "angstrom" and "rows taken in order" in best["numbering"]
    assert "not in the same order" in best["matching"]
    result = reproduce(rset, _published(tmp_path, offset=101))
    best = next(r for r in result["recipes"] if r["recipe"] == result["best_recipe"])
    assert result["verdict"] == "reproduced" and "offset by -100" in best["numbering"]


def test_reproduce_says_so_when_numbers_dont_follow(tmp_path):
    rset = from_runs([(D.PSF, D.DCD), (D.PSF, D.DCD2)], name="adk")
    assert reproduce(rset, _published(tmp_path, shuffle=True))["verdict"] == "not reproduced"


def test_identify_and_reproduce_commands(capsys, misnamed, tmp_path):
    assert main(["identify", str(misnamed)]) == 0
    out = capsys.readouterr().out
    assert "Runs found, grouped by protein" in out and "concordance report --run" in out
    published = _published(tmp_path)
    assert main(["--out-dir", str(tmp_path), "reproduce", "--run", D.PSF, D.DCD, "--run", D.PSF, D.DCD2, "--published", str(published)]) == 0
    assert "Reproduced" in capsys.readouterr().out
    assert (tmp_path / "reproductions").exists()


def test_reproduce_pairs_columns_when_the_run_counts_differ(tmp_path):
    rset = from_runs([(D.PSF, D.DCD), (D.PSF, D.DCD2), (D.TPR, D.XTC)], name="adk", unwrap=True)
    result = reproduce(rset, _published(tmp_path))  # two published columns, three runs
    best = next(r for r in result["recipes"] if r["recipe"] == result["best_recipe"])
    assert {c["ours"] for c in best["comparisons"]} == {"run 1", "run 2"}  # the CHARMM runs the table came from
    assert "run 3 matched no published column" in best["matching"]
    assert result["verdict"] == "largely reproduced" and "different numbers of runs" in result["caveats"][0]


def test_identify_takes_a_downloaded_zip_and_skips_tables_that_arent_rmsf(capsys, tmp_path):
    import zipfile

    folder = tmp_path / "entry"
    folder.mkdir()
    shutil.copy(D.PSF, folder / "protein.psf")
    shutil.copy(D.DCD, folder / "entry_R1.dcd")
    shutil.copy(D.DCD2, folder / "entry_R2.dcd")
    (folder / "entry_RMSF.tsv").write_text("RMSF_R1\tRMSF_R2\n" + "1.0\t1.1\n" * 214)
    (folder / "entry_pLDDT.tsv").write_text("pLDDT\n" + "90\n" * 214)
    archive = tmp_path / "entry_analysis.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in folder.iterdir():
            z.write(p, f"entry/{p.name}")
    assert main(["--out-dir", str(tmp_path / "out"), "identify", str(archive)]) == 0
    out = capsys.readouterr().out
    assert "Unzipped entry_analysis.zip" in out and "uv run concordance reproduce" in out
    assert "entry_RMSF.tsv" in out.split("Next:")[1] and "pLDDT" not in out.split("Next:")[1]
    assert main(["identify", str(tmp_path / "Entry_analysis")]) == 1
    assert "Can't find" in capsys.readouterr().err
