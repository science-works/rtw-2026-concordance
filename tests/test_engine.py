"""The flexibility numbers: MDAnalysis engine, independent NumPy cross-check, file formats."""

import warnings

import numpy as np
import pytest

from concordance.analysis import Method, analyse, validate
from concordance.mdcath import find_domain_files, open_domain, read_dssp, read_stored
from concordance.rmsf import ca_rmsf, superimpose
from concordance.sources import from_files, from_mdcath
from concordance.synthetic import _random_rotation, backbone_pdb, _parse

from conftest import DOMAIN


def ca_base():
    names, _, _, coords = _parse(backbone_pdb(30))
    return coords[[i for i, n in enumerate(names) if n == "CA"]]


def test_reads_the_documented_mdcath_layout(domain_path, data_dir):
    info = open_domain(domain_path)
    assert info.domain_id == DOMAIN and info.n_residues == 30
    assert info.replicas["320"] == ["0", "1", "2", "3", "4"]
    assert read_dssp(info, "320")["0"].shape == (80, 30)
    assert read_stored(info, "320", "rmsf")["0"].shape == (30,)
    assert DOMAIN in find_domain_files(data_dir)
    with pytest.raises(ValueError, match="available: 320, 348"):
        info.replicas_at("450")


def test_superposition_removes_rigid_motion():
    rng = np.random.default_rng(0)
    base = ca_base()
    frames = np.stack([base @ _random_rotation(rng).T + rng.normal(scale=5.0, size=3) for _ in range(20)])
    aligned = superimpose(frames, base)
    np.testing.assert_allclose(aligned, np.broadcast_to(base - base.mean(axis=0), aligned.shape), atol=1e-8)
    assert ca_rmsf(frames).max() < 1e-6


def test_numpy_rmsf_recovers_designed_noise():
    rng = np.random.default_rng(2)
    base = ca_base()
    sigma = np.full(len(base), 0.2)
    sigma[[12, 13, 14]] = 2.0
    frames = np.stack([(base + rng.normal(size=base.shape) * sigma[:, None]) @ _random_rotation(rng).T for _ in range(600)])
    rmsf = ca_rmsf(frames)
    expected = np.sqrt(3.0) * 2.0
    assert np.all(np.abs(rmsf[[12, 13, 14]] - expected) / expected < 0.25)


def test_mdanalysis_and_numpy_agree(domain_path):
    an = analyse(from_mdcath(domain_path, "320"))
    assert an.check["agree"] and an.check["max_abs_diff_nm"] < 1e-5


def test_window_is_common_and_skips_equilibration(domain_path):
    an = analyse(from_mdcath(domain_path, "320"), Method(discard=0.25))
    assert an.window.start == 20 and an.window.stop == 80 and an.window.used == 60


def test_stored_check_matches_designed_values(domain_path):
    report = validate(from_mdcath(domain_path, "320"))
    assert all(row["pearson"] > 0.95 for row in report["stored_check"]["per_run"].values())


def test_xtc_files_give_the_same_answer(domain_path, xtc_files):
    topology, trajectories = xtc_files
    from_h5 = analyse(from_mdcath(domain_path, "320"))
    from_xtc = analyse(from_files(topology, trajectories))
    assert np.array_equal(from_h5.hot.agreement, from_xtc.hot.agreement)
    for a, b in zip(from_h5.rmsf_nm.values(), from_xtc.rmsf_nm.values()):
        assert np.abs(a - b).max() < 2e-3  # xtc stores coordinates to 0.001 nm
    assert from_xtc.rset.residues[0].label == "ALA1" and from_xtc.rset.residues[0].chain == "A"


def test_real_gromacs_files_match_the_documented_recipe():
    datafiles = pytest.importorskip("MDAnalysisTests.datafiles")
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe(datafiles.TPR, datafiles.XTC)
        avg = align.AverageStructure(u, u, select="protein and name CA", ref_frame=0).run()
        align.AlignTraj(u, avg.results.universe, select="protein and name CA", in_memory=True).run()
        reference = rms.RMSF(u.select_atoms("protein and name CA")).run().results.rmsf / 10
    an = analyse(from_files(datafiles.TPR, [datafiles.XTC, datafiles.XTC, datafiles.XTC]), Method(discard=0.0))
    assert np.abs(an.rmsf_nm["1"] - reference).max() < 1e-5
    assert set(an.hot.agreement.tolist()) <= {-1, 0, 3}  # identical runs can only agree completely


def test_chain_check_catches_a_protein_split_across_the_box():
    datafiles = pytest.importorskip("MDAnalysisTests.datafiles")
    broken = analyse(from_files(datafiles.TPR, [datafiles.XTC, datafiles.XTC]), Method(discard=0.0))
    assert broken.geometry["ok"] is False and broken.geometry["largest_A"] > 40 and "trjconv" in broken.geometry["advice"]
    whole = analyse(from_files(datafiles.TPR, [datafiles.XTC, datafiles.XTC], unwrap=True), Method(discard=0.0))
    assert whole.geometry["ok"] is True and whole.geometry["largest_A"] < 4.5


def test_synthetic_chain_stays_connected(domain_path):
    assert analyse(from_mdcath(domain_path, "320")).geometry["ok"] is True
