"""Shared fixtures: synthetic replicate data with a designed answer (see concordance/synthetic.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from concordance.synthetic import write_synthetic_domain, write_trajectory_files

DOMAIN = "1abcA00"


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    data = tmp_path_factory.mktemp("concordance") / "data"
    write_synthetic_domain(data / f"mdcath_dataset_{DOMAIN}.h5", domain=DOMAIN)
    return data


@pytest.fixture(scope="session")
def domain_path(data_dir) -> Path:
    return data_dir / f"mdcath_dataset_{DOMAIN}.h5"


@pytest.fixture(scope="session")
def xtc_files(domain_path, tmp_path_factory):
    return write_trajectory_files(domain_path, tmp_path_factory.mktemp("files"))
