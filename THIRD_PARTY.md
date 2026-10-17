# Third-party software, assets and data

Concordance's own code is MIT-licensed (see `LICENSE`).

## Bundled in this repository (in `concordance/assets/`)

| Asset | Used for | Licence | Notice |
|---|---|---|---|
| 3Dmol.js 2.5.5 (`3Dmol-min.js`), University of Pittsburgh and contributors | 3D structure views in the offline report | BSD-3-Clause; incorporates code from GLmol, Three.js and jQuery | `concordance/assets/3Dmol-LICENSE.txt` |
| Atkinson Hyperlegible (Latin 400 and 700, WOFF2), Braille Institute of America | Typeface of the report, designed for legibility | SIL Open Font License 1.1 | `concordance/assets/AtkinsonHyperlegible-OFL.txt` |

## Installed as dependencies (not copied into this repository)

Licences as declared by each package (`pip show`, package metadata):

| Package | Used for | Licence |
|---|---|---|
| MDAnalysis | reading trajectories; the RMSF recipe | LGPLv3+ (code under LGPLv2.1+). MDAnalysis states it can be used by packages under any licence |
| NumPy | arrays; the independent cross-check | BSD-3-Clause (with bundled MIT, Zlib, 0BSD, CC0 components) |
| SciPy | correlations | BSD-3-Clause |
| h5py | reading mdCATH HDF5 files | BSD-3-Clause |
| mcp (Model Context Protocol Python SDK, below version 2) | the agent endpoint | MIT |
| huggingface_hub (optional, `data` extra) | downloading mdCATH | Apache-2.0 |
| pytest (optional, `dev` extra) | tests | MIT |

## Data (never committed)

**mdCATH**: Mirarchi, A., Giorgino, T. & De Fabritiis, G. *mdCATH: A Large-Scale MD Dataset for Data-Driven Computational Biophysics.* Scientific Data 11, 1299 (2024). Hosted at `huggingface.co/datasets/compsciencelab/mdCATH`.

The licence is stated inconsistently: the published paper states CC BY 4.0, while some Hugging Face dataset-card revisions state CC BY-ND 4.0. Concordance therefore never redistributes mdCATH data or derived structure files. `scripts/download_demo_data.py` fetches files at run time and records the dataset revision and a SHA-256 for each file in `data/manifest.json`.

## Test data

The optional tests that use real GROMACS files read them from the `MDAnalysisTests` package when it is installed. No file from that package is copied into this repository.

**Example structure on the no-install page:** adenylate kinase, PDB entry 4AKE (Müller, Schlauderer, Reinstein & Schulz, *Structure* 4, 147, 1996), backbone atoms only, taken from MDAnalysisTests' `adk_open.pdb`. wwPDB data are available under CC0. The RMSF values shown with it are synthetic.

**ATLAS** (Vander Meersche et al., *Nucleic Acids Research* 2024; CC BY-NC 4.0) is suggested as real test data in the README; none of it is included.
