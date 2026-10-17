# Concordance

**Does what your protein simulation tells you hold up when it is run again?**

Concordance adds replicate awareness to [MDAnalysis](https://www.mdanalysis.org), the open-source Python library for analysing molecular dynamics (MD) simulations. You ask a plain question, such as *"is this loop flexible?"*. Concordance answers it separately in every replicate run, tells you how many runs agree, shows what a single run on its own would have told you, and leaves a small receipt (an *evidence passport*) that anyone can replay.

Built at **Reinvent the Wheel** (Science Works × Inherent), London, 16–18 October 2026.

```bash
uvx --from git+https://github.com/tammy357/concordance concordance check ~/Downloads/your_data.zip --open
```

One command: tell you what each file is, compare the runs, check any published RMSF table, chart the rest, and open a report. No install, nothing to clone. See [Quick start](#quick-start) for more ways in, or try the [live website](#no-terminal-use-the-website) — no terminal needed.

### Contents

- [The wheel it reinvents](#the-wheel-it-reinvents) — the old workflow this replaces
- [The software it builds on](#the-software-it-builds-on) · [What was not working](#what-was-not-working) · [What Concordance adds](#what-concordance-adds)
- [Quick start](#quick-start) — every way in: a zip, mdCATH, your own GROMACS/AMBER/NAMD runs, files you already have, a paper's data, the website, Claude
- [How the answer is worked out](#how-the-answer-is-worked-out) — the rule, and why
- [A first result on real data](#a-first-result-on-real-data)
- [What has been tested](#what-has-been-tested) — 71 automated tests, plus real cold-download results
- [How this relates to existing tools](#how-this-relates-to-existing-tools) · [What it doesn't do](#what-it-doesnt-do) · [Limitations](#limitations)
- [Data, licences and credit](#data-licences-and-credit)

## The wheel it reinvents

Today, checking whether repeated simulations of a protein agree is a chain of old tools and hand work:

1. run `gmx rmsf` once per run, giving one `.xvg` file each;
2. open each one in **xmgrace**, the plotting program GROMACS writes `.xvg` files for. Grace's last release was in 2015 and it is built on X11 and Motif, so on a modern Mac it needs XQuartz just to open;
3. paste the columns into a spreadsheet and judge by eye which peaks line up;
4. and when the data came with a paper, first work out what a folder of oddly named files even is, and whether the published numbers actually follow from it. In practice this last step is rarely done.

Concordance replaces that chain with open, tested, modern pieces:

| Step | Before | With Concordance |
|---|---|---|
| What are these files? | Know which program wrote what, and trust the file names | `concordance identify`: about 20 formats recognised by their contents, even when misnamed or compressed; trajectories paired with topologies; runs grouped by protein |
| Look at an `.xvg` | xmgrace | `concordance plot`: any `.xvg`, `.csv` or `.tsv` as a modern chart that works offline, runs overlaid with their range shaded |
| Do the runs agree? | A spreadsheet and your eyes | `concordance report`: a verdict for every residue, with a replayable receipt |
| Do a paper's numbers follow from its data? | Rarely checked | `concordance reproduce` |
| All of it | – | `concordance check <folder or .zip>` |

It does not replace GROMACS or MDAnalysis: it builds on MDAnalysis and reads GROMACS's files. Everything is MIT-licensed, installs with one command, and its pages work offline.

## The software it builds on

A molecular dynamics simulation is a film of a protein moving: a computer follows every atom over time. One of the first things people measure is how much each residue wanders from its average position, the **root-mean-square fluctuation (RMSF)**. High RMSF marks the floppy parts of a protein, which matter for where drugs bind and how designed proteins behave.

Two everyday ways to get it:

- **MDAnalysis** reads trajectories from GROMACS, AMBER, NAMD, CHARMM, LAMMPS and more, and has a documented RMSF recipe (`AverageStructure`, then `AlignTraj`, then `RMSF`).
- **GROMACS** has the command `gmx rmsf`, which takes one trajectory (`-f`) and writes `.xvg` plot files meant for the Grace plotting program.

## What was not working

- **One trajectory at a time.** Both work on a single trajectory. Neither has a notion of *replicates*: running the same simulation several times and comparing the answers.
- **The guidelines ask for exactly that comparison.** Best-practice advice for simulations says to "consider multiple runs instead of a single simulation" and, "if multiple runs have been performed, compare results" ([Grossfield et al., *LiveCoMS* 2018](https://doi.org/10.33011/livecoms.1.1.5067)). The *Communications Biology* "Reliability and reproducibility checklist for molecular dynamics simulations" asks whether there are "at least 3 simulations per simulation condition with statistical analysis" ([2023](https://doi.org/10.1038/s42003-023-04653-0)). A systematic study found that conclusions drawn from single simulations "are often not reproducible" ([Knapp, Ospina & Deane, *JCTC* 2018](https://doi.org/10.1021/acs.jctc.8b00391)).
- **So the comparison happens by hand, if at all.** It is done in private notebooks, rarely reported, and hard for anyone else to check.
- **Nothing for AI agents to call.** Scientists increasingly work alongside AI assistants, and there was no endpoint that answers these questions with checkable numbers.

## What Concordance adds

- **The same MDAnalysis recipe on every replicate**, over the same window of frames, from mdCATH files or from your own trajectory files.
- **A verdict for each claim**:
  - *consistent across replicas* (at least 4 of 5 runs agree);
  - *replica-dependent* (2–3 of 5);
  - *not seen in most replicas* (0–1 of 5).
- **The single-run risk**: how many runs, analysed alone, would have given you the opposite answer.
- **What a smaller study would have seen**: for 1, 2, 3 or 4 runs, how often you would reach the same answer as all 5. It enumerates every subset, so the result is exact and repeatable.
- **An evidence passport** for every answer. It records file fingerprints (SHA-256), every analysis choice, the per-run numbers, software versions and the exact command to replay it. The same inputs always give the same passport id.
- **Checks on every answer**:
  - RMSF is computed twice, by MDAnalysis and by an independent NumPy implementation, and the passport records how closely they agree.
  - A chain-continuity check catches proteins split across the periodic box, a common reason RMSF goes wrong.
- **Outputs that open where people already work**:
  - a command line with the flags GROMACS users know (`-s`, `-f`);
  - a one-file visual report that works offline and adapts to the reader ("I'm new to simulations", "I run simulations", "I'm checking the work"). Type a question in your own words ("is 40–43 flexible?", "where do the runs disagree?", "what does replica-dependent mean?") or drag across the chart, and the five runs answer straight away, with the command that gives that answer a receipt;
  - a PDB file with agreement in the B-factor column, plus a PyMOL script;
  - CSV tables and JSON passports;
  - GROMACS-style `.xvg` files in the same layout as `gmx rmsf -res`, but with one column per run plus the mean, so they drop into existing plotting scripts;
  - tools for AI agents over the Model Context Protocol (MCP).

## Quick start

You need [uv](https://docs.astral.sh/uv/) (or pip) and Python 3.10 or newer. Tested on Python 3.13 with MDAnalysis 2.10.

Jump to what you have: [a zip or folder](#already-have-per-residue-rmsf-from-any-tool-use-the-files-you-have) (one command, below) · [mdCATH](#on-the-mdcath-dataset) · [your own GROMACS/AMBER/NAMD runs](#on-your-own-gromacs-or-amber-namd--runs) · [RMSF files you already have](#already-have-per-residue-rmsf-from-any-tool-use-the-files-you-have) · [a paper's unlabelled files](#files-from-a-paper-in-formats-you-dont-recognise) · [no terminal](#no-terminal-use-the-website) · [ask in plain English](#ask-in-plain-english-claude-code)

**Got a folder or a .zip of simulation files?** One command, nothing to clone:

```bash
uvx --from git+https://github.com/tammy357/concordance concordance check ~/Downloads/your_data.zip --open
```

It says what each file is, compares the runs, checks any published RMSF table against the trajectories, charts the other tables, and opens the report.

```bash
uvx --from git+https://github.com/tammy357/concordance concordance selftest   # try it without cloning

git clone https://github.com/tammy357/concordance && cd concordance
uv run concordance selftest      # checks the tool on synthetic data with a known answer (no download)
```

### On the mdCATH dataset

[mdCATH](https://huggingface.co/datasets/compsciencelab/mdCATH) has 5,398 protein domains, each simulated five times at five temperatures (320–450 K). Concordance downloads a small subset chosen by a rule fixed in advance (by default the N smallest files), so the examples are not hand-picked.

```bash
uv run --extra data python scripts/download_demo_data.py --n 6 --dry-run   # see what would download
uv run --extra data python scripts/download_demo_data.py --n 6

uv run concordance domains                                   # what you have
uv run concordance ask hotspots --domain <id>                # which residues move, and do the runs agree?
uv run concordance ask flexible --domain <id> --region 40-55 # is this region flexible?
uv run concordance report --domain <id> --region 40-55 --open
uv run concordance rmsf --domain <id>                        # every run's RMSF as CSV and GROMACS-style .xvg
uv run concordance validate                                  # cross-checks, written to VALIDATION.md
uv run concordance scan                                      # agreement over every downloaded domain
```

### On your own GROMACS (or AMBER, NAMD, ...) runs

Give one topology and one trajectory file per replicate run:

```bash
uv run concordance ask flexible -s topol.tpr -f run1.xtc run2.xtc run3.xtc --region 40-55 --unwrap
uv run concordance report -s topol.tpr -f run1.xtc run2.xtc run3.xtc --unwrap --open
```

### Already have per-residue RMSF from any tool? Use the files you have

If you have per-residue RMSF for each run, Concordance compares them directly. Nothing is re-simulated or recomputed. It reads:
- GROMACS `gmx rmsf -res` `.xvg` files (nm), one per run;
- AMBER `cpptraj atomicfluct ... byres` `.dat` files (ångström, detected from the header);
- CSV, TSV or plain-text tables: one file per run, or one table with a column per run (for example exported from VMD, a notebook or a spreadsheet). A header row and residue-name columns are fine; summary columns named mean, SD or runs_flagging are left out; a table of values only (like the ATLAS database's RMSF tables) is numbered in order, or as in the structure if you give one.

```bash
uv run concordance ask flexible --rmsf run1/rmsf.xvg run2/rmsf.xvg run3/rmsf.xvg --region 40-55
uv run concordance report --rmsf my_runs.csv -s structure.pdb --open       # -s is optional, for the 3D views
uv run concordance report --rmsf run*.dat --units A --open                 # say the units if the files don't
```

### Files from a paper, in formats you don't recognise

Data shared with a paper often arrives as a folder of files with unhelpful names, from more than one program. Concordance reads each file's contents rather than trusting its name, works out which trajectory belongs with which topology, and groups the runs by protein:

```bash
uv run concordance identify ~/Downloads/paper_data            # what is each file, and which ones are runs?
```

It recognises GROMACS (`.tpr`, `.gro`, `.xtc`, `.trr`), CHARMM/NAMD/OpenMM (`.psf`, `.dcd`, `.coor`), AMBER (`.prmtop`, `.nc`, `.mdcrd`, `.inpcrd`), H5MD and mdCATH HDF5, PDB/mmCIF (including NMR-style multi-model ensembles), LAMMPS, XYZ and per-residue RMSF tables, also when gzip- or bzip2-compressed. It then prints the commands to run.

**Runs from different programs, analysed together.** Give each run as its own topology and trajectory pair; Concordance matches the runs residue by residue by their sequence (at least 90% identity, residue names normalised across force fields, so `HSD`, `HIE` and `HISA` are all histidine), compares them on the residues they share, and warns if two "runs" are the same simulation saved twice:

```bash
uv run concordance report --run charmm.psf run1.dcd --run charmm.psf run2.dcd --run topol.tpr run3.xtc --unwrap --open
```

**Check a paper's published RMSF against its own trajectories.** Point `reproduce` at the published per-residue RMSF table. It recomputes RMSF with four common recipes (fit to the average or the first frame; skip 10% or no frames), works out the table's units, residue numbering offset and which column is which run, and says whether the published numbers follow from the trajectories, and whether the same residues come out as flexible:

```bash
uv run concordance reproduce --run topol.tpr run1.xtc --run topol.tpr run2.xtc --published paper_rmsf.tsv
```

"Reproduced" means every compared profile has Pearson r ≥ 0.95 and a median difference within 15% of the published value; "largely reproduced" means every r ≥ 0.85, or that the table has a different number of runs from the trajectories (each published column is then paired with its best-matching run, and you're told which were left over). These are stated conventions, written into the result file (`outputs/reproductions/`), with SHA-256 checksums of every input.

To try all this without data of your own: `uv run --with MDAnalysisTests python scripts/make_mystery_folder.py` copies MDAnalysis's real test simulations into `outputs/mystery/` under unhelpful names, with a made-up stand-in for a published table, then `uv run concordance identify outputs/mystery` takes it from there.

### Real data to try

- **Your mdCATH downloads:** `uv run concordance export --domain 1e8rA00` writes one `gmx rmsf`-style `.xvg` per run plus `structure.pdb` into `outputs/export/`, ready to drop into the page.
- **The ATLAS database** ([dsimb.inserm.fr/ATLAS](https://www.dsimb.inserm.fr/ATLAS)), an independent set of GROMACS simulations with three replicates per protein (CC BY-NC 4.0). Download a protein's analysis archive (it includes three 1,000-frame protein-only trajectories), for example `https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/6in7_A`, and run `uv run concordance identify` on the downloaded zip or the unzipped folder; it prints the `report` and `reproduce` commands for those files. `reproduce` then checks ATLAS's own published RMSF table against its three trajectories. For a very flexible example, try ATLAS entry 6in7_A (MucA, 81 residues). Concordance does not redistribute ATLAS data.

### No Terminal? Use the website

The same page is published with GitHub Pages at **https://tammy357.github.io/concordance/** (built from `docs/index.html`; rebuild it with `uv run concordance page --out docs/index.html`). Drop in RMSF files and a structure, or try the example. Everything runs in your browser and nothing is uploaded.

**Ask Claude on the website.** Outside Claude, the Ask Claude button asks for your own Anthropic API key. The key is kept in your browser tab (in memory, or until the tab closes if you tick the box), is sent only to `api.anthropic.com`, and is never written into the page or this repository. Claude receives your question and the engine's answers, not your files. Use a key with a spending limit, and delete it when you're done. Claude Max and Team plans include monthly API credits that can be claimed in claude.ai settings ([details](https://platform.claude.com/docs/en/about-claude/api-credits-for-subscribers)). Inside Claude (as a published artifact) no key is needed.

### No Terminal? Use the page

`uv run concordance page` writes one HTML file (`outputs/concordance.html`) that anyone can open in a browser. Drop in the same files (and optionally a `.pdb` for 3D) and it builds the full report, question box included, in the browser. Nothing is uploaded. It applies the same rules as the Python engine; a test runs the page's code in Node on `.xvg`, `.dat` and `.csv` inputs and checks it gives the same flags, counts and stretches as Python.

### Claude in the question box

When the page is opened as a published Claude artifact, an **Ask Claude** button appears next to Ask. Claude gets the engine as tools (check a stretch, look at the whole protein, a run on its own, rank residues, re-check under other thresholds) and never computes a number itself. The page shows each tool call as it happens, drives the five films and the 3D from it, and then checks every number in Claude's answer against what the engine returned: numbers that match are underlined, and any that don't are flagged. Claude sees the question and the engine's answers, not your files. Opened as a plain file, the page works the same without this button. A test checks the tools and the number check in Node.

### Disordered proteins

Each run's "moves a lot" is judged against its own typical residue. When most of the chain is disordered (a long floppy tail, say), that typical residue is itself disordered, and real flexible regions in the folded part can be missed. Concordance spots this (when more than 40% of residues move over three times as much as the stillest tenth), says so, and suggests measuring against the still part instead: `--reference 4-60` on the command line, or one click on the page. A test builds a protein whose disordered tail is 60% of the chain and checks that the warning appears and that the reference recovers a loop the plain rule misses.

### Comparing runs and amino acids

The page shows how often each pair of runs agrees (pick a pair to see both on the chart and the stretches where they differ), where each run's numbers came from, the strip under the chart coloured by amino-acid type, and the average movement of each amino-acid type. The chart switches to a log scale when a few residues move far more than the rest.

### Doesn't the answer depend on your thresholds?

Ask the report ("does this depend on the threshold?"). It re-runs the verdict under nine reasonable settings of the flexibility rule (k = 2, 3, 4 × ratio 1.25, 1.5, 2) and says how many give the same answer, and which residues count as consistent under every setting. A test checks these against the Python engine.

`--xvg` still works as another name for `--rmsf`. Files with GROMACS's `#` and `@` header lines and bare `-xvg none` files both work. Because Concordance didn't compute these numbers, the passport says so, and the MDAnalysis cross-check doesn't apply: frames, fitting and atom selection are whatever you chose in GROMACS. If the first column looks like atom numbers rather than residue numbers, Concordance warns you to rerun with `-res`.

`--unwrap` makes the protein whole in every frame, using MDAnalysis and the bonds in the topology. Without it, Concordance still warns you if neighbouring residues are implausibly far apart.

### Ask in plain English (Claude Code)

Open this folder in Claude Code (`claude`) and approve the `concordance` server when it asks; it is set up in `.mcp.json`. Then just ask, for example:

> Is the loop at residues 40 to 43 of 1e8rA00 flexible? Can I trust it?

Claude calls Concordance's tools, answers in plain language with the verdict and what a single run would have told you, and quotes the passport id so the answer can be checked. `CLAUDE.md` tells Claude how to use the tools and not to invent numbers. To add the server to Claude Code from anywhere instead:

```bash
claude mcp add concordance -- uv run --quiet --directory /path/to/concordance concordance-mcp
```

The agent then has eleven tools:
- `check_dataset` (one call for a folder or .zip: what's here, do the runs agree, do the published numbers follow?);
- `list_proteins`;
- `check_region_flexible`;
- `check_region_flexible_in_files`;
- `find_flexible_residues`;
- `make_report`;
- `check_region_flexible_in_gmx_rmsf`;
- `get_passport`;
- `identify_files` (what is each file in a folder, and which are runs);
- `check_region_flexible_in_runs` (runs from different programs together);
- `reproduce_published_rmsf` (does a paper's RMSF table follow from its trajectories?).

Every answer includes a passport id, so whatever the agent says can be checked against the passport and replayed.

## How the answer is worked out

Defaults below; every one is a command-line option and is written into each passport.

| Step | Default |
|---|---|
| Atoms | C-alpha |
| Frames | All runs cut to the shortest run; the first 10% skipped as equilibration (`--discard`) |
| RMSF | MDAnalysis: frames fitted to each run's own average structure, then RMSF (Welford's algorithm, as in MDAnalysis) |
| Flexible, within one run | RMSF more than 3 median absolute deviations (MAD) above that run's median, **and** at least 1.5 times the median (`--mad-k`, `--min-ratio`); the first and last 3 residues are not scored (`--trim-termini`) |
| Consistent | Flexible in at least 80% of runs, i.e. 4 of 5 (`--consistency`) |
| Region flexible, within one run | At least half of its scored residues are flexible (`--min-fraction`) |

**Where the "3 MAD" rule comes from.** Measuring distance from the median in units of the median absolute deviation is the standard robust way to call values unusually high, because a few very floppy residues can't drag the threshold up the way they drag a mean and standard deviation (Leys et al., *Journal of Experimental Social Psychology* 49, 764–766, 2013, who recommend 2.5 as a default for outliers). To be exact about it: Concordance uses the plain MAD, not the MAD scaled by 1.4826 that Leys et al. use, so 3 plain MADs is about 2 scaled MADs: a looser cut, which suits "moves a lot" rather than "is an outlier". The floor of 1.5 times the median keeps a very still protein from having tiny wobbles called flexible. Because any cut is a choice, every answer can be re-checked under nine settings (k = 2, 3, 4 × ratio 1.25, 1.5, 2), and the passport records the rule used.

**Why not simply "the top 15% of residues in each run"?** That rule forces every run to flag the same number of residues. On test data with a known answer, it invented disagreement: residues that are rigid in every run were called replica-dependent, and residues that are flexible in every run were demoted. The default rule recovers the designed answer. `concordance selftest` shows both, and `--rule topk` is still available for comparison.

## A first result on real data

On the six smallest domains in mdCATH (chosen by file size, a rule fixed before looking at any results), at 320 K:

- **57 residues** moved a lot in at least one run, but **only 16** did so in at least 4 of the 5 runs.
- **2 of the 6 domains** have a stretch of 3 or more residues in a row that only some runs flag.

In other words, the list of "flexible residues" you would report from a single run changes a lot from run to run. These are small domains (50–56 residues), so this is a first look, not a survey. The full numbers are in `results/scan_320K.json`.

## What has been tested

There are 71 automated tests (`pytest`) and a self-test anyone can run (`concordance selftest`).

### On real, cold-downloaded data

- **On a real database entry, from a cold download.** ATLAS entry 6in7_A (MucA, 81 residues): `identify` went through all 16 files in the downloaded zip, paired the three protein-only trajectories with the structure (and not with the whole-system `.tpr` files, which have 19,671 atoms), picked the RMSF table out of about ten tables, and left out the ones that aren't per-residue values.
- **`plot` reads real GROMACS `.xvg` headers** (titles, axis labels, legends, Grace escape codes such as Δ and superscripts, and multiple data sets), tables with a time column, and tables of values only; runs are shaded as a group, but different quantities (say, potential energy and temperature) get a chart each instead of sharing one axis.
- **Ask Claude with your own key** was tested in Chromium against a stand-in for the Claude API: the key stays out of storage unless asked, the request carries the engine as tools, the tool results go back to Claude, a made-up number in the answer is flagged, and a rejected key brings the key form back. A test checks the published page carries no key.
- **`check` runs the whole chain on a downloaded zip** in a test: identify, report, reproduce, and charts of the other tables.
- **Files are identified by their contents.** Nine real MDAnalysis test files copied under misleading names (an `.xtc` saved as `run_a.dat`, a `.tpr` as `system.bin`, a `.psf` as `topology.txt`, and so on) are all identified correctly, paired with the right topology, and the CHARMM and GROMACS adenylate kinase runs are grouped as one protein.
- **Runs from different programs combine.** Two CHARMM runs (`.psf` + `.dcd`) and one GROMACS run (`.tpr` + `.xtc`) of adenylate kinase are matched by sequence (214 residues) and analysed together; both of the enzyme's known mobile lids (NMP, residues 50–58, and LID, 127–131) move a lot in all three runs.
- **The same simulation in three formats gives the same answer.** Cobrotoxin saved as `.xtc`, `.trr` and H5MD gives the same RMSF to within 0.002 nm (`.xtc` stores 0.001 nm), and Concordance warns that these are one simulation saved several times, not independent runs.
- **Published tables are lined up and checked.** Given a published-style RMSF table in ångström, with no residue column or with numbering offset by 100, and with its columns in the opposite order to the files, `reproduce` finds the units, the offset, the column order and the recipe that made it, and calls it reproduced. The same table with its rows shuffled is called not reproduced.

### Correctness checks

- **It reproduces the dataset's own numbers.** On the same six real mdCATH domains, Concordance's RMSF matches the values stored in mdCATH for every run (Pearson r 0.970 to 1.000), and MDAnalysis and NumPy agree to within 10⁻⁷ nm. See [`VALIDATION.md`](VALIDATION.md).
- **Two implementations agree.** On the test data, MDAnalysis and the independent NumPy implementation agree to within 10⁻⁷ nm, and every passport records the difference for its own data.
- **It matches MDAnalysis's own recipe.** On MDAnalysis's GROMACS test files (adenylate kinase, `adk_oplsaa.tpr` and `.xtc`, 47,681 atoms), Concordance's numbers equal MDAnalysis's documented recipe run on the full system (largest difference 3 × 10⁻⁷ nm). The same holds for a `.gro` topology.
- **Identical runs agree completely.** Three copies of the same run give zero single-run risk.
- **Designed answers are recovered.** Residues built to move in all 5 runs are found in 5; residues built to move in 3 runs are found in exactly 3. A helix present in 2 runs is called replica-dependent.
- **File formats don't change the answer.** The same runs written as GROMACS `.xtc` files give the same verdicts as the mdCATH files, to within the 0.001 nm precision `.xtc` stores.
- **Split proteins are caught.** In MDAnalysis's GROMACS test trajectory the protein is split across the 80 Å periodic box: two neighbouring residues sit 79.5 Å apart. Concordance flags this, and `--unwrap` repairs it (largest neighbour distance 3.9 Å).
- **Passports replay exactly.** Replaying a passport's command reproduces the same passport id; changing any parameter changes it.

### Feature coverage

- **The PyMOL script colours correctly.** Run in PyMOL 3.2, it colours each residue by its class.
- **The report's live answers match the engine.** Dragging across the chart, or typing a question, recomputes the verdict in the browser from the embedded numbers. A test runs the page's own JavaScript (`concordance/assets/logic.js`) in Node and checks it against the Python engine on every possible stretch of the test protein (all 465), and the command the page offers reproduces the same verdict at the command line.
- **The question box is simple on purpose.** It is not a language model: it recognises a handful of question types with word patterns (36 test questions, from "is 9–11 flexible in run 2?" to "what does the orange mean?"), shows how it read each one, answers only from the page's numbers, and says so when it doesn't understand. Nothing is sent anywhere, so it works offline. For open-ended questions, use Claude with the Concordance tools (below).
- **Claude answers plain-English questions through the tools.** Asked "Is the loop at residues 36 to 52 flexible? Can I trust it?", Claude Code called Concordance, gave the verdict in plain language and quoted the passport id of the answer it had just written.
- **The agent endpoint works.** The MCP server responds to an MCP client over stdio, and Claude Code's `claude mcp list` reports it as connected.
- **Existing `gmx rmsf` output gives the same verdicts.** Five `.xvg` files, with GROMACS headers or as bare `-xvg none` columns, give exactly the same per-residue agreement as computing from the trajectories. Mismatched runs are rejected, and atom-number columns trigger a warning.
- **The `.xvg` files parse.** MDAnalysis's own `.xvg` reader reads them correctly (GROMACS and Grace themselves were not available to test with).
- **The download script follows its rule.** Tested against a stand-in for Hugging Face: it applies the selection rule and writes a checksummed manifest.

[`VALIDATION.md`](VALIDATION.md) holds the numbers from `concordance validate`. On mdCATH data, that includes a comparison with the RMSF values stored in the dataset itself.

## How this relates to existing tools

Comparing replicate simulations is established good practice, and several tools already help:

- [mdaencore](https://mdaencore.readthedocs.io/en/latest/) (an MDAnalysis toolkit) scores how similar whole ensembles are and whether they have converged.
- [PENSA](https://pensa.readthedocs.io/en/stable/tut-4-comparison.html) compares two ensembles feature by feature.
- [MD DaVis](https://md-davis.readthedocs.io/en/latest/guides/xvg.html) plots several GROMACS outputs together.
- The ATLAS database shows its replicate simulations side by side.
- `gmx rmsf -oq` can already write RMSF into a PDB's B-factors.
- MD agents such as [MDCrow](https://github.com/ur-whitelab/MDCrow), and MCP servers for VMD and GROMACS, let AI assistants run simulations and compute RMSF.

What Concordance adds is narrower. It works at the level of the claim you want to make: a per-residue, plain-language verdict across replicates, a "would one run have misled you?" count, a replayable passport, and the same questions available to people and to AI agents, all on top of MDAnalysis. We did not find an MDAnalysis toolkit that does this, but our search covered the main tools rather than every repository.

## What it doesn't do

Concordance does one job carefully rather than many loosely.

- **It measures one property: how much each residue moves (C-alpha RMSF).** Not contacts, secondary structure, free energies or binding (the `folded` and `stable` questions only use values mdCATH already stores).
- **It doesn't run simulations, or say whether a simulation is physically right.** It says whether repeated runs agree, and whether published numbers follow from shared data.
- **`plot` draws what is in the file;** it doesn't recompute or reinterpret it.
- **`reproduce` checks one table, not a whole paper.**

## Limitations

- **Agreement is not correctness.** Concordance measures agreement between replicate runs, not whether the force field or the physics is right. Runs that start from the same structure can agree and still all be wrong. Disagreement, though, is a clear warning.
- **Five runs is a small sample.** The thresholds are explicit conventions, not statistical guarantees. Knapp & Deane suggest that around 10–20 replicas are needed to sample reliably.
- **mdCATH's replicates are not fully independent.** At one temperature they share one equilibrated starting structure.
- **Only flexibility is computed from your own files.** The `folded` and `stable` questions currently use values stored in mdCATH files. For your own files, only flexibility (`flexible`, `hotspots`) is computed.

## Data, licences and credit

- **Code:** MIT (see `LICENSE`). See `THIRD_PARTY.md` for dependencies and bundled assets: 3Dmol.js (BSD-3-Clause) and Atkinson Hyperlegible (SIL OFL 1.1).
- **mdCATH:** Mirarchi, Giorgino & De Fabritiis, *Scientific Data* 11, 1299 (2024). The paper states CC BY 4.0, while some Hugging Face card revisions say CC BY-ND 4.0. Concordance downloads the data at run time and redistributes none of it.
- **Please cite MDAnalysis** if you use Concordance:
  - Michaud-Agrawal et al., *J. Comput. Chem.* 32, 2319 (2011);
  - Gowers et al., *Proc. 15th Python in Science Conf.*, 98 (2016).
- See `CITATION.cff` for citing Concordance itself.

Concordance is written in the style of an MDAKit (an MDAnalysis toolkit: a separate package that builds on MDAnalysis). Submitting it to the MDAKit registry is a next step.

## How it was made

Built by Tamana Sisodiya (MIT Media Lab) during the hackathon, starting on the evening of Friday 16 October 2026 after the kickoff, with Claude (Anthropic) as a coding assistant. Every claim above was checked by running it, and the tests are in `tests/`.
