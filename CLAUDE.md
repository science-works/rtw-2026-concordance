# Working with Concordance in Claude Code

This repository includes an MCP server (`.mcp.json`) that gives you Concordance's tools. Use them to answer questions about protein flexibility across replicate molecular dynamics runs.

When someone asks about a protein's flexibility, a loop, a region or "which parts move":

1. If they give you a folder or a downloaded .zip of simulation files (data from a paper, an ATLAS entry, their own runs), call `check_dataset` with its path: it identifies the files, compares the runs, checks any published RMSF table, and returns a passport id and a report path. If you don't know what data is available, call `list_proteins` first.
2. For "which parts move?", call `find_flexible_residues`. For a specific stretch, call `check_region_flexible` (mdCATH), `check_region_flexible_in_files` (their own trajectories) or `check_region_flexible_in_gmx_rmsf` (their existing gmx rmsf .xvg files).
3. Answer in plain language first: the verdict, how many runs agree, and what a single run alone would have told them. Quote numbers exactly as the tool returned them.
4. Always give the `passport_id`, and offer `make_report` if they would like the visual report.
5. Say clearly that agreement between runs is not proof that an answer is right, only that it holds up across these runs.

Do not invent numbers, residues or verdicts that a tool did not return. If a tool returns an error, explain it plainly and suggest the fix it gives.
