"""Open any .xvg (or a CSV/TSV table) as a modern chart: an open replacement for xmgrace.

Every GROMACS analysis tool writes its results as .xvg files laid out for Grace (xmgrace), a
plotting program built on X11 and Motif whose last release was in 2015; on a modern Mac it
needs XQuartz just to open. `concordance plot` reads the same files (and the tables other
tools write) and makes one self-contained HTML page: one line per file or column, the range
across runs shaded, a hover readout, a summary table, and CSV/SVG downloads. Nothing is uploaded
and the page works offline.
"""

from __future__ import annotations

import bz2
import gzip
import html
import json
import re
from pathlib import Path

import numpy as np

DIRECTIVE = re.compile(r'^@\s*(title|subtitle|xaxis\s+label|yaxis\s+label)\s+"(.*)"\s*$', re.I)
LEGEND = re.compile(r'^@\s*s(\d+)\s+legend\s+"(.*)"\s*$', re.I)
SPLIT = re.compile(r"[\s,;]+")
X_NAME = re.compile(r"^#?\s*(time|t|frame|frames|step|steps|x|ns|ps|time[ _(].*|res(idue)?([ _.-]?(number|num|no|nr|id|i))?|resi|resid|resnum|index|position)$", re.I)
GRACE_ESCAPES = re.compile(r"\\[a-zA-Z0-9]|\\\\")


def _clean(label: str) -> str:
    """Grace labels carry escapes such as \\S2\\N (superscript) and \\xa\\f{} (Greek): keep the words."""
    greek = {"\\xa\\f{}": "α", "\\xb\\f{}": "β", "\\xD\\f{}": "Δ", "\\xl\\f{}": "λ", "\\xm\\f{}": "μ", "\\xs\\f{}": "σ", "\\xc\\f{}": "χ"}
    for k, v in greek.items():
        label = label.replace(k, v)
    label = label.replace("\\S", "^").replace("\\N", "").replace("\\s", "_")
    return GRACE_ESCAPES.sub("", label).strip()


def _number(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    elif raw[:3] == b"BZh":
        raw = bz2.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def parse(text: str, name: str = "file") -> dict:
    """Series in a .xvg, .dat, .csv or .tsv file.

    Returns {"title", "xlabel", "ylabel", "series": [{"label", "x", "y"}], "x_from": "column" | "row"}.
    Grace sets separated by '&' each become their own series. The first column is the x-axis when
    it is named like time/frame/residue, or (with no names) when it only ever increases; otherwise
    rows are numbered 1, 2, 3, ... and every column is a series. Text columns are skipped.
    """
    meta = {"title": "", "subtitle": "", "xaxis label": "", "yaxis label": ""}
    legends: dict[int, str] = {}
    blocks: list[list[list[str]]] = [[]]
    names: list[str] | None = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("&"):
            blocks.append([])
            continue
        if s[0] in "#@":
            m = DIRECTIVE.match(s)
            if m:
                meta[re.sub(r"\s+", " ", m.group(1).lower())] = _clean(m.group(2))
            m = LEGEND.match(s)
            if m:
                legends[int(m.group(1))] = _clean(m.group(2))
            continue
        tokens = [t.strip("\"'") for t in SPLIT.split(s) if t.strip("\"'")]
        if not tokens:
            continue
        if names is None and not blocks[0] and len(blocks) == 1 and any(_number(t) is None for t in tokens):
            names = tokens
            continue
        blocks[-1].append(tokens)
    blocks = [b for b in blocks if b]
    if not blocks:
        raise ValueError(f"{name}: no rows of numbers found.")
    series: list[dict] = []
    x_from = "row"
    for b_index, block in enumerate(blocks):
        width = len(block[0])
        rows = [r for r in block if len(r) == width]
        numeric_cols = [j for j in range(width) if all(_number(r[j]) is not None for r in rows)]
        if not numeric_cols:
            continue
        data = {j: np.array([float(r[j]) for r in rows]) for j in numeric_cols}
        col_names = None
        if names is not None and len(names) == width:
            col_names = names
        elif names is not None and len(names) == len(numeric_cols):
            col_names = {j: names[k] for k, j in enumerate(numeric_cols)}
            col_names = [col_names.get(j, f"column {j + 1}") for j in range(width)]
        first = numeric_cols[0]
        use_x = False
        if len(numeric_cols) >= 2:
            named = col_names is not None and bool(X_NAME.match(col_names[first]))
            use_x = named or bool(np.all(np.diff(data[first]) > 0))
        if use_x:
            x = data[first]
            ys = numeric_cols[1:]
            x_from = "column"
        else:
            x = np.arange(1, len(rows) + 1, dtype=float)
            ys = numeric_cols
        for k, j in enumerate(ys):
            if col_names is not None:
                label = col_names[j]
            elif (len(series)) in legends:
                label = legends[len(series)]
            elif len(ys) == 1 and len(blocks) == 1:
                label = name
            else:
                label = f"column {j + 1}" if len(blocks) == 1 else f"set {b_index + 1}" + (f", column {j + 1}" if len(ys) > 1 else "")
            series.append({"label": label, "x": x.tolist(), "y": data[j].tolist()})
    if not series:
        raise ValueError(f"{name}: no numeric columns found.")
    if not meta["xaxis label"] and x_from == "row" and any("rmsf" in s["label"].lower() for s in series):
        meta["xaxis label"] = "residue (in table order)"
    xlabel = meta["xaxis label"] or ((names[0] if names and x_from == "column" else "") if names else "") or ("row" if x_from == "row" else "")
    return {"title": meta["title"], "subtitle": meta["subtitle"], "xlabel": xlabel, "ylabel": meta["yaxis label"], "series": series, "x_from": x_from}


def summary(series: list[dict]) -> list[dict]:
    """Plain numbers per series: points, mean, SD, min, max, and the mean of the second half (for time series)."""
    out = []
    for s in series:
        y = np.asarray(s["y"], dtype=float)
        half = y[len(y) // 2:]
        out.append({"label": s["label"], "points": int(len(y)), "mean": float(y.mean()), "sd": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
                    "min": float(y.min()), "max": float(y.max()), "second_half_mean": float(half.mean())})
    return out


def build(paths: list[str | Path], title: str | None = None) -> dict:
    """Read several files into one chart's data. With one series per file, each file is labelled by its name."""
    parsed = []
    for p in paths:
        p = Path(p)
        parsed.append((p, parse(read_text(p), p.name)))
    series = []
    for p, d in parsed:
        for s in d["series"]:
            label = s["label"]
            if len(parsed) > 1:
                label = p.stem if (len(d["series"]) == 1 or label == p.name) else f"{p.stem}: {label}"
            series.append({**s, "label": label, "file": p.name})
    first = parsed[0][1]
    xlabels = {d["xlabel"] for _, d in parsed if d["xlabel"]}
    ylabels = {d["ylabel"] for _, d in parsed if d["ylabel"]}
    run_tag = re.compile(r"(^|[^a-z])(r|run|rep|replica|replicate|seed|traj)[ _.-]?\d+$", re.I)
    runs_like = len(series) >= 2 and (
        (len(parsed) > 1 and all(len(d["series"]) == 1 for _, d in parsed)) or all(run_tag.search(s["label"]) for s in series)
    )
    same_x = len(series) >= 2 and all(len(s["x"]) == len(series[0]["x"]) and np.allclose(s["x"], series[0]["x"]) for s in series)
    return {
        "title": title or first["title"] or (parsed[0][0].stem if len(parsed) == 1 else ", ".join(p.stem for p, _ in parsed)),
        "subtitle": first["subtitle"] if len(parsed) == 1 else "",
        "xlabel": xlabels.pop() if len(xlabels) == 1 else (" / ".join(sorted(xlabels)) if xlabels else ""),
        "ylabel": ylabels.pop() if len(ylabels) == 1 else (" / ".join(sorted(ylabels)) if ylabels else ""),
        "files": [p.name for p, _ in parsed],
        "series": series,
        "same_x": bool(same_x),
        "runs_like": bool(runs_like),
        "summary": summary(series),
    }


def render(data: dict) -> str:
    from . import __version__
    from .report import _b64

    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__TITLE__", html.escape(data["title"])).replace("__VERSION__", __version__)
    page = page.replace("__FONT400__", _b64("atkinson-hyperlegible-latin-400-normal.woff2")).replace("__FONT700__", _b64("atkinson-hyperlegible-latin-700-normal.woff2"))
    return page.replace("__PLOT_DATA__", payload, 1)


def write(paths: list[str | Path], out: str | Path, title: str | None = None) -> tuple[Path, dict]:
    data = build(paths, title)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")
    return out, data


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
@font-face { font-family: "Atkinson Hyperlegible"; font-weight: 400; src: url(data:font/woff2;base64,__FONT400__) format("woff2"); }
@font-face { font-family: "Atkinson Hyperlegible"; font-weight: 700; src: url(data:font/woff2;base64,__FONT700__) format("woff2"); }
:root {
  color-scheme: light;
  --page: #edf0f3; --panel: #ffffff; --ink: #0c1724; --ink-2: #3e5166; --muted: #6a7b8e; --rule: #dce2e8; --rule-strong: #aebac6;
  --band: rgba(12, 23, 36, 0.10); --focus: #2a78d6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0f141a; --panel: #161d26; --ink: #e7edf3; --ink-2: #b4c1ce; --muted: #8fa0b2; --rule: #27313d; --rule-strong: #3a4757;
    --band: rgba(231, 237, 243, 0.12); --focus: #3987e5;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0f141a; --panel: #161d26; --ink: #e7edf3; --ink-2: #b4c1ce; --muted: #8fa0b2; --rule: #27313d; --rule-strong: #3a4757;
  --band: rgba(231, 237, 243, 0.12); --focus: #3987e5;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767;
}
* { box-sizing: border-box; }
[hidden] { display: none !important; }
main > * { min-width: 0; }
body { margin: 0; background: var(--page); color: var(--ink); font-family: "Atkinson Hyperlegible", Verdana, sans-serif; line-height: 1.5; }
main { max-width: 1080px; margin: 0 auto; padding: 32px 16px 56px; display: grid; gap: 18px; }
header { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
h1 { font-size: clamp(1.5rem, 3.4vw, 2.1rem); line-height: 1.15; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--ink-2); margin: 0; max-width: 70ch; }
.panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 16px; padding: 18px 18px 14px; }
.controls { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; margin-bottom: 8px; font-size: 0.92rem; color: var(--ink-2); }
.legend { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { font: inherit; font-size: 0.9rem; display: inline-flex; gap: 7px; align-items: center; padding: 4px 11px; border-radius: 999px; border: 1px solid var(--rule-strong); background: var(--panel); color: var(--ink); cursor: pointer; }
.chip[aria-pressed="false"] { color: var(--muted); border-style: dashed; }
.chip[aria-pressed="false"] .sw { opacity: 0.25; }
.sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
select, .btn { font: inherit; font-size: 0.9rem; padding: 4px 10px; border-radius: 8px; border: 1px solid var(--rule-strong); background: var(--panel); color: var(--ink); }
.btn { cursor: pointer; }
label.tick { display: inline-flex; gap: 6px; align-items: center; }
.chart { position: relative; }
#charts { display: grid; gap: 14px; }
.small-title { font-size: 0.98rem; margin: 2px 0 0; color: var(--ink-2); }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.axis text, .lab { fill: var(--muted); font-size: 12px; font-family: inherit; }
.axis-title { fill: var(--ink-2); font-size: 13px; }
.grid line { stroke: var(--rule); }
.tip { position: absolute; pointer-events: none; background: var(--panel); border: 1px solid var(--rule-strong); border-radius: 10px; padding: 8px 10px; font-size: 0.85rem; box-shadow: 0 6px 20px rgba(0,0,0,0.12); min-width: 150px; display: none; }
.tip b { display: block; margin-bottom: 2px; }
.tip div { display: flex; gap: 8px; align-items: center; justify-content: space-between; }
.tip span.k { display: inline-flex; gap: 6px; align-items: center; color: var(--ink-2); }
.note { color: var(--muted); font-size: 0.88rem; margin: 6px 0 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 8px; border-bottom: 1px solid var(--rule); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--ink-2); font-weight: 700; }
.tablewrap { overflow-x: auto; }
footer { color: var(--muted); font-size: 0.85rem; }
@media (max-width: 600px) { .panel { padding: 12px 10px; } }
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1 id="title"></h1>
      <p class="sub" id="sub"></p>
    </div>
    <button class="btn" id="theme" type="button">Dark</button>
  </header>
  <section class="panel" aria-label="Chart">
    <div class="controls">
      <div class="legend" id="legend" role="group" aria-label="Show or hide each series"></div>
    </div>
    <div class="controls">
      <label>Layout <select id="layout"><option value="overlay">on one chart</option><option value="separate">one chart each</option></select></label>
      <label class="tick" id="band-wrap"><input type="checkbox" id="band" checked> Shade the range across runs, with their mean</label>
      <label>Smooth <select id="smooth"><option value="1">off</option><option value="5">5 points</option><option value="25">25 points</option><option value="101">101 points</option></select></label>
      <button class="btn" id="csv" type="button">Download CSV</button>
      <button class="btn" id="svgdl" type="button">Download SVG</button>
    </div>
    <div id="charts"></div>
    <p class="note" id="smooth-note" hidden>Lines are smoothed for display only (running mean); the table uses the raw numbers.</p>
  </section>
  <section class="panel" aria-label="Summary table">
    <div class="tablewrap"><table id="table"></table></div>
    <p class="note">"Second half" is the mean over the last half of the points, a common check that a time series has settled.</p>
  </section>
  <footer>Made with <b>concordance plot</b> __VERSION__: an open, offline replacement for opening .xvg files in xmgrace. Nothing was uploaded.</footer>
</main>
<script>
var D = __PLOT_DATA__;
(function () {
  var $ = function (id) { return document.getElementById(id); };
  var NS = "http://www.w3.org/2000/svg";
  var visible = D.series.map(function () { return true; });
  var hoverIndex = null;
  function colour(i) { return i < 8 ? "var(--s" + (i + 1) + ")" : "var(--muted)"; }
  function fmt(v) { if (!isFinite(v)) return "–"; var a = Math.abs(v); if (a >= 1e5 && a < 1e12) return Math.round(v).toString(); return a !== 0 && (a < 0.001 || a >= 1e12) ? v.toExponential(2) : (+v.toPrecision(4)).toString(); }
  function tfmt(v, step) { var d = Math.max(0, Math.min(8, -Math.floor(Math.log10(step) + 1e-9))); return Math.abs(v) >= 1e12 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d); }
  function el(name, attrs, parent) { var n = document.createElementNS(NS, name); for (var k in attrs) n.setAttribute(k, attrs[k]); if (parent) parent.appendChild(n); return n; }
  function ticks(lo, hi, n) {
    if (lo === hi) { lo -= 1; hi += 1; }
    var step = Math.pow(10, Math.floor(Math.log10((hi - lo) / n))), err = (hi - lo) / n / step;
    step *= err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
    var out = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(12));
    return out;
  }
  function smooth(y, w) {
    if (w <= 1) return y;
    var h = Math.floor(w / 2), out = new Array(y.length), s = 0, c = 0, i;
    for (i = 0; i < Math.min(h, y.length); i++) { s += y[i]; c++; }
    for (i = 0; i < y.length; i++) {
      if (i + h < y.length) { s += y[i + h]; c++; }
      if (i - h - 1 >= 0) { s -= y[i - h - 1]; c--; }
      out[i] = s / c;
    }
    return out;
  }
  $("title").textContent = D.title;
  var many = D.series.length > 1;
  $("sub").textContent = (D.subtitle ? D.subtitle + ". " : "") + D.series.length + " series from " + D.files.join(", ") + ".";
  if (!many) $("layout").parentNode.hidden = true;

  D.series.forEach(function (s, i) {
    var b = document.createElement("button");
    b.className = "chip"; b.type = "button"; b.setAttribute("aria-pressed", "true");
    b.innerHTML = '<span class="sw"></span>';
    b.querySelector(".sw").style.background = colour(i);
    b.appendChild(document.createTextNode(s.label));
    b.addEventListener("click", function () { visible[i] = !visible[i]; b.setAttribute("aria-pressed", visible[i] ? "true" : "false"); draw(); });
    $("legend").appendChild(b);
  });
  if (!many) $("legend").hidden = true;

  // summary table
  var t = $("table");
  t.innerHTML = "<thead><tr><th>Series</th><th>Points</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th><th>Second half</th></tr></thead>";
  var tb = document.createElement("tbody");
  D.summary.forEach(function (r, i) {
    var tr = document.createElement("tr");
    var td = document.createElement("td"); var sw = document.createElement("span"); sw.className = "sw"; sw.style.background = colour(i); sw.style.marginRight = "8px"; sw.style.verticalAlign = "middle";
    td.appendChild(sw); td.appendChild(document.createTextNode(r.label)); tr.appendChild(td);
    [r.points, fmt(r.mean), fmt(r.sd), fmt(r.min), fmt(r.max), fmt(r.second_half_mean)].forEach(function (v) { var c = document.createElement("td"); c.textContent = v; tr.appendChild(c); });
    tb.appendChild(tr);
  });
  t.appendChild(tb);

  function drawChart(host, list, opts) {
    var wrap = document.createElement("div"); wrap.className = "chart";
    if (opts.heading) { var h = document.createElement("h2"); h.className = "small-title"; h.textContent = opts.heading; wrap.appendChild(h); }
    var svg = el("svg", { role: "img" }); wrap.appendChild(svg);
    var tip = document.createElement("div"); tip.className = "tip"; wrap.appendChild(tip);
    host.appendChild(wrap);
    var W = Math.max(280, host.clientWidth), H = opts.height || Math.round(Math.min(460, Math.max(260, W * 0.5)));
    var labels = opts.labels;
    var m = { l: 76, r: labels ? 120 : 16, t: 12, b: 46 };
    svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.setAttribute("aria-label", (opts.heading || D.title) + ": line chart of " + list.map(function (s) { return D.series[s.i].label; }).join(", "));
    var xs = [], ys = [];
    list.forEach(function (s) { xs.push(Math.min.apply(null, s.x), Math.max.apply(null, s.x)); ys.push(Math.min.apply(null, s.y), Math.max.apply(null, s.y)); });
    if (!xs.length) { xs = [0, 1]; ys = [0, 1]; }
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var yt = ticks(y0, y1, opts.height ? 4 : 5); y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]);
    var X = function (v) { return m.l + (v - x0) / ((x1 - x0) || 1) * (W - m.l - m.r); };
    var Y = function (v) { return H - m.b - (v - y0) / ((y1 - y0) || 1) * (H - m.t - m.b); };
    var g = el("g", { "class": "grid" }, svg), ax = el("g", { "class": "axis" }, svg);
    yt.forEach(function (v) { el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); var tx = el("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }, ax); tx.textContent = tfmt(v, yt.length > 1 ? yt[1] - yt[0] : 1); });
    var xt_ = ticks(x0, x1, Math.max(3, Math.floor((W - m.l - m.r) / 90)));
    xt_.forEach(function (v) { if (v < x0 || v > x1) return; var tx = el("text", { x: X(v), y: H - m.b + 18, "text-anchor": "middle" }, ax); tx.textContent = tfmt(v, xt_.length > 1 ? xt_[1] - xt_[0] : 1); });
    el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--rule-strong)" }, svg);
    if (D.xlabel) { var xt = el("text", { x: (m.l + W - m.r) / 2, y: H - 8, "text-anchor": "middle", "class": "axis-title" }, svg); xt.textContent = D.xlabel; }
    if (opts.ylabel) { var yl = el("text", { x: 14, y: (m.t + H - m.b) / 2, "text-anchor": "middle", "class": "axis-title", transform: "rotate(-90 14 " + (m.t + H - m.b) / 2 + ")" }, svg); yl.textContent = opts.ylabel; }
    var band = opts.band && list.length >= 2, meanY = [];
    if (band) {
      var top = [], bot = [], mean = [];
      for (var k = 0; k < list[0].x.length; k++) {
        var vals = list.map(function (s) { return s.y[k]; });
        var mv = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length; meanY.push(mv);
        top.push(X(list[0].x[k]) + "," + Y(Math.max.apply(null, vals)));
        bot.unshift(X(list[0].x[k]) + "," + Y(Math.min.apply(null, vals)));
        mean.push(X(list[0].x[k]) + "," + Y(mv));
      }
      el("polygon", { points: top.concat(bot).join(" "), fill: "var(--band)", stroke: "none" }, svg);
    }
    var ends = [];
    list.forEach(function (s) {
      var pts = s.x.map(function (xv, k) { return X(xv).toFixed(1) + "," + Y(s.y[k]).toFixed(1); }).join(" ");
      el("polyline", { points: pts, fill: "none", stroke: colour(s.i), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: band ? 0.85 : 1 }, svg);
      ends.push({ i: s.i, y: Y(s.y[s.y.length - 1]) });
    });
    if (band) {
      el("polyline", { points: mean.join(" "), fill: "none", stroke: "var(--ink)", "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
      ends.push({ i: -1, y: Y(meanY[meanY.length - 1]) });
    }
    if (labels) {
      ends.sort(function (a, b) { return a.y - b.y; });
      for (var e = 1; e < ends.length; e++) if (ends[e].y - ends[e - 1].y < 15) ends[e].y = ends[e - 1].y + 15;
      ends.forEach(function (e) { var tx = el("text", { x: W - m.r + 8, y: e.y + 4, "class": "lab" }, svg); tx.textContent = e.i < 0 ? "mean" : D.series[e.i].label; });
    }
    var cross = el("line", { y1: m.t, y2: H - m.b, stroke: "var(--rule-strong)", "stroke-width": 1, visibility: "hidden" }, svg);
    var dots = list.map(function (s) { return el("circle", { r: 4.5, fill: colour(s.i), stroke: "var(--panel)", "stroke-width": 2, visibility: "hidden" }, svg); });
    var hit = el("rect", { x: m.l, y: m.t, width: Math.max(1, W - m.l - m.r), height: Math.max(1, H - m.t - m.b), fill: "transparent" }, svg);
    function nearest(arr, v) { var lo = 0, hi = arr.length - 1; while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (arr[mid] < v) lo = mid; else hi = mid; } return Math.abs(arr[lo] - v) <= Math.abs(arr[hi] - v) ? lo : hi; }
    function show(evt) {
      if (!list.length) return;
      var r = svg.getBoundingClientRect(), px = (evt.clientX - r.left) * W / r.width;
      var xv = x0 + (px - m.l) / (W - m.l - m.r) * (x1 - x0);
      var k = nearest(list[0].x, xv), xk = list[0].x[k];
      cross.setAttribute("x1", X(xk)); cross.setAttribute("x2", X(xk)); cross.setAttribute("visibility", "visible");
      var rows = "<b>" + (D.xlabel || "x") + " " + fmt(xk) + "</b>";
      list.forEach(function (s, j) {
        var kk = nearest(s.x, xk);
        dots[j].setAttribute("cx", X(s.x[kk])); dots[j].setAttribute("cy", Y(s.y[kk])); dots[j].setAttribute("visibility", "visible");
        rows += '<div><span class="k"><span class="sw" style="background:' + colour(s.i) + '"></span>' + D.series[s.i].label.replace(/</g, "&lt;") + "</span><span>" + fmt(s.y[kk]) + "</span></div>";
      });
      if (band) rows += '<div><span class="k">mean</span><span>' + fmt(meanY[k]) + "</span></div>";
      tip.innerHTML = rows; tip.style.display = "block";
      var left = (X(xk) / W) * r.width + 14; if (left + tip.offsetWidth > r.width) left = (X(xk) / W) * r.width - tip.offsetWidth - 14;
      tip.style.left = Math.max(0, left) + "px"; tip.style.top = (svg.offsetTop + 8) + "px";
    }
    function hide() { cross.setAttribute("visibility", "hidden"); dots.forEach(function (d) { d.setAttribute("visibility", "hidden"); }); tip.style.display = "none"; }
    hit.addEventListener("mousemove", show); hit.addEventListener("mouseleave", hide);
    hit.addEventListener("touchstart", function (e) { show(e.touches[0]); }, { passive: true });
    hit.addEventListener("touchmove", function (e) { show(e.touches[0]); }, { passive: true });
  }
  function draw() {
    var host = $("charts");
    while (host.firstChild) host.removeChild(host.firstChild);
    var w = parseInt($("smooth").value, 10);
    $("smooth-note").hidden = w <= 1;
    var shown = [];
    D.series.forEach(function (s, i) { if (visible[i]) shown.push({ i: i, x: s.x, y: smooth(s.y, w) }); });
    var together = $("layout").value === "overlay";
    $("band-wrap").hidden = !(together && D.same_x && D.runs_like && many);
    if (together || shown.length <= 1) {
      drawChart(host, shown, { band: $("band").checked && D.same_x && D.runs_like, labels: many && shown.length <= 4, ylabel: D.ylabel });
    } else {
      shown.forEach(function (s) { drawChart(host, [s], { heading: D.series[s.i].label, height: 220, ylabel: "" }); });
    }
  }
  $("layout").value = D.runs_like || D.series.length === 1 ? "overlay" : "separate";
  $("layout").addEventListener("change", draw);
  $("band").addEventListener("change", draw);
  $("smooth").addEventListener("change", draw);
  window.addEventListener("resize", draw);

  function save(name, type, text) { var a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: type })); a.download = name; document.body.appendChild(a); a.click(); a.remove(); }
  $("csv").addEventListener("click", function () {
    var lines;
    if (D.same_x || D.series.length === 1) {
      lines = [["x"].concat(D.series.map(function (s) { return JSON.stringify(s.label); })).join(",")];
      D.series[0].x.forEach(function (xv, k) { lines.push([xv].concat(D.series.map(function (s) { return s.y[k]; })).join(",")); });
    } else {
      lines = ["series,x,y"];
      D.series.forEach(function (s) { s.x.forEach(function (xv, k) { lines.push(JSON.stringify(s.label) + "," + xv + "," + s.y[k]); }); });
    }
    save(D.title.replace(/[^\w.-]+/g, "_") + ".csv", "text/csv", lines.join("\n") + "\n");
  });
  $("svgdl").addEventListener("click", function () {
    var clone = document.querySelector("#charts svg").cloneNode(true), cs = getComputedStyle(document.documentElement);
    var txt = new XMLSerializer().serializeToString(clone).replace(/var\(--([\w-]+)\)/g, function (_, k) { return cs.getPropertyValue("--" + k).trim(); });
    txt = txt.replace("<svg", '<svg style="font-family:Verdana,sans-serif;font-size:12px" ');
    save(D.title.replace(/[^\w.-]+/g, "_") + ".svg", "image/svg+xml", txt);
  });
  var root = document.documentElement;
  function themeLabel() { var dark = root.getAttribute("data-theme") === "dark" || (!root.getAttribute("data-theme") && matchMedia("(prefers-color-scheme: dark)").matches); $("theme").textContent = dark ? "Light" : "Dark"; return dark; }
  $("theme").addEventListener("click", function () { root.setAttribute("data-theme", themeLabel() ? "light" : "dark"); themeLabel(); draw(); });
  themeLabel();
  draw();
})();
</script>
</body>
</html>
"""
