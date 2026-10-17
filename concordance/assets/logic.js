/* Concordance report logic: pure functions over the report data, with no access to the page.
   The report inlines this file. tests/test_question_box.py runs it in Node and checks it against
   the Python engine, so the browser and the command line give the same answers.

   The question box is deliberately simple: it recognises a handful of question types with word
   patterns, answers only from the numbers in the report, shows how it read each question, and says
   so when it does not understand. It is not a language model and it never guesses. */
var ConcordanceLogic = (function () {
  "use strict";

  var WORD_NUMBERS = { one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10 };
  var ORDINALS = { first: 1, second: 2, third: 3, fourth: 4, fifth: 5, sixth: 6, seventh: 7, eighth: 8, ninth: 9, tenth: 10 };
  var NUMBER_WORD = "(\\d+|one|two|three|four|five|six|seven|eight|nine|ten)";
  var RUN_NOUN = "(?:runs?|replicas?|replicates?|simulations?|films?|repeats?)";

  var STILL = /\b(stable|stiff|rigid|still|calm|steady|static|frozen|fixed|immobile|stays? put|(?:does ?n'?o?t|do ?n'?o?t|does not|do not|not) move)\b/;
  var MOVE = /\b(flexib\w*|mov\w*|mobile|mobility|floppy|flop\w*|wobbl\w*|wiggl\w*|jiggl\w*|loose|dynamic\w*|fluctuat\w*|hot ?spots?|swing\w*|bend\w*|shak\w*|rmsf)\b/;
  var CUE = /\b(mean|means|meaning|define|definition|explain|stand for|stands for)\b/;

  // Glossary entries, most specific first. `bare: false` means "what is X?" alone is not read as a definition request.
  var TERMS = [
    { key: "dependent", re: "replica[- ]?dependent|depend\\w*" },
    { key: "dependent", re: "disagree\\w*|split\\w*", bare: false },
    { key: "consistent", re: "consistent(?: across (?:runs|replicas))?" },
    { key: "flexible", re: "flexib\\w*|moves? a lot|moving a lot", bare: false },
    { key: "passport", re: "(?:evidence )?passport(?: id)?|receipt" },
    { key: "threshold", re: "threshold|cut-?off|mad|median absolute deviation" },
    { key: "ends", re: "chain ends?|termin\\w*|not scored|unscored|grey|gray" },
    { key: "colours", re: "colou?rs?|blue|orange|pale orange" },
    { key: "rmsf", re: "rmsf|root mean square fluctuation|fluctuations?" },
    { key: "units", re: "nm|nanomet\\w*|units?|angstroms?" },
    { key: "mdcath", re: "mdcath" },
    { key: "page", re: "concordance|this (?:tool|page|report)|(?:i'?m|i am|am i) looking at" },
    { key: "residue", re: "residues?|amino acids?" },
    { key: "simulation", re: "simulations?|molecular dynamics|md|trajector(?:y|ies)|films?" },
    { key: "run", re: "runs?|replicas?|replicates?|repeats?" }
  ];

  function wordNumber(w) { return /^\d+$/.test(w) ? Number(w) : WORD_NUMBERS[w]; }
  function pct(x) { return Math.round(x * 100) + "%"; }
  function fixed(x, d) { return Number(x).toFixed(d === undefined ? 2 : d); }
  function plural(n, one, many) { return n === 1 ? one : many; }
  function joinList(items) {
    if (items.length <= 1) return items.join("");
    return items.slice(0, -1).join(", ") + " and " + items[items.length - 1];
  }


  // ================================================================ engine (mirrors concordance/verdict.py)
  function pyRound(x) {  // Python's round(): halves go to the even neighbour
    var r = Math.round(x);
    if (Math.abs(x - Math.trunc(x)) === 0.5) r = 2 * Math.round(x / 2);
    return r;
  }
  function median(values) {
    var v = values.slice().sort(function (a, b) { return a - b; }), n = v.length;
    if (!n) return NaN;
    return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
  }
  function coreMask(n, trim) {
    var mask = [];
    for (var i = 0; i < n; i++) mask.push(true);
    if (trim > 0 && n > 2 * trim + 3) for (var j = 0; j < trim; j++) { mask[j] = false; mask[n - 1 - j] = false; }
    return mask;
  }
  function coreValues(values, core) { return values.filter(function (v, i) { return core[i]; }); }
  function thresholdFor(values, core, rule, ref) {
    var vals = values.filter(function (v, i) { return core[i] && (!ref || ref[i]); });
    if (rule.rule === "topk") {
      var k = Math.max(1, pyRound(rule.top_fraction * vals.length));
      return vals.slice().sort(function (a, b) { return a - b; })[vals.length - k];
    }
    var med = median(vals);
    var mad = median(vals.map(function (v) { return Math.abs(v - med); }));
    return Math.max(med + rule.k * mad, rule.min_ratio * med);
  }
  function flagResidues(values, core, rule, ref) {
    var t = thresholdFor(values, core, rule, ref);
    return values.map(function (v, i) { return core[i] && (rule.rule === "topk" ? v >= t : v > t) ? 1 : 0; });
  }
  function describeRule(rule) {
    var trim = rule.trim_termini ? " The first and last " + rule.trim_termini + " residues are not scored, because chain ends are always floppy." : "";
    if (rule.rule === "topk") return "a residue counts as flexible in a run if its C-alpha RMSF is in that run's top " + Math.round(rule.top_fraction * 100) + "%." + trim;
    return "a residue counts as flexible in a run if its C-alpha RMSF is more than " + fmtG(rule.k) + " median absolute deviations above that run's median, and at least " +
      fmtG(rule.min_ratio) + " times that median." + trim;
  }
  function fmtG(x) { return String(Number(x)); }
  function describeReference(reference) {  // mirrors Method.describe_reference()
    if (!reference) return "";
    return " Each run's typical movement is measured over residues " + reference[0] + "-" + reference[1] + " only " +
      "(for example a folded domain, so a long disordered tail doesn't set the baseline).";
  }
  function referenceMask(residues, core, reference) {  // mirrors reference_mask()
    if (!reference) return null;
    var mask = residues.map(function (r) { return r.n >= reference[0] && r.n <= reference[1]; });
    if (mask.filter(function (m, i) { return m && core[i]; }).length < 5) throw new Error("The reference residues " + reference[0] + "-" + reference[1] + " include fewer than 5 scored residues; choose a longer stretch.");
    return mask;
  }
  function percentile(values, q) {  // numpy's default (linear) percentile
    var v = values.slice().sort(function (a, b) { return a - b; }), pos = (v.length - 1) * q / 100, lo = Math.floor(pos), hi = Math.ceil(pos);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
  }
  function disorderHint(rmsf, ids, core, residues, hasReference) {  // mirrors disorder_hint() in analysis.py
    if (hasReference) return null;
    var scored = [];
    core.forEach(function (c, i) { if (c) scored.push(i); });
    if (scored.length < 20) return null;
    var mean = residues.map(function (r, i) { var t = 0; ids.forEach(function (id) { t += rmsf[id][i]; }); return t / ids.length; });
    var floor = percentile(scored.map(function (i) { return mean[i]; }), 10);
    var share = scored.filter(function (i) { return mean[i] > 3 * floor; }).length / scored.length;
    if (share <= 0.4) return null;
    var groups = [];
    scored.forEach(function (i) {
      if (mean[i] > 3 * floor) return;
      if (groups.length && i - groups[groups.length - 1][groups[groups.length - 1].length - 1] <= 10) groups[groups.length - 1].push(i);
      else groups.push([i]);
    });
    var best = null;
    groups.forEach(function (g) { if (!best || g[g.length - 1] - g[0] > best[best.length - 1] - best[0]) best = g; });
    var rounded = Math.round(share * 1000) / 1000;
    if (!best || best[best.length - 1] - best[0] < 9) return { share: rounded, reference: null };
    return { share: rounded, reference: [residues[best[0]].n, residues[best[best.length - 1]].n] };
  }
  function needFor(consistency, n) { return Math.ceil(consistency * n - 1e-9); }
  function classOf(a, n, need) { return a < 0 ? "u" : a === 0 ? "n" : a <= n - need ? "m" : a < need ? "d" : "c"; }
  function groupStretches(mask, residues) {
    var out = [];
    mask.forEach(function (flag, i) {
      if (!flag) return;
      var last = out[out.length - 1];
      if (last && last[1] === i - 1 && residues[i].c === residues[i - 1].c && residues[i].n === residues[i - 1].n + 1) last[1] = i;
      else out.push([i, i]);
    });
    return out;
  }
  function contestedStretches(agreement, n, need, minLength) {  // mirrors contested_stretches()
    minLength = minLength || 3;
    var out = [], start = null;
    agreement.forEach(function (a, i) {
      var flag = a > 0 && a > n - need && a < need;
      if (flag && start === null) start = i;
      if (!flag && start !== null) { if (i - start >= minLength) out.push([start, i - 1]); start = null; }
    });
    if (start !== null && agreement.length - start >= minLength) out.push([start, agreement.length - 1]);
    return out;
  }
  function ranks(values) {
    var order = values.map(function (v, i) { return [v, i]; }).sort(function (a, b) { return a[0] - b[0]; });
    var r = new Array(values.length), k = 0;
    while (k < order.length) {
      var j = k;
      while (j + 1 < order.length && order[j + 1][0] === order[k][0]) j++;
      for (var m = k; m <= j; m++) r[order[m][1]] = (k + j) / 2 + 1;
      k = j + 1;
    }
    return r;
  }
  function spearman(x, y) {
    var rx = ranks(x), ry = ranks(y), n = x.length;
    var mx = rx.reduce(function (a, b) { return a + b; }, 0) / n, my = ry.reduce(function (a, b) { return a + b; }, 0) / n;
    var sxy = 0, sxx = 0, syy = 0;
    for (var i = 0; i < n; i++) { sxy += (rx[i] - mx) * (ry[i] - my); sxx += (rx[i] - mx) * (rx[i] - mx); syy += (ry[i] - my) * (ry[i] - my); }
    return sxx && syy ? sxy / Math.sqrt(sxx * syy) : NaN;
  }
  function round4(x) { return Math.round(x * 10000) / 10000; }

  // ---------------------------------------------------------------- reading files (mirrors parse_rmsf_text in xvg.py)
  var NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;
  function detectUnit(header) {
    var t = header.toLowerCase();
    if (t.indexOf("(nm)") >= 0 || /\bnm\b/.test(t)) return "nm";
    if (t.indexOf("atomicflx") >= 0 || t.indexOf("angstrom") >= 0 || t.indexOf("(å)") >= 0 || t.indexOf("(a)") >= 0 || t.indexOf("å") >= 0) return "A";
    return null;
  }
  var RESIDUE_NAME = /^#?\s*(res(idue)?([ _.-]?(number|num|no|nr|id|seq|i))?|resi|resid|resnum|resseq|seq|position|pos)$/i;
  var SUMMARY_NAME = /^(mean|average|avg|median|sd|std|stdev|std[ _.-]?dev|sem|se|stderr|err|error|min|max|range|runs?[ _-]?flagging|flagged|count|agreement|scored)([ _(].*)?$/i;
  var LEGEND = /^@\s*s(\d+)\s+legend\s+"(.*)"\s*$/i;
  function parseRmsfText(text, name) {
    name = name || "file";
    var headerBits = [], legends = {}, names = null, numeric = null, width = 0, rows = [];
    String(text).split(/\r\n|\r|\n/).forEach(function (line, k) {
      var n = k + 1, stripped = line.trim();
      if (!stripped) return;
      if ("#@&".indexOf(stripped[0]) >= 0) {
        headerBits.push(stripped);
        var legend = LEGEND.exec(stripped);
        if (legend) legends[Number(legend[1])] = legend[2];
        return;
      }
      var tokens = stripped.split(/[\s,;]+/).map(function (t) { return t.replace(/^["']+|["']+$/g, ""); }).filter(function (t) { return t; });
      if (!tokens.length) return;
      if (numeric === null && !tokens.some(function (t) { return NUMBER.test(t); })) { names = tokens; headerBits.push(stripped); return; }
      if (numeric === null) {
        numeric = [];
        tokens.forEach(function (t, i) { if (NUMBER.test(t)) numeric.push(i); });
        width = tokens.length;
      }
      if (tokens.length !== width || !numeric.every(function (i) { return NUMBER.test(tokens[i]); })) {
        throw new Error(name + ", line " + n + ": expected " + width + " columns like the first data row, found “" + stripped.slice(0, 60) + "”");
      }
      rows.push(numeric.map(function (i) { return Number(tokens[i]); }));
    });
    if (!rows.length) throw new Error(name + ": no data rows found. Each row should hold one residue's values.");
    var aligned = names !== null && names.length === width, residueCol = null;
    if (aligned) for (var k = 0; k < numeric.length; k++) if (RESIDUE_NAME.test(names[numeric[k]])) { residueCol = k; break; }
    if (residueCol === null && numeric.length >= 2 && rows.every(function (r) { return Math.abs(r[0] - Math.round(r[0])) <= 1e-8; })) residueCol = 0;
    var numbered = residueCol === null;
    var numbers = rows.map(function (r, i) { return numbered ? i + 1 : Math.round(r[residueCol]); });
    var columns = [], skipped = [];
    numeric.forEach(function (i, k) {
      if (k === residueCol) return;
      var label = aligned ? names[i] : residueCol === 0 && (k - 1) in legends ? legends[k - 1] : "column " + (columns.length + skipped.length + 1);
      if (SUMMARY_NAME.test(label)) { skipped.push(label); return; }
      columns.push({ name: label, values: rows.map(function (r) { return r[k]; }) });
    });
    if (!columns.length) throw new Error(name + ": found residue numbers but no value columns.");
    return { numbers: numbers, columns: columns, unit: detectUnit(headerBits.join(" ")), numbered: numbered, skipped: skipped };
  }
  function parsePdbResidues(text) {
    var byNumber = {}, order = [], lines = [];
    String(text).split(/\r\n|\r|\n/).some(function (line) {
      if (/^ENDMDL/.test(line)) return true;
      if (!/^(ATOM  |HETATM)/.test(line)) return false;
      lines.push(line);
      var atom = line.slice(12, 16).trim(), num = parseInt(line.slice(22, 26), 10);
      if (atom === "CA" && !isNaN(num) && !(num in byNumber)) { byNumber[num] = { name: line.slice(17, 20).trim(), chain: line.slice(21, 22).trim(), ins: line.slice(26, 27).trim() }; order.push(num); }
      return false;
    });
    return { byNumber: byNumber, order: order, pdb: lines.join("\n") + "\nEND\n", atoms: lines.length };
  }
  function shellQuote(word) { return /^[\w@%+=:,./-]+$/.test(word) ? word : "'" + word.replace(/'/g, "'\"'\"'") + "'"; }

  // Build the report's data from files read in the browser. Mirrors from_xvg() and report_data() in Python.
  function buildData(input) {
    var files = input.files, rule = input.rule, consistency = input.consistency, minFraction = input.min_fraction;
    if (!files.length) throw new Error("Add your files first: one file per run, or one table with a column per run.");
    var tables = files.map(function (f) { return parseRmsfText(f.text, f.file); });
    var numbers, runs, detected;
    if (files.length === 1) {
      if (tables[0].columns.length < 2) throw new Error("Give at least two runs: one file per run, or one table with a column per run.");
      numbers = tables[0].numbers;
      runs = tables[0].columns.map(function (c) { return { label: c.name, values: c.values }; });
      detected = [tables[0].unit];
    } else {
      tables.forEach(function (t, k) {
        if (t.columns.length !== 1) throw new Error(files[k].file + " has " + t.columns.length + " value columns. Give one file per run, or one table with a column per run.");
      });
      numbers = tables[0].numbers;
      tables.forEach(function (t, k) {
        if (t.numbers.length !== numbers.length || t.numbers.some(function (x, i) { return x !== numbers[i]; })) {
          throw new Error(files[k].file + " covers different residues from " + files[0].file + "; every run must list the same residues.");
        }
      });
      runs = tables.map(function (t, k) { return { label: files[k].file, values: t.columns[0].values }; });
      detected = tables.map(function (t) { return t.unit; });
    }
    var seen = {};
    numbers.forEach(function (x) { if (seen[x]) throw new Error("Residue numbers repeat (several chains?). Give one chain at a time for now."); seen[x] = true; });
    var known = detected.filter(function (u) { return u; }).filter(function (u, i, a) { return a.indexOf(u) === i; });
    var units = input.units, unitSource = "chosen on the page";
    if (!units) {
      if (known.length > 1) throw new Error("Some files are in nm and some in ångström; convert them to the same unit first.");
      units = known.length ? known[0] : "nm";
      unitSource = known.length ? "from the file header" : "assumed, as the files don't say";
    }
    var scale = units === "A" ? 0.1 : 1;
    var structure = input.structure ? parsePdbResidues(input.structure.text) : null;
    var numbered = tables.some(function (t) { return t.numbered; }), numbering = null;
    if (numbered && structure && structure.order.length === numbers.length) {
      numbers = structure.order.slice();
      numbering = "the files have no residue column, so residues are numbered as in the structure, in order";
    } else if (numbered && structure) {
      numbering = "the files have no residue column, so residues are numbered 1 to " + numbers.length + "; the structure has " + structure.order.length + " residues, so its numbering was not used";
    } else if (numbered) {
      numbering = "the files have no residue column, so residues are numbered 1 to " + numbers.length;
    }
    var skippedColumns = [];
    tables.forEach(function (t) { t.skipped.forEach(function (c) { if (skippedColumns.indexOf(c) < 0) skippedColumns.push(c); }); });
    skippedColumns.sort();
    var residues = numbers.map(function (x) {
      var r = structure && structure.byNumber[x];
      return { l: (r && r.name ? r.name : "res") + x + (r && r.ins ? r.ins : ""), n: x, c: r ? r.chain : "" };
    });
    var R = residues.length, n = runs.length;
    var core = coreMask(R, rule.trim_termini);
    var reference = input.reference || null, ref = referenceMask(residues, core, reference);
    var need = needFor(consistency, n);
    var data = { rmsf: {}, flags: {}, summaries: {} };
    var ids = runs.map(function (r, k) { return String(k + 1); });
    runs.forEach(function (run, k) { data.rmsf[ids[k]] = run.values.map(function (v) { return v * scale; }); });
    ids.forEach(function (id) { data.flags[id] = flagResidues(data.rmsf[id], core, rule, ref); });
    var agreement = [];
    for (var i = 0; i < R; i++) {
      if (!core[i]) { agreement.push(-1); continue; }
      var a = 0;
      ids.forEach(function (id) { a += data.flags[id][i]; });
      agreement.push(a);
    }
    var classes = agreement.map(function (a) { return classOf(a, n, need); });
    var counts = {
      scored_residues: core.filter(Boolean).length,
      flagged_in_any_run: agreement.filter(function (a) { return a > 0; }).length,
      consistent: classes.filter(function (c) { return c === "c"; }).length,
      replica_dependent: classes.filter(function (c) { return c === "d"; }).length,
      flagged_by_a_minority: classes.filter(function (c) { return c === "m"; }).length
    };
    ids.forEach(function (id) {
      var f = data.flags[id], missed = 0, extra = 0, total = 0;
      for (var i = 0; i < R; i++) {
        var consistent = agreement[i] >= need;
        total += f[i];
        if (consistent && !f[i]) missed++;
        if (f[i] && !consistent) extra++;
      }
      data.summaries[id] = { flexible_residues: total, missed_consistent: missed, extra_not_consistent: extra,
        threshold_nm: round4(thresholdFor(data.rmsf[id], core, rule, ref)), median_rmsf_nm: round4(median(coreValues(data.rmsf[id], core))) };
    });
    var spear = {};
    for (var x = 0; x < n; x++) for (var y = x + 1; y < n; y++) spear[ids[x] + "-" + ids[y]] = spearman(coreValues(data.rmsf[ids[x]], core), coreValues(data.rmsf[ids[y]], core));
    var ruleDict = rule.rule === "topk" ? { rule: "topk", top_fraction: rule.top_fraction, trim_termini: rule.trim_termini }
      : { rule: "mad", k: rule.k, min_ratio: rule.min_ratio, trim_termini: rule.trim_termini };
    var fileWords = [];
    files.forEach(function (f) { fileWords.push(f.file); });
    var dataWords = ["--rmsf"].concat(fileWords)
      .concat(input.structure ? ["-s", input.structure.file] : [])
      .concat(input.name ? ["--name", input.name] : [])
      .concat(input.units ? ["--units", input.units] : []);
    var methodWords = ["--rule", rule.rule].concat(rule.rule === "mad" ? ["--mad-k", fmtG(rule.k), "--min-ratio", fmtG(rule.min_ratio)] : ["--top-fraction", fmtG(rule.top_fraction)])
      .concat(["--trim-termini", String(rule.trim_termini), "--discard", "0.1", "--consistency", fmtG(consistency)])
      .concat(reference ? ["--reference", reference[0] + "-" + reference[1]] : []);
    var words = dataWords.concat(methodWords).map(shellQuote).join(" ");
    var allXvg = files.every(function (f) { return /\.xvg$/i.test(f.file); });
    var name = input.name || (input.structure ? input.structure.file.replace(/\.[^.]+$/, "") : allXvg ? "gmx_rmsf" : files[0].file.replace(/\.[^.]+$/, ""));
    var condition = allXvg ? "gmx rmsf output" : "RMSF from files";
    var created = new Date().toISOString().replace(/\.\d+Z$/, "+00:00");
    var limitations = [
      "Measures agreement between replicate runs, not whether the force field or the physics is right.",
      "Agreement is not proof: runs that start from the same structure can agree and still all be wrong. Disagreement, though, is a clear warning.",
      (n === 5 ? "Five" : String(n)) + " replicates is a small sample; the thresholds are explicit conventions, not statistical guarantees."
    ];
    return {
      version: input.version || "", title: "Concordance: " + name, name: name, kind: "xvg", condition: condition, synthetic: !!input.synthetic,
      audience: "new", n: n, need: need,
      runs: ids.map(function (id, k) { return { id: id, label: "Run " + (k + 1), source: runs[k].label }; }),
      residues: residues, rmsf: data.rmsf, flags: data.flags, agreement: agreement, classes: classes, counts: counts,
      stretches: {
        consistent: groupStretches(classes.map(function (c) { return c === "c"; }), residues),
        some: groupStretches(classes.map(function (c) { return c === "d"; }), residues),
        few: groupStretches(classes.map(function (c) { return c === "m"; }), residues),
        contested: contestedStretches(agreement, n, need)
      },
      summaries: data.summaries, spearman: spear,
      window: { frames_available_per_run: {}, common_length: 0, frames_discarded: 0, frames_used_per_run: 0, frame_range: [0, 0] },
      method: { flexibility_rule: ruleDict, reference_residues: reference, discard_fraction: 0.1, consistency_threshold: consistency, region_min_fraction: minFraction,
        engine: "Concordance in your browser (same rules as the Python engine)", atoms: "as in your files" },
      rule_text: describeRule(rule) + describeReference(reference),
      disorder: disorderHint(data.rmsf, ids, core, residues, !!reference),
      check: { description: "Not applicable: the RMSF values were read from your files (gmx rmsf, cpptraj or a table), not computed here. Frames, fitting and atom selection are whatever was used when they were made.", max_abs_diff_nm: null, agree: null },
      geometry: { ok: true, checked: false, description: "Not applicable: no coordinates were supplied." },
      units: { unit: units, source: unitSource },
      numbering: numbering, skipped_columns: skippedColumns,
      claim: null,
      passport: {
        passport_id: input.passport_id || "made in the browser",
        created_utc: created,
        question: { type: "hotspots", text: "which residues of " + name + " are flexible, and do the runs agree?" },
        source: { files: files.map(function (f) { return { file: f.file, sha256: f.sha256 || "not computed" }; }).concat(input.structure ? [{ file: input.structure.file, sha256: input.structure.sha256 || "not computed" }] : []) },
        software: { "concordance (browser page)": input.version || "" },
        replay: "concordance report " + words,
        limitations: limitations,
        result: counts
      },
      created: created,
      pdb: structure ? structure.pdb : "",
      has_structure: !!(structure && structure.atoms),
      mda_code: "Not applicable: the RMSF values came from your files. To compute them from the simulations instead, use the Concordance command line with -s TOPOLOGY -f RUN1 RUN2 ...",
      region_command: "concordance ask flexible " + words.replace(/ --discard \S+/, " --discard 0.1") + " --region {region} --min-fraction " + fmtG(minFraction),
      hotspots_command: "concordance ask hotspots " + words
    };
  }

  // The same verdicts under other reasonable settings for "moves a lot" (answers "doesn't it depend on your threshold?").
  function settingsGrid(rule) {
    var cells = [];
    if (rule.rule === "topk") [0.1, 0.15, 0.2].forEach(function (t) { cells.push({ rule: "topk", top_fraction: t, trim_termini: rule.trim_termini }); });
    else [2, 3, 4].forEach(function (k) { [1.25, 1.5, 2].forEach(function (r) { cells.push({ rule: "mad", k: k, min_ratio: r, trim_termini: rule.trim_termini }); }); });
    return cells;
  }
  function sameRule(a, b) {
    return a.rule === b.rule && (a.rule === "topk" ? a.top_fraction === b.top_fraction : a.k === b.k && a.min_ratio === b.min_ratio) && a.trim_termini === b.trim_termini;
  }
  function describeCell(c) { return c.rule === "topk" ? "top " + Math.round(c.top_fraction * 100) + "%" : "k " + c.k + ", ×" + c.min_ratio; }

  // Amino acids: one-letter code, name and a simple chemical group for colouring.
  var AA = {
    A: { name: "alanine", group: "hydrophobic" }, V: { name: "valine", group: "hydrophobic" }, L: { name: "leucine", group: "hydrophobic" },
    I: { name: "isoleucine", group: "hydrophobic" }, M: { name: "methionine", group: "hydrophobic" }, C: { name: "cysteine", group: "hydrophobic" },
    F: { name: "phenylalanine", group: "aromatic" }, W: { name: "tryptophan", group: "aromatic" }, Y: { name: "tyrosine", group: "aromatic" },
    S: { name: "serine", group: "polar" }, T: { name: "threonine", group: "polar" }, N: { name: "asparagine", group: "polar" }, Q: { name: "glutamine", group: "polar" },
    K: { name: "lysine", group: "positive" }, R: { name: "arginine", group: "positive" }, H: { name: "histidine", group: "positive" },
    D: { name: "aspartate", group: "negative" }, E: { name: "glutamate", group: "negative" },
    G: { name: "glycine", group: "glycine/proline" }, P: { name: "proline", group: "glycine/proline" }
  };
  var THREE = { ALA: "A", VAL: "V", LEU: "L", ILE: "I", MET: "M", CYS: "C", CYX: "C", CYM: "C", PHE: "F", TRP: "W", TYR: "Y", SER: "S", THR: "T",
    ASN: "N", GLN: "Q", LYS: "K", LYN: "K", ARG: "R", HIS: "H", HSD: "H", HSE: "H", HSP: "H", HID: "H", HIE: "H", HIP: "H", ASP: "D", ASH: "D",
    GLU: "E", GLH: "E", GLY: "G", PRO: "P", MSE: "M" };
  function residueName(r) { var m = /^([A-Za-z]{3,4})-?\d/.exec(r.l || ""); return m ? m[1].toUpperCase() : ""; }
  function aaCode(name) { return THREE[name] || null; }
  function aaGroup(residue) { var c = aaCode(residueName(residue)); return c ? AA[c].group : null; }

  function create(D) {
    var N = D.n, NEED = D.need, FEW = N - NEED, R = D.residues.length;
    var rule = D.method.flexibility_rule;
    var chains = {};
    D.residues.forEach(function (r) { chains[r.c || ""] = true; });
    var multiChain = Object.keys(chains).length > 1;
    var scoredIdx = [];
    for (var s = 0; s < R; s++) if (D.classes[s] !== "u") scoredIdx.push(s);

    function dash(a, b) { return a === b ? String(a) : a + "–" + b; }
    function span(lo, hi) {
      var text = dash(D.residues[lo].n, D.residues[hi].n);
      return multiChain ? "chain " + (D.residues[lo].c || "?") + " " + text : text;
    }
    function stretchList(list, limit) {
      limit = limit || 8;
      var items = list.slice(0, limit).map(function (st) { return span(st[0], st[1]); });
      if (list.length > limit) items.push(list.length - limit + " more");
      return joinList(items);
    }
    function dependentRange() { return dash(FEW + 1, NEED - 1); }
    function resName(i) { var l = D.residues[i].l; return /^res-?\d/.test(l) ? "residue " + D.residues[i].n : l; }
    function residueCount(n) { return n + " " + plural(n, "residue", "residues"); }

    // ---------------------------------------------------------------- verdicts
    // Mirrors region_flexible() and classify() in concordance/verdict.py.
    function verdictFor(indices, flagsByRun) {
      flagsByRun = flagsByRun || D.flags;
      var scored = indices.filter(function (i) { return D.classes[i] !== "u"; });
      if (!scored.length) return null;
      var minFraction = D.method.region_min_fraction;
      var answers = {}, flagged = {}, yes = 0;
      D.runs.forEach(function (run) {
        var f = 0;
        scored.forEach(function (i) { f += flagsByRun[run.id][i] ? 1 : 0; });
        flagged[run.id] = f;
        answers[run.id] = f / scored.length >= minFraction;
        if (answers[run.id]) yes += 1;
      });
      var verdict = yes >= NEED ? "consistent across replicas" : yes <= N - NEED ? "not seen in most replicas" : "replica-dependent";
      var no = N - yes, risk = yes === no ? null : Math.min(yes, no);
      return { answers: answers, flagged: flagged, scored: scored.length, scoredIdx: scored, yes: yes, verdict: verdict, risk: risk, consensus: yes > no };
    }
    function regionVerdict(lo, hi) {
      var idx = [];
      for (var i = lo; i <= hi; i++) idx.push(i);
      return verdictFor(idx);
    }
    // Mirrors ReplicateSet.residue_positions(): every residue whose number lies in [a, b].
    function positions(a, b) {
      var out = [];
      D.residues.forEach(function (r, i) { if (r.n >= a && r.n <= b) out.push(i); });
      return out;
    }
    function verdictName(v) {
      if (v === "consistent across replicas") return { label: "Consistent across runs", dot: "--flex" };
      if (v === "replica-dependent") return { label: "Replica-dependent", dot: "--dep" };
      return { label: "Not seen in most runs", dot: "--none" };
    }
    function riskSentence(v) {
      if (v.risk === null) return "The runs split evenly, so there is no majority answer.";
      if (v.risk === 0) return "Every run, analysed alone, gives the same answer.";
      return v.risk + " of the " + N + " runs, analysed alone, would have told you " + (v.consensus ? "no" : "yes") + ".";
    }
    function meanRmsf(i) {
      var total = 0;
      D.runs.forEach(function (run) { total += D.rmsf[run.id][i]; });
      return total / N;
    }

    // ---------------------------------------------------------------- reading a question
    function parse(text) {
      var raw = String(text || "").trim();
      var t = " " + raw.toLowerCase()
        .replace(/[‐-―−]/g, "-")
        .replace(/[‘’ʼ]/g, "'")
        .replace(/[?!,;"“”]+/g, " ")
        .replace(/\.(\s|$)/g, " ")
        .replace(/\s+/g, " ").trim() + " ";
      var q = { text: raw, run: null, runWord: null, runCount: null, top: null, lo: null, hi: null, several: false };

      // numbers that are not residue numbers: temperatures, decimals, "4 of 5", units
      t = t.replace(/\d+\s*(?:k|kelvin|degrees?)\b/g, " ")
        .replace(/\d+\.\d+/g, " ")
        .replace(/\b\d+\s*(?:\/|out of|of)\s*\d+\b/g, " ")
        .replace(/\d+\s*(?:%|percent|nm|ns|ps|angstroms?|frames?|times)(?![a-z])/g, " ");
      // which run: "the third run", "run 2", "replica #4", "run two"
      t = t.replace(new RegExp("\\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\\s+" + RUN_NOUN + "\\b"), function (m, w) {
        q.run = ORDINALS[w]; q.runWord = "run"; return " ";
      });
      t = t.replace(new RegExp("\\b(run|replica|replicate|simulation|film)\\s*(?:number |no\\.? ?|#)?\\s*" + NUMBER_WORD + "\\b"), function (m, w, k) {
        if (q.run === null) { q.run = wordNumber(k); q.runWord = w; }
        return " ";
      });
      // how many runs: "with 3 runs", "only one simulation", "just once"
      t = t.replace(new RegExp("\\b" + NUMBER_WORD + "\\s+" + RUN_NOUN + "\\b"), function (m, k) {
        q.runCount = wordNumber(k); return " runs ";
      });
      if (/\b(once|a single (?:run|simulation|film)|single (?:run|simulation|film)|one (?:run|simulation|film))\b/.test(t)) q.runCount = 1;
      if (/\btwice\b/.test(t)) q.runCount = 2;
      // "top 5"
      t = t.replace(/\btop\s*(\d+)\b/, function (m, k) { q.top = Number(k); return " top "; });

      var range = t.match(/(?:^|[^\d.])(\d{1,5})\s*(?:-|to|through|thru|until|till|\.\.|:|and)\s*(?:residues?\s*|res\s*|[a-z]{3}\s?)?(\d{1,5})(?![\d.])/);
      if (range) {
        var a = Number(range[1]), b = Number(range[2]);
        q.lo = Math.min(a, b); q.hi = Math.max(a, b);
      } else {
        var found = [], rx = /(?:^|[^\d.])(\d{1,5})(?![\d.])/g, m;
        while ((m = rx.exec(t)) !== null) found.push(Number(m[1]));
        if (found.length) {
          q.lo = Math.min.apply(null, found); q.hi = Math.max.apply(null, found);
          q.several = found.length > 1;
        }
      }
      q.moves = MOVE.test(t);
      // "still" alone can be an adverb ("is it still flexible?"), so it only counts when nothing says "moves"
      q.still = STILL.test(t.replace(/\bstill\b/g, " ")) || (/\bstill\b/.test(t) && !q.moves);
      q.intent = intentOf(t, q);
      return q;
    }

    function termFor(t, bareOnly) {
      var rest = t.trim().replace(/^(?:so |and |ok |okay |please )?(?:what|whats|what's|who|whos|who's)(?: is| are|'s)?(?: an?\b| the\b)?\s*/, "");
      for (var k = 0; k < TERMS.length; k++) {
        var term = TERMS[k];
        if (bareOnly) {
          if (term.bare === false) continue;
          if (new RegExp("^(?:" + term.re + ")$").test(rest.trim())) return term.key;
        } else if (new RegExp("\\b(?:" + term.re + ")\\b").test(t)) {
          return term.key;
        }
      }
      return null;
    }

    function intentOf(t, q) {
      var bare = t.trim();
      if (q.lo !== null) return q.run !== null ? "run-region" : "region";
      if (!bare) return q.run !== null ? "run" : "help";
      if (/^(hi|hello|hey|hiya|good (morning|afternoon|evening))\b/.test(bare) && bare.split(" ").length <= 3) return "hello";
      if (/^(?:(?:great|lovely|perfect|ok|okay|brilliant)\s+)?(thanks|thank you|cheers|ta)\b/.test(bare) && bare.split(" ").length <= 5) return "thanks";
      if (/\b(help|what can (i|you) (ask|do)|what (questions|should i ask)|how do i use|examples?)\b/.test(t)) return "help";
      if (q.run !== null) return "run";
      if (CUE.test(t)) { var term = termFor(t, false); if (term) return "define:" + term; }
      var bareTerm = termFor(t, true);
      if (bareTerm) return "define:" + bareTerm;
      if (/\bhow\b.*\b(decid\w*|calculat\w*|comput\w*|work\w*|measur\w*|made|done|determin\w*|know|tell|judg\w*|scor\w*|pick\w*|choos\w*|chose|get)\b/.test(t) || /\bmethod\w*\b/.test(t)) return "method";
      if (/\bwhy\b.*\b(five|several|multiple|many|more than once|repeat\w*|again|replica\w*|runs|times|simulat\w*)\b/.test(t)) return "why-repeat";
      if (/\b(thresholds?|settings?|parameters?|cut-?offs?|arbitrary|robust\w*|sensitiv\w*|tweak\w*|the rule|your rule|my rule|k value|choices)\b/.test(t)) return "sensitivity";
      if (q.runCount !== null || /\b(enough|fewer|less runs|fewer runs)\b/.test(t)) return "fewer-runs";
      if (/\b(trust\w*|reliab\w*|confiden\w*|sure|believe|proof|prove|correct|accurate|true|valid\w*|certain\w*)\b/.test(t)) return "trust";
      if (/\b(disagree\w*|differ\w*|inconsisten\w*|contest\w*|conflict\w*|unsure|uncertain\w*|careful|caution|depend\w*|vary|varies|variab\w*|split|controvers\w*|doubt\w*|unclear|agree\w*|orange)\b/.test(t)) return "disagree";
      if (/\b(stillest|stiffest|calmest|most rigid|most stable)\b/.test(t)) return "least";
      var most = /\b(most|more|biggest|largest|highest|max\w*|greatest|top|floppiest|wobbliest|loosest)\b/.test(t);
      var least = /\b(least|less|lowest|smallest|minimum|min)\b/.test(t);
      if ((most || least) && (q.moves || q.still || /\b(top|floppiest|wobbliest|loosest)\b/.test(t))) {
        var direction = most ? 1 : -1;
        if (q.still) direction = -direction;
        return direction > 0 ? "most" : "least";
      }
      if (/\bhow many\b/.test(t)) {
        if (q.moves) return "overview";
        if (/\b(residues?|amino acids?)\b/.test(t)) return "count-residues";
        if (new RegExp("\\b" + RUN_NOUN + "\\b").test(t)) return "count-runs";
      }
      if (q.still) return "still";
      if (q.moves || /\b(which part|what part|where)\b/.test(t)) return "overview";
      return "unknown";
    }

    // ---------------------------------------------------------------- answering
    function base(q, intent) {
      return { intent: intent, question: q.text, understood: null, verdict: null, lines: [], links: [], explore: null, run: null, command: null, unknown: false };
    }
    function runByNumber(q) {
      if (q.run === null) return null;
      if (/^replica/.test(q.runWord || "") && D.kind === "mdcath") {
        for (var k = 0; k < D.runs.length; k++) if (String(D.runs[k].id) === String(q.run)) return { run: D.runs[k], note: "mdCATH replica " + q.run + " (" + D.runs[k].label + " on this page)" };
      }
      if (q.run >= 1 && q.run <= N) return { run: D.runs[q.run - 1], note: D.runs[q.run - 1].label };
      return null;
    }
    function noSuchRun(out, q) {
      out.lines.push("There are " + N + " runs here, Run 1 to Run " + N + ", so there is no run " + q.run + ".");
      return out;
    }
    function regionCommand(a, b) { return D.region_command.replace("{region}", a + "-" + b); }
    function foldedNote(a, b) {
      return D.kind === "mdcath"
        ? "This page measures how much each part moves. Whether a stretch keeps its folded shape is a separate check: concordance ask folded --region " + a + "-" + b + "."
        : "This page measures how much each part moves, not whether a stretch keeps its folded shape.";
    }

    function answerRegion(q, audience, out) {
      var first = D.residues[0].n, last = D.residues[R - 1].n;
      var idx = positions(q.lo, q.hi);
      if (!idx.length) {
        out.lines.push("This protein's residues are numbered " + first + " to " + last + ", so there is no " + (q.lo === q.hi ? "residue " + q.lo : "residue between " + q.lo + " and " + q.hi) + " here.");
        out.unknown = true;
        return out;
      }
      var lo = idx[0], hi = idx[idx.length - 1];
      var a = D.residues[lo].n, b = D.residues[hi].n;
      var one = idx.length === 1;
      var where = one ? (/^residue /.test(resName(lo)) ? resName(lo) : "residue " + resName(lo)) : "residues " + span(lo, hi);
      var Where = where.charAt(0).toUpperCase() + where.slice(1);
      var picked = runByNumber(q);
      out.understood = "Read as: " + where + (picked ? ", " + picked.note : "") + (one ? (q.still ? ", does it stay still?" : ", does it move a lot?") : (q.still ? ", do they stay still?" : ", do they move a lot?"));
      if (q.several) out.understood += " (several numbers, so I used the stretch from the lowest to the highest)";
      if (q.lo < first || q.hi > last) out.understood += " (the protein runs from " + first + " to " + last + ")";
      out.explore = { lo: lo, hi: hi, idx: idx };
      var v = verdictFor(idx);
      if (!v) {
        out.lines.push(Where + (one ? " is at a chain end" : " are at the chain ends") + ", which are not scored: chain ends tend to flop about whatever the rest of the protein does. Try a stretch further in.");
        return out;
      }
      out.command = regionCommand(a, b);
      if (q.run !== null && !picked) return noSuchRun(out, q);
      var vn = verdictName(v.verdict);
      out.verdict = vn;
      var it = one ? "it" : "they", move = one ? "moves" : "move";
      if (picked) {
        var yes = v.answers[picked.run.id];
        out.run = picked.run.id;
        out.lines.push(picked.run.label + " says " + (yes ? "yes" : "no") + ": " + (one
          ? resName(lo) + (v.flagged[picked.run.id] ? " moves" : " doesn't move") + " a lot in that run."
          : v.flagged[picked.run.id] + " of the " + v.scored + " scored residues in " + span(lo, hi) + " move a lot in that run."));
        out.lines.push("All " + N + " runs together: " + v.yes + " of " + N + " say yes (" + vn.label.toLowerCase() + ").");
        return out;
      }
      if (q.still) {
        out.lines.push((N - v.yes) + " of " + N + " runs say " + where + " " + (one ? "stays" : "stay") + " fairly still (" + (one ? "it doesn't" : "they don't") + " move a lot); " + v.yes + " say " + it + " " + move + " a lot.");
      } else {
        out.lines.push(v.yes + " of " + N + " runs say " + where + " " + move + " a lot.");
      }
      var thing = one ? "this residue" : "this stretch";
      if (v.verdict === "consistent across replicas") out.lines.push("That holds up across the runs, so it's reasonable to describe " + thing + " as flexible in these simulations.");
      else if (v.verdict === "replica-dependent") out.lines.push("The runs disagree, so be careful: a single run could have told you either answer.");
      else out.lines.push(v.yes === 0 ? "No run says " + it + " " + move + " a lot." : "Most runs say " + it + (one ? " doesn't" : " don't") + " move a lot; only " + v.yes + " " + plural(v.yes, "run says", "runs say") + " " + it + " " + (one ? "does" : "do") + ".");
      out.lines.push(riskSentence(v));
      out.lines.push(settingsLine(sensitivity(idx), audience));
      if (v.scored < idx.length) out.lines.push((idx.length - v.scored) + " of these residues " + plural(idx.length - v.scored, "is", "are") + " at a chain end and not scored.");
      if (audience !== "new") {
        var means = D.runs.map(function (run) {
          var total = 0;
          v.scoredIdx.forEach(function (i) { total += D.rmsf[run.id][i]; });
          return fixed(total / v.scored, 3);
        });
        out.lines.push("Mean RMSF over the scored residues, run by run: " + means.join(", ") + " nm. A run says yes when at least " + pct(D.method.region_min_fraction) + " of them are above that run's threshold.");
      }
      if (q.still) out.lines.push(foldedNote(a, b));
      return out;
    }


    function settingsLine(sv, audience) {
      var line = sv.same === sv.total
        ? "It holds under all " + sv.total + " reasonable settings for what counts as \u201cmoves a lot\u201d, so it doesn't hinge on the threshold."
        : "It holds under " + sv.same + " of " + sv.total + " reasonable settings for what counts as \u201cmoves a lot\u201d" + (sv.same <= sv.total / 2 ? ", so treat it as sensitive to the threshold." : ".");
      if (audience !== "new" && sv.same < sv.total) {
        line += " Different under: " + sv.rows.filter(function (r) { return r.verdict !== sv.verdict; }).map(function (r) {
          return describeCell(r.cell) + " (" + verdictName(r.verdict).label.toLowerCase() + ", " + r.yes + " of " + N + ")";
        }).join("; ") + ".";
      }
      return line;
    }
    function answerSensitivity(q, audience, out) {
      out.understood = "Read as: does the answer depend on the settings?";
      var grid = settingsGrid(rule), ps = proteinSensitivity();
      var counts = ps.rows.map(function (r) { return r.consistent; });
      var lo = Math.min.apply(null, counts), hi = Math.max.apply(null, counts);
      out.lines.push("I re-ran the rule for \u201cmoves a lot\u201d with " + grid.length + " reasonable settings" + (rule.rule === "mad" ? " (2, 3 or 4 typical spreads above each run's median, and at least 1.25, 1.5 or 2 times that median)" : " (top 10%, 15% or 20%)") + ".");
      out.lines.push("The number of residues that move a lot in at least " + NEED + " of " + N + " runs " + (lo === hi ? "is " + lo + " under every setting" : "ranges from " + lo + " to " + hi) +
        (ps.always.length ? (lo === hi && ps.always.length === lo ? ", always the same residues: " : "; " + ps.always.length + " count under every setting: ") + stretchList(groupList(ps.always)) + "." : ", and none counts under every setting."));
      if (D.claim && D.claim.region) {
        var idx = positions(D.claim.region[0], D.claim.region[1]);
        out.lines.push("For the question at the top (" + dash(D.claim.region[0], D.claim.region[1]) + "): " + settingsLine(sensitivity(idx), audience).replace(/^It/, "the answer"));
      } else {
        out.lines.push("Ask about a stretch, such as \u201cis " + suggestedRegion() + " flexible?\u201d, to see whether its answer changes with the settings.");
      }
      if (audience !== "new") {
        out.lines.push(ps.rows.map(function (r) { return describeCell(r.cell) + (r.isDefault ? " (used here)" : "") + ": " + r.consistent; }).join("; ") + ". Re-checked in this page from the numbers shown.");
      }
      ps.always.length && groupList(ps.always).slice(0, 8).forEach(function (st) { out.links.push({ lo: st[0], hi: st[1], text: "Show " + span(st[0], st[1]) }); });
      out.command = D.hotspots_command;
      return out;
    }
    function groupList(indices) {
      var out = [];
      indices.forEach(function (i) {
        var last = out[out.length - 1];
        if (last && last[1] === i - 1 && D.residues[i].c === D.residues[i - 1].c && D.residues[i].n === D.residues[i - 1].n + 1) last[1] = i;
        else out.push([i, i]);
      });
      return out;
    }

    function answerRun(q, audience, out) {
      var picked = runByNumber(q);
      if (!picked) return noSuchRun(out, q);
      var run = picked.run, sm = D.summaries[run.id], c = D.counts;
      out.understood = "Read as: what would " + picked.note + " have told you on its own?";
      out.run = run.id;
      out.lines.push(run.label + ", analysed on its own, finds " + residueCount(sm.flexible_residues) + " that move a lot.");
      if (sm.missed_consistent === 0 && sm.extra_not_consistent === 0) {
        out.lines.push("That matches what the runs agree on as a group.");
      } else {
        out.lines.push("Compared with all " + N + " runs together, it would miss " + sm.missed_consistent + " of the " + c.consistent + " " + plural(c.consistent, "residue", "residues") +
          " that move a lot in at least " + NEED + " runs, and flag " + sm.extra_not_consistent + " that fewer than " + NEED + " runs agree on.");
      }
      if (audience !== "new") out.lines.push("Its threshold is " + fixed(sm.threshold_nm, 3) + " nm (median RMSF " + fixed(sm.median_rmsf_nm, 3) + " nm)." + (D.kind === "mdcath" ? " This is mdCATH replica " + run.id + "." : ""));
      out.command = D.hotspots_command;
      return out;
    }

    function answerOverview(q, audience, out) {
      var c = D.counts, st = D.stretches;
      out.understood = "Read as: which parts move a lot?";
      if (st.consistent.length) out.lines.push(residueCount(c.consistent) + " " + plural(c.consistent, "moves", "move") + " a lot in at least " + NEED + " of " + N + " runs: " + stretchList(st.consistent) + ".");
      else out.lines.push("No residue moves a lot in at least " + NEED + " of " + N + " runs.");
      if (st.some.length) out.lines.push(c.replica_dependent + " more " + plural(c.replica_dependent, "moves", "move") + " a lot in only some runs (" + dependentRange() + " of " + N + "): " + stretchList(st.some) + ". Be careful with those.");
      else out.lines.push("No residue is replica-dependent: the runs never split on one.");
      if (audience !== "new" && c.flagged_by_a_minority) out.lines.push(c.flagged_by_a_minority + " " + plural(c.flagged_by_a_minority, "residue moves", "residues move") + " a lot in just " + dash(1, FEW) + " of " + N + " " + plural(FEW, "run", "runs") + ", too few to count.");
      st.consistent.concat(st.some).slice(0, 10).forEach(function (s2) { out.links.push({ lo: s2[0], hi: s2[1], text: "Show " + span(s2[0], s2[1]) }); });
      out.command = D.hotspots_command;
      return out;
    }

    function answerDisagree(q, audience, out) {
      var c = D.counts, st = D.stretches;
      out.understood = "Read as: where do the runs disagree?";
      if (!st.some.length) {
        out.lines.push("The runs never split on a residue: each one moves a lot in at least " + NEED + " runs, or in at most " + FEW + ".");
        if (c.flagged_by_a_minority) out.lines.push(c.flagged_by_a_minority + " " + plural(c.flagged_by_a_minority, "residue moves", "residues move") + " a lot in just " + dash(1, FEW) + " of " + N + " " + plural(FEW, "run", "runs") + ", which is too few to count as a finding.");
      } else {
        if (st.contested.length) out.lines.push("The runs disagree most about " + stretchList(st.contested) + ": three or more residues in a row that only some runs say move a lot.");
        out.lines.push("In all, " + residueCount(c.replica_dependent) + " " + plural(c.replica_dependent, "is", "are") + " replica-dependent (each moves a lot in " + dependentRange() + " of " + N + " runs): " + stretchList(st.some) + ".");
        out.lines.push("Those are the parts where one run alone could have told you either answer.");
      }
      var seen = {};
      st.contested.concat(st.some).forEach(function (s2) {
        var key = s2[0] + ":" + s2[1];
        if (seen[key] || out.links.length >= 10) return;
        seen[key] = true;
        out.links.push({ lo: s2[0], hi: s2[1], text: "Show " + span(s2[0], s2[1]) });
      });
      out.command = D.hotspots_command;
      return out;
    }

    function answerExtreme(q, audience, out, most) {
      var k = Math.max(1, Math.min(q.top || 3, 10));
      var ranked = scoredIdx.slice().sort(function (i, j) { return most ? meanRmsf(j) - meanRmsf(i) : meanRmsf(i) - meanRmsf(j); }).slice(0, k);
      out.understood = "Read as: which " + plural(k, "residue", k + " residues") + " move" + (k === 1 ? "s" : "") + (most ? " most" : " least") + " on average?";
      var items = ranked.map(function (i) {
        return resName(i) + (audience !== "new" ? " (" + fixed(meanRmsf(i), 3) + " nm" + ", flagged in " + D.agreement[i] + " of " + N + ")" : " (moves a lot in " + D.agreement[i] + " of " + N + " runs)");
      });
      out.lines.push((most ? "On average over the " + N + " runs, the " + plural(k, "residue that moves most is ", "residues that move most are ") : "On average over the " + N + " runs, the " + plural(k, "stillest residue is ", "stillest residues are ")) + joinList(items) + ".");
      if (most) out.lines.push("Moving most on average is not the same as moving a lot in every run: the count says how well the runs agree.");
      out.lines.push("Chain ends are left out, because they are not scored.");
      ranked.forEach(function (i) { out.links.push({ lo: i, hi: i, text: "Show " + resName(i) }); });
      out.command = D.hotspots_command;
      return out;
    }

    function answerStill(q, audience, out) {
      var none = scoredIdx.filter(function (i) { return D.agreement[i] === 0; }).length, total = scoredIdx.length;
      out.understood = "Read as: which parts stay still?";
      out.lines.push(none + " of the " + total + " scored residues move a lot in no run" + (none / total > 0.5 ? ", so most of the protein stays fairly still in every run." : "."));
      out.lines.push("Ask “which parts move a lot?” to see the rest.");
      out.lines.push(D.kind === "mdcath"
        ? "This page measures how much each part moves. Whether a stretch keeps its folded shape is a separate check (concordance ask folded)."
        : "This page measures how much each part moves, not whether a stretch keeps its folded shape.");
      out.command = D.hotspots_command;
      return out;
    }

    function curveLine(curve, k) {
      var v = curve[k] || curve[String(k)];
      if (!v) return null;
      return (k === 1 ? "With a single run" : "With " + k + " runs") + ", you would reach the same answer as all " + N + " runs " + pct(v.same) + " of the time" +
        (v.opposite ? ", and the opposite answer " + pct(v.opposite) + " of the time" : "") + (v.undecided ? " (tied " + pct(v.undecided) + ")" : "") + ".";
    }

    function answerFewer(q, audience, out) {
      var k = q.runCount === null ? 1 : q.runCount;
      out.understood = "Read as: what if you had run " + (k === 1 ? "only one simulation" : k + " simulations") + "?";
      if (k >= N) {
        out.lines.push("This page already uses all " + N + " runs.");
        return out;
      }
      if (k < 1) { out.lines.push("You need at least one run to say anything."); return out; }
      var curve = D.claim && D.claim.agreement_curve;
      if (curve && curveLine(curve, k)) {
        out.lines.push("For the question at the top of the page: " + curveLine(curve, k));
        out.lines.push("That counts every possible choice of " + k + " " + plural(k, "run", "runs") + " out of the " + N + ".");
        return out;
      }
      if (k === 1) {
        var c = D.counts, worst = null;
        D.runs.forEach(function (run) {
          var sm = D.summaries[run.id], score = sm.missed_consistent + sm.extra_not_consistent;
          if (!worst || score > worst.score) worst = { run: run, sm: sm, score: score };
        });
        if (worst && worst.score > 0) {
          out.lines.push("It depends which run you happened to get. Analysed alone, " + worst.run.label + " would have missed " + worst.sm.missed_consistent + " of the " + c.consistent +
            " residues that move a lot in at least " + NEED + " runs, and flagged " + worst.sm.extra_not_consistent + " that fewer than " + NEED + " runs agree on.");
        } else {
          out.lines.push("Here, every run analysed alone picks out the same flexible residues.");
        }
        out.lines.push("Ask about a stretch, such as “is " + suggestedRegion() + " flexible?”, to see what each run would have said about it.");
        out.links = D.runs.map(function (run) { return { ask: run.label, text: run.label + " alone" }; });
        out.command = D.hotspots_command;
        return out;
      }
      out.lines.push("Ask about a stretch, such as “is " + suggestedRegion() + " flexible?”: the answer shows how many of the " + N + " runs agree, and how many would have said the opposite on their own.");
      return out;
    }

    function answerTrust(q, audience, out) {
      out.understood = "Read as: how far can I trust this?";
      out.lines.push("Agreement between runs is not proof that an answer is right: runs that start from the same structure can agree and still all be wrong. Disagreement, though, is a clear sign that one run is not enough to support a claim.");
      if (D.kind === "mdcath") out.lines.push("These mdCATH runs share one starting structure, so they are not fully independent, and " + N + " runs is a small sample.");
      else out.lines.push(N + " runs is a small sample, and the thresholds are stated conventions, not statistical guarantees.");
      if (D.check && D.check.agree === true) out.lines.push("The numbers themselves were cross-checked: MDAnalysis and an independent NumPy implementation agree to within " + D.check.max_abs_diff_nm.toExponential(1) + " nm.");
      else if (D.check && D.check.agree === null) out.lines.push("The movement numbers were read from gmx rmsf files, so they were computed in GROMACS rather than here.");
      if (audience !== "new") out.lines.push("Evidence passport " + D.passport.passport_id + ": the replay command below recreates this result and the same id.");
      out.command = D.passport.replay;
      return out;
    }

    function answerMethod(q, audience, out) {
      out.understood = "Read as: how was this worked out?";
      if (D.kind === "xvg") out.lines.push("The movement of each residue (its RMSF) was read from your gmx rmsf files, one per run.");
      else out.lines.push("For each run, MDAnalysis lines up every frame of the simulation and measures how far each residue wanders from its average position (its RMSF).");
      out.lines.push("Within each run, a residue counts as moving a lot when it moves clearly more than that run's typical residue. Then Concordance counts, residue by residue, how many runs agree.");
      out.lines.push("At least " + NEED + " of " + N + " runs agreeing counts as consistent; " + dependentRange() + " is replica-dependent. Every choice is written into the evidence passport.");
      if (audience !== "new") {
        out.lines.push("Rule: " + D.rule_text);
        if (D.kind !== "xvg") out.lines.push("Frames: all runs cut to the shortest (" + D.window.common_length + "), and the first " + D.window.frames_discarded + " skipped as equilibration.");
      }
      out.command = D.passport.replay;
      return out;
    }

    function plainRule() {
      if (rule.rule === "topk") return "it is among the " + pct(rule.top_fraction) + " of residues that move most in that run";
      return "it moves at least " + rule.min_ratio + " times as much as that run's typical (median) residue, and stands well clear of the usual spread";
    }

    var GLOSSARY = {
      dependent: function () { return ["Replica-dependent means the runs split: " + dependentRange() + " of the " + N + " runs say a part moves a lot, and the rest say it doesn't. It's a finding to be careful with, because a single run could have told you either answer."]; },
      consistent: function () { return ["Consistent across runs means at least " + NEED + " of the " + N + " runs agree that a part moves a lot. It's the strongest answer this page gives, and it means the finding holds up when the simulation is repeated, not that the simulation is right."]; },
      flexible: function (audience) {
        return ["In each run, a residue counts as flexible (it “moves a lot”) when " + plainRule() + "." + (rule.trim_termini ? " The first and last " + rule.trim_termini + " residues are not scored." : "")]
          .concat(audience !== "new" ? ["Exact rule: " + D.rule_text] : []);
      },
      passport: function () { return ["The evidence passport is a receipt for the answer. It records fingerprints of the data files, every setting, the result and the command that recreates it, so anyone can check the work. The same inputs always give the same passport id; this one is " + D.passport.passport_id + "."]; },
      threshold: function (audience) {
        var lines = ["Each run gets its own threshold for “moves a lot”, worked out from that run's typical movement, so a run that is a bit livelier overall isn't unfairly flagged everywhere."];
        if (audience !== "new") lines.push(D.runs.map(function (run) { return run.label + " " + fixed(D.summaries[run.id].threshold_nm, 3) + " nm"; }).join(", ") + ". Rule: " + D.rule_text);
        return lines;
      },
      ends: function () { return ["The first and last " + rule.trim_termini + " residues are not scored, because chain ends tend to flop about whatever the rest of the protein does, and would crowd out everything else. They are shown in light grey."]; },
      colours: function () { return ["Blue: moves a lot in at least " + NEED + " of " + N + " runs. Orange: moves a lot in only " + dependentRange() + " runs, so the runs disagree. Pale orange: moves a lot in just " + dash(1, FEW) + ". Grey: moves a lot in no run. Light grey: chain end, not scored. In the five small pictures, blue simply means “moved a lot in this run”."]; },
      rmsf: function () { return ["RMSF (root mean square fluctuation) measures how far each residue wanders from its average position during a run. A bigger number means that part moves more. It's in nanometres: a carbon–carbon bond is about 0.15 nm long."]; },
      units: function () { return ["Movement is given in nanometres (nm), a millionth of a millimetre. For scale, a carbon–carbon bond is about 0.15 nm long."]; },
      mdcath: function () { return ["mdCATH is a public dataset of simulations of thousands of protein domains, each simulated five times at each of five temperatures (Mirarchi, Giorgino & De Fabritiis, Scientific Data, 2024)."]; },
      page: function () { return ["Each small picture is the same protein, simulated separately " + (N === 5 ? "five" : N) + " times; blue marks the parts that moved a lot in that run. The large picture puts the runs together: blue parts moved a lot in at least " + NEED + " runs, orange parts in only some of them, so those are the claims to be careful with."]; },
      residue: function () { return ["A residue is one amino acid: one bead on the protein's chain. They are numbered along the chain; here they run from " + D.residues[0].n + " to " + D.residues[R - 1].n + "."]; },
      simulation: function () { return ["A molecular dynamics simulation is a computer film of a protein: it follows every atom, step by step, using physics. Repeated runs come out a bit differently, because tiny differences at the start grow over time, which is why runs are repeated and compared."]; },
      run: function () { return ["A run (also called a replica) is one complete simulation. This page compares " + N + " runs of the same protein under the same conditions."]; }
    };

    function suggestedRegion() {
      var st = D.stretches, pick = D.claim && D.claim.region ? D.claim.region : null;
      if (pick) return dash(pick[0], pick[1]);
      var list = st.contested.length ? st.contested : st.some.length ? st.some : st.consistent;
      if (list.length) return dash(D.residues[list[0][0]].n, D.residues[list[0][1]].n);
      var mid = scoredIdx[Math.floor(scoredIdx.length / 2)] || 0;
      return dash(D.residues[mid].n, D.residues[Math.min(mid + 3, R - 1)].n);
    }

    function suggestions() {
      return [
        "Which parts move a lot?",
        "Is " + suggestedRegion() + " flexible?",
        "Where do the runs disagree?",
        "What would one run have told me?",
        "Does it depend on the settings?",
        "What does replica-dependent mean?",
        "Can I trust this?"
      ];
    }

    function capabilities() {
      return "You can ask which parts move a lot, about any stretch (“is " + suggestedRegion() + " flexible?” or “does residue " + D.residues[scoredIdx[0] || 0].n + " move?”), where the runs disagree, what one run would have told you, about a single run (“run 2”), or what a word means (“what is RMSF?”).";
    }

    function ask(text, audience) {
      audience = audience || "new";
      var q = parse(text);
      var out = base(q, q.intent);
      switch (q.intent) {
        case "region": case "run-region": return answerRegion(q, audience, out);
        case "run": return answerRun(q, audience, out);
        case "overview": return answerOverview(q, audience, out);
        case "disagree": return answerDisagree(q, audience, out);
        case "most": return answerExtreme(q, audience, out, true);
        case "least": return answerExtreme(q, audience, out, false);
        case "still": return answerStill(q, audience, out);
        case "fewer-runs": return answerFewer(q, audience, out);
        case "trust": return answerTrust(q, audience, out);
        case "sensitivity": return answerSensitivity(q, audience, out);
        case "method": return answerMethod(q, audience, out);
        case "why-repeat":
          out.understood = "Read as: why run a simulation more than once?";
          out.lines.push("Every simulation comes out a bit differently, so a part that looks floppy in one run can look calm in another. Guidelines for simulation studies ask for at least three independent runs, compared with each other.");
          out.lines.push("This page does that comparison for you, residue by residue, and shows its working.");
          return out;
        case "count-runs":
          out.understood = "Read as: how many runs?";
          out.lines.push(D.kind === "xvg" ? "This page compares " + N + " runs, read from " + N + " gmx rmsf files." :
            "This page compares " + N + " runs of " + D.name + ", with " + D.window.frames_used_per_run + (D.kind === "mdcath" && !D.synthetic ? " ns" : " frames") + " of each run analysed.");
          return out;
        case "count-residues":
          out.understood = "Read as: how many residues?";
          out.lines.push(D.name + " has " + R + " residues here, numbered " + D.residues[0].n + " to " + D.residues[R - 1].n + ". " + scoredIdx.length + " are scored" + (rule.trim_termini ? "; the first and last " + rule.trim_termini + " are not." : "."));
          return out;
        case "hello":
          out.lines.push("Hello! Ask me about this protein in your own words. " + capabilities());
          return out;
        case "thanks":
          out.lines.push("You're welcome. Ask another question any time.");
          return out;
        case "help":
          out.understood = "Read as: what can I ask?";
          out.lines.push(capabilities());
          out.lines.push("Answers are worked out in your browser from this page's numbers. Nothing is sent anywhere, and it tells you when it doesn't understand.");
          return out;
        default:
          if (q.intent.indexOf("define:") === 0) {
            var key = q.intent.slice(7);
            var shown = { dependent: "replica-dependent", passport: "evidence passport", ends: "chain end", colours: "the colours", rmsf: "RMSF", units: "nm", mdcath: "mdCATH" }[key] || key;
            out.understood = key === "page" ? "Read as: what am I looking at?" : "Read as: what does \u201c" + shown + "\u201d mean?";
            out.lines = GLOSSARY[key](audience);
            return out;
          }
          out.unknown = true;
          out.lines.push("Sorry, I didn't understand that one. This box recognises a few kinds of question and answers them from this page's numbers; it doesn't guess.");
          out.lines.push(capabilities());
          if (audience !== "new") out.lines.push("For open-ended questions, Concordance's tools for Claude answer with an evidence passport (see the README).");
          return out;
      }
    }

    var gridCache = null;
    function gridFlags() {
      if (gridCache) return gridCache;
      var core = D.classes.map(function (c) { return c !== "u"; });
      gridCache = settingsGrid(rule).map(function (cell) {
        var flags = {};
        if (sameRule(cell, rule)) flags = D.flags;
        else D.runs.forEach(function (run) { flags[run.id] = flagResidues(D.rmsf[run.id], core, cell, referenceMask(D.residues, core, D.method.reference_residues)); });
        return { cell: cell, flags: flags, isDefault: sameRule(cell, rule) };
      });
      return gridCache;
    }
    function sensitivity(indices) {
      var grid = gridFlags(), base = verdictFor(indices);
      if (!base) return null;
      var rows = grid.map(function (g) { var v = verdictFor(indices, g.flags); return { cell: g.cell, isDefault: g.isDefault, verdict: v.verdict, yes: v.yes }; });
      return { total: rows.length, same: rows.filter(function (r) { return r.verdict === base.verdict; }).length, rows: rows, verdict: base.verdict };
    }
    function proteinSensitivity() {
      var grid = gridFlags(), always = null;
      var rows = grid.map(function (g) {
        var consistent = [];
        for (var i = 0; i < R; i++) {
          if (D.classes[i] === "u") continue;
          var a = 0;
          D.runs.forEach(function (run) { a += g.flags[run.id][i] ? 1 : 0; });
          if (a >= NEED) consistent.push(i);
        }
        always = always === null ? consistent : always.filter(function (i) { return consistent.indexOf(i) >= 0; });
        return { cell: g.cell, isDefault: g.isDefault, consistent: consistent.length };
      });
      return { rows: rows, always: always || [] };
    }

    // ---------------------------------------------------------------- comparing runs, and amino-acid types
    // How often two runs agree on "moves a lot or not", residue by residue (scored residues only).
    function pairAgreement() {
      var out = [];
      D.runs.forEach(function (a, i) {
        D.runs.forEach(function (b, j) {
          if (j <= i) return;
          var same = 0, differ = [];
          scoredIdx.forEach(function (k) { if (D.flags[a.id][k] === D.flags[b.id][k]) same++; else differ.push(k); });
          out.push({ a: i, b: j, same: same, total: scoredIdx.length, share: scoredIdx.length ? same / scoredIdx.length : 1,
            spearman: D.spearman[a.id + "-" + b.id] !== undefined ? D.spearman[a.id + "-" + b.id] : D.spearman[b.id + "-" + a.id],
            differ: differ, differStretches: groupList(differ) });
        });
      });
      return out;
    }
    // Mean movement by amino-acid type, over scored residues (needs residue names, i.e. a structure).
    function aminoAcidSummary() {
      var groups = {};
      scoredIdx.forEach(function (i) {
        var code = aaCode(residueName(D.residues[i]));
        if (!code) return;
        if (!groups[code]) groups[code] = { code: code, name: AA[code].name, group: AA[code].group, residues: [], total: 0, most: 0 };
        groups[code].residues.push(D.residues[i].n);
        groups[code].total += meanRmsf(i);
        if (D.agreement[i] >= NEED) groups[code].most++;
      });
      return Object.keys(groups).map(function (k) {
        var g = groups[k];
        return { code: g.code, name: g.name, group: g.group, count: g.residues.length, mean_rmsf_nm: g.total / g.residues.length, moving_in_most_runs: g.most, residues: g.residues };
      }).sort(function (x, y) { return y.mean_rmsf_nm - x.mean_rmsf_nm; });
    }

    // ---------------------------------------------------------------- the engine as tools for Claude
    // Small plain-data answers. Claude never computes a number itself: it asks these, and the page
    // checks every number in Claude's reply against what they returned (auditNumbers below).
    function verdictWords(v) { return v === "consistent across replicas" ? "consistent across runs" : v === "replica-dependent" ? "replica-dependent" : "not seen in most runs"; }
    function stretchText(st) { return dash(D.residues[st[0]].n, D.residues[st[1]].n); }
    var engine = {
      checkRegion: function (start, end, run) {
        var a = Math.round(Number(start)), b = Math.round(Number(end === undefined || end === null ? start : end));
        if (!isFinite(a) || !isFinite(b)) throw new Error("start and end must be residue numbers");
        if (a > b) { var t = a; a = b; b = t; }
        var idx = positions(a, b);
        if (!idx.length) throw new Error("No residues numbered " + a + "-" + b + "; this protein runs from " + D.residues[0].n + " to " + D.residues[R - 1].n + ".");
        var v = verdictFor(idx);
        if (!v) return { region: a + "-" + b, scored: false, note: "Only chain-end residues, which are not scored." };
        var sv = sensitivity(idx);
        var out = {
          region: dash(D.residues[idx[0]].n, D.residues[idx[idx.length - 1]].n), residues_scored: v.scored, total_runs: N,
          runs_saying_it_moves_a_lot: v.yes, verdict: verdictWords(v.verdict),
          runs_that_alone_would_give_the_opposite_answer: v.risk,
          per_run: D.runs.map(function (r, k) { return { run: k + 1, says_moves_a_lot: v.answers[r.id], residues_moving_a_lot: v.flagged[r.id] }; }),
          same_verdict_under_settings: sv.same + " of " + sv.total
        };
        if (run !== undefined && run !== null && run !== "") {
          var k = Math.round(Number(run));
          if (!(k >= 1 && k <= N)) throw new Error("Runs are numbered 1 to " + N + ".");
          out.asked_about_run = { run: k, says_moves_a_lot: v.answers[D.runs[k - 1].id], residues_moving_a_lot: v.flagged[D.runs[k - 1].id], of: v.scored };
        }
        out._explore = { lo: idx[0], hi: idx[idx.length - 1], idx: idx, run: out.asked_about_run ? D.runs[out.asked_about_run.run - 1].id : null };
        return out;
      },
      overview: function () {
        var c = D.counts, st = D.stretches;
        return {
          protein: D.name, total_runs: N, residues: dash(D.residues[0].n, D.residues[R - 1].n), scored_residues: c.scored_residues,
          consistent_rule: "moves a lot in at least " + NEED + " of " + N + " runs", consistent_residues: c.consistent, consistent_stretches: st.consistent.map(stretchText),
          replica_dependent_rule: "moves a lot in " + dash(FEW + 1, NEED - 1) + " of " + N + " runs", replica_dependent_residues: c.replica_dependent, replica_dependent_stretches: st.some.map(stretchText),
          longest_disagreements: st.contested.map(stretchText), residues_moving_a_lot_in_too_few_runs_to_count: c.flagged_by_a_minority
        };
      },
      runAlone: function (run) {
        var k = Math.round(Number(run));
        if (!(k >= 1 && k <= N)) throw new Error("Runs are numbered 1 to " + N + ".");
        var sm = D.summaries[D.runs[k - 1].id];
        return { run: k, residues_moving_a_lot_in_this_run: sm.flexible_residues, consistent_residues_it_would_miss: sm.missed_consistent,
          residues_it_would_flag_that_most_runs_do_not: sm.extra_not_consistent, threshold_nm: sm.threshold_nm, median_rmsf_nm: sm.median_rmsf_nm, _run: D.runs[k - 1].id };
      },
      ranking: function (direction, count) {
        var k = Math.max(1, Math.min(Math.round(Number(count) || 5), 10)), most = direction !== "least";
        return scoredIdx.slice().sort(function (i, j) { return most ? meanRmsf(j) - meanRmsf(i) : meanRmsf(i) - meanRmsf(j); }).slice(0, k).map(function (i) {
          return { residue: D.residues[i].l, number: D.residues[i].n, mean_rmsf_nm: Math.round(meanRmsf(i) * 1000) / 1000, runs_saying_it_moves_a_lot: D.agreement[i] };
        });
      },
      thresholdCheck: function (start, end) {
        if (start === undefined || start === null || start === "") {
          var ps = proteinSensitivity();
          return { settings: ps.rows.map(function (r) { return { setting: describeCell(r.cell), consistent_residues: r.consistent, is_default: r.isDefault }; }),
            residues_consistent_under_every_setting: ps.always.length, stretches_consistent_under_every_setting: groupList(ps.always).map(stretchText) };
        }
        var a = Math.round(Number(start)), b = Math.round(Number(end === undefined || end === null ? start : end));
        var idx = positions(Math.min(a, b), Math.max(a, b));
        var sv = idx.length ? sensitivity(idx) : null;
        if (!sv) throw new Error("No scored residues numbered " + a + "-" + b + ".");
        return { region: dash(Math.min(a, b), Math.max(a, b)), default_verdict: verdictWords(sv.verdict), same_verdict_under: sv.same + " of " + sv.total + " settings",
          settings: sv.rows.map(function (r) { return { setting: describeCell(r.cell), verdict: verdictWords(r.verdict), runs_saying_it_moves_a_lot: r.yes }; }) };
      },
      fewerRuns: function () {
        var curve = D.claim && D.claim.agreement_curve;
        if (!curve) return { note: "Only available for the region asked about at the top of the page; use check_region and run_alone instead." };
        var out = { region: D.claim.region ? dash(D.claim.region[0], D.claim.region[1]) : null };
        Object.keys(curve).forEach(function (k) { out["with_" + k + "_runs"] = { same_answer_as_all_runs_percent: Math.round(curve[k].same * 100), opposite_percent: Math.round(curve[k].opposite * 100), tied_percent: Math.round(curve[k].undecided * 100) }; });
        return out;
      }
    };

    // Every number Claude writes must appear in what the engine returned (or in the question, or as
    // a percentage or unit conversion of one). Returns {checked, unsupported, spans}.
    function collectNumbers(value, into) {
      if (value === null || value === undefined) return into;
      if (typeof value === "number") { into.push(value); return into; }
      if (typeof value === "boolean") return into;
      if (typeof value === "string") { (value.match(/\d+(?:\.\d+)?/g) || []).forEach(function (x) { into.push(Number(x)); }); return into; }
      if (Array.isArray(value)) { value.forEach(function (v) { collectNumbers(v, into); }); return into; }
      Object.keys(value).forEach(function (k) { if (k.charAt(0) !== "_") { collectNumbers(k, into); collectNumbers(value[k], into); } });
      return into;
    }
    function auditNumbers(answer, results, question) {
      var allowed = collectNumbers(results, []).concat(collectNumbers(String(question || ""), []));
      allowed.push(N, NEED, D.residues[0].n, D.residues[R - 1].n, 100);
      for (var k = 0; k <= N; k++) allowed.push(Math.round(k / N * 100));  // "60% of runs"
      var extended = [];
      allowed.forEach(function (a) { extended.push(a, a * 100, a * 10, a / 10); });
      var spans = [], unsupported = [], checked = 0, rx = /(^|[^\w.])(\d+(?:\.\d+)?)(?![\d.]|[A-Z]{2})/g, m;
      var text = String(answer || "");
      while ((m = rx.exec(text)) !== null) {
        var raw = m[2], value = Number(raw), decimals = (raw.split(".")[1] || "").length;
        var ok = extended.some(function (a) { return Math.abs(Number(a.toFixed(decimals)) - value) < 1e-9 || Math.abs(a - value) < 1e-9; });
        var start = m.index + m[1].length;
        spans.push({ start: start, end: start + raw.length, value: raw, ok: ok });
        checked++;
        if (!ok) unsupported.push(raw);
      }
      return { checked: checked, unsupported: unsupported, spans: spans };
    }

    return { pairAgreement: pairAgreement, aminoAcidSummary: aminoAcidSummary, engine: engine, auditNumbers: auditNumbers, verdictFor: verdictFor, regionVerdict: regionVerdict, sensitivity: sensitivity, proteinSensitivity: proteinSensitivity, positions: positions, verdictName: verdictName, riskSentence: riskSentence,
      parse: parse, ask: ask, suggestions: suggestions };
  }

  return { create: create, aaGroup: aaGroup, AA: AA, parseRmsfText: parseRmsfText, disorderHint: disorderHint, parsePdbResidues: parsePdbResidues, buildData: buildData,
    flagResidues: flagResidues, thresholdFor: thresholdFor, coreMask: coreMask, settingsGrid: settingsGrid };
})();
if (typeof module !== "undefined" && module.exports) module.exports = ConcordanceLogic;
