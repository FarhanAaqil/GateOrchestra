/*
 * reports/deck/build_deck.js
 * Builds reports/deck/GateOrchestra_final.pptx from the real benchmark results.
 *
 *   npm install pptxgenjs@3.12.0      (once, anywhere on NODE_PATH)
 *   node reports/deck/build_deck.js [analysis.json] [master_results.json] [out.pptx]
 *
 * Every number on the slides is read from reports/final_analysis.json and
 * results/real/final/master_results.json; nothing is typed in by hand.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const ANALYSIS = process.argv[2] || path.join(ROOT, "reports", "final_analysis.json");
const MASTER = process.argv[3] || path.join(ROOT, "results", "real", "final", "master_results.json");
const OUT = process.argv[4] || path.join(__dirname, "GateOrchestra_final.pptx");
const NARRATIVE = path.join(__dirname, "narrative.json");

const a = JSON.parse(fs.readFileSync(ANALYSIS, "utf8"));
const m = JSON.parse(fs.readFileSync(MASTER, "utf8"));
const story = fs.existsSync(NARRATIVE) ? JSON.parse(fs.readFileSync(NARRATIVE, "utf8")) : {};

// Dataset sizes come from the split files, so the deck can never disagree with the data.
const countRows = (split) =>
  fs.readFileSync(path.join(ROOT, "dataset", "masbench_mini", split, `${split}.jsonl`), "utf8")
    .split("\n").filter(Boolean).length;
const SIZE = { train: countRows("train"), val: countRows("val"), test: countRows("test") };
SIZE.total = SIZE.train + SIZE.val + SIZE.test;

// The console's palette: navy dominates, ice for content, ember as the accent.
const THEME = {
  name: "GateOrchestra",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1B1F", lt1: "FFFFFF", dk2: "1B263B", lt2: "F1FAFF",
    accent1: "FF6B35", accent2: "A8DBFF", accent3: "8B1538", accent4: "FF8C42",
    accent5: "00E5FF", accent6: "FFF56B", hlink: "8B1538", folHlink: "1B263B",
  },
};
const HEX = { ink: "1B1B1F", navy: "1B263B", ice: "F1FAFF", sky: "A8DBFF", ember: "FF6B35",
  wine: "8B1538", tangerine: "FF8C42", cyan: "00E5FF", lemon: "FFF56B", grid: "D6E6F2", muted: "5A6B82" };
const METHOD_HEX = { "CoT-SC-only": HEX.sky, "Always-MAS": HEX.wine, RandomGate: HEX.navy,
  RuleBasedGate: HEX.tangerine, GateOrchestra: HEX.ember };

const pct = (x, d = 1) => (x == null ? "n/a" : `${(100 * x).toFixed(d)}%`);
const num = (x) => Math.round(x).toLocaleString("en-US");
const row = (method) => m.aggregated.find((r) => r.method === method);

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.author = "Farhan Aaqil Durrani";
pres.title = "GateOrchestra: final results";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// Layouts
const FOOT = { x: 0.5, y: 5.2, w: 9, h: 0.25, fontSize: 9, color: C.text2 };
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.5, w: 8.8, h: 1.2,
      fontSize: 44, bold: true, color: C.background1, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.8, w: 8.8, h: 1.4,
      fontSize: 18, color: C.accent2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.ice },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9, h: 0.75,
      fontSize: 30, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "GateOrchestra · final benchmark", options: { ...FOOT } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.25, fontSize: 9, color: HEX.muted, align: "right" },
});

function content(title, section) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  return s;
}
function text(s, t, o) {
  s.addText(t, { isTextBox: true, fontSize: 14, color: C.text1, valign: "top", margin: 0, ...o });
}
function card(s, x, y, w, h, fill = "FFFFFF") {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12,
    fill: { color: fill }, line: { color: HEX.grid, width: 0.75 } });
}
function stat(s, x, y, w, value, label, color = HEX.ember) {
  card(s, x, y, w, 1.45);
  text(s, value, { x: x + 0.25, y: y + 0.18, w: w - 0.5, h: 0.7, fontSize: 36, bold: true,
    color, fontFace: THEME.headFontFace });
  text(s, label, { x: x + 0.25, y: y + 0.9, w: w - 0.5, h: 0.45, fontSize: 12, color: HEX.muted });
}
const AXIS = { catAxisLabelColor: HEX.navy, valAxisLabelColor: HEX.navy, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 11, valAxisLabelFontSize: 10,
  dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.ink,
  titleFontFace: "+mn-lt", titleColor: HEX.navy, titleFontSize: 13,
  valGridLine: { color: HEX.grid, size: 0.5 }, catGridLine: { style: "none" } };

// 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
s.addText("GateOrchestra", { placeholder: "title" });
s.addText(story.subtitle || "Can a cheap probe tell when a question needs a multi-agent system?",
  { placeholder: "body" });
text(s, "AIML Capstone · November 2026 · Farhan Aaqil Durrani and team", {
  x: 0.6, y: 4.7, w: 8.8, h: 0.35, fontSize: 12, color: HEX.sky });
s.addNotes("GateOrchestra decides, before spending tokens on a multi-agent system, whether a question needs one.");

// 2. Pipeline
s = content("A probe, a gate, and a budget", "Introduction");
const steps = [
  ["CoT-SC probe", "3 samples vote;\nagreement = confidence", HEX.sky, HEX.ink],
  ["8 features", "consistency, tokens,\nentities, clauses ...", "FFFFFF", HEX.ink],
  ["Learned gate", "STOP or ESCALATE", HEX.cyan, HEX.ink],
  ["Multi-agent system", "ReAct · Debate · Reflexion\ncapped at k x probe tokens", HEX.wine, "FFFFFF"],
];
steps.forEach(([h, d, fill, fg], i) => {
  const x = 0.5 + i * 2.32;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.55, w: 2.0, h: 1.75, rectRadius: 0.15,
    fill: { color: fill }, line: { color: HEX.navy, width: 0.75 }, objectName: `step-${i + 1}` });
  text(s, `${i + 1}`, { x: x + 0.18, y: 1.68, w: 0.5, h: 0.4, fontSize: 18, bold: true, color: fg });
  text(s, h, { x: x + 0.18, y: 2.08, w: 1.68, h: 0.4, fontSize: 15, bold: true, color: fg });
  text(s, d, { x: x + 0.18, y: 2.5, w: 1.7, h: 0.75, fontSize: 11, color: fg });
  if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.02, y: 2.3, w: 0.28, h: 0.28,
    fill: { color: HEX.navy }, line: { color: HEX.navy } });
});
text(s, "STOP returns the probe's answer at probe cost. ESCALATE also pays for the MAS run, so the gate "
  + "only saves tokens if it stops on questions the probe already gets right.",
  { x: 0.5, y: 3.65, w: 9, h: 0.9, fontSize: 14, color: HEX.navy });
s.addNotes("The MAS budget is k times the probe's own token count, so every escalation is capped. "
  + "A learned LinUCB router picks the sub-agent for each escalated question.");

// 3. Questions
s = content("What we set out to answer", "Introduction");
const qs = [
  ["RQ1", "Does a learned gate save tokens without losing accuracy against Always-MAS?"],
  ["RQ2", "How does the budget multiplier k trade accuracy for cost?"],
  ["RQ3", "Which probe features tell the gate a question needs MAS?"],
  ["H1", "MAS is more accurate than CoT-SC on hard multi-hop questions, at a higher cost."],
];
qs.forEach(([k, q], i) => {
  const x = 0.5 + (i % 2) * 4.6;
  const y = 1.3 + Math.floor(i / 2) * 1.85;
  card(s, x, y, 4.4, 1.6);
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: y + 0.3, w: 0.8, h: 0.8,
    fill: { color: k === "H1" ? HEX.ember : HEX.navy }, line: { color: HEX.navy, width: 0 } });
  text(s, k, { x: x + 0.25, y: y + 0.52, w: 0.8, h: 0.4, fontSize: 15, bold: true, color: "FFFFFF", align: "center" });
  text(s, q, { x: x + 1.25, y: y + 0.3, w: 2.95, h: 1.15, fontSize: 14, color: HEX.ink });
});

// 4. Setup
pres.addSection({ title: "Method" });
s = content("How we measured it", "Method");
const p = a.premise;
stat(s, 0.5, 1.3, 2.1, `${SIZE.total}`, `tasks: ${SIZE.train} train,\n${SIZE.val} val, ${SIZE.test} test`, HEX.navy);
stat(s, 2.8, 1.3, 2.1, `${m.seeds.length}`, `seeds, each on all\n${SIZE.test} test tasks`, HEX.navy);
stat(s, 5.1, 1.3, 2.1, "3B", "Qwen2.5-3B on a\nlocal GPU (Ollama)", HEX.navy);
stat(s, 7.4, 1.3, 2.1, `${p ? p.n_runs : "?"}`, "graded runs per system\nfor calibration", HEX.ember);
text(s, [
  { text: "Every token counted: ", options: { bold: true } },
  { text: "provider-reported prompt + completion tokens, budgets enforced before each call.", options: { breakLine: true } },
  { text: "Same budget for everyone: ", options: { bold: true } },
  { text: "Always-MAS and escalations both get k x the paired probe's tokens.", options: { breakLine: true } },
  { text: "Honest statistics: ", options: { bold: true } },
  { text: "Wilson 95% intervals, exact McNemar tests on paired tasks, gate calibrated on validation only." },
], { x: 0.5, y: 3.05, w: 9, h: 1.8, fontSize: 14, color: HEX.ink, paraSpaceAfter: 6 });

// 5. Premise
pres.addSection({ title: "Results" });
s = content(story.premise_title || "The premise: does MAS beat CoT-SC?", "Results");
if (p) {
  s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: ["CoT-SC", "MAS"],
    values: [100 * p.cot_sc.accuracy, 100 * p.mas.accuracy] }], {
    x: 0.5, y: 1.2, w: 4.6, h: 3.8, barDir: "col", chartColors: [HEX.sky, HEX.wine],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"",
    valAxisMinVal: 0, valAxisMaxVal: 100, showLegend: false, showTitle: true,
    title: `Accuracy over ${p.n_runs} runs each`, ...AXIS });
  stat(s, 5.5, 1.2, 4.0, `${p.accuracy_gain_pts >= 0 ? "+" : ""}${p.accuracy_gain_pts.toFixed(1)} pts`,
    "MAS minus CoT-SC accuracy", p.accuracy_gain_pts >= 0 ? HEX.ember : HEX.wine);
  stat(s, 5.5, 2.85, 4.0, `${p.token_ratio.toFixed(2)}x`, "MAS tokens per run vs CoT-SC", HEX.navy);
  text(s, `Per task: MAS better on ${p.tasks_mas_better}, worse on ${p.tasks_cot_better}, `
    + `tied on ${p.tasks_tied} (sign test p = ${p.sign_test_p.toFixed(3)}).`,
    { x: 5.5, y: 4.45, w: 4.0, h: 0.6, fontSize: 12, color: HEX.muted });
  s.addNotes(story.premise_note || "This is H1, tested on every calibration run.");
}

// 6. Test results
s = content(story.results_title || "Test results across seeds", "Results");
const order = ["CoT-SC-only", "Always-MAS", "RandomGate", "RuleBasedGate", "GateOrchestra"];
const rows = order.map(row).filter(Boolean);
s.addChart(pres.charts.BAR, [{ name: "Accuracy", labels: rows.map((r) => r.method),
  values: rows.map((r) => r.accuracy_mean) }], {
  x: 0.5, y: 1.15, w: 4.5, h: 3.9, barDir: "bar", chartColors: rows.map((r) => METHOD_HEX[r.method]),
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", valAxisMinVal: 0,
  valAxisMaxVal: 120, valAxisMajorUnit: 20, showLegend: false, showTitle: true,
  title: "Accuracy (mean over seeds)", ...AXIS });
const header = ["Method", "95% CI", "Tokens", "Escalate"].map((t) => ({ text: t,
  options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy }, fontSize: 11 } }));
s.addTable([header, ...rows.map((r) => [
  { text: r.method, options: { bold: r.method === "GateOrchestra" } },
  `${r.accuracy_pooled_ci95[0].toFixed(0)}-${r.accuracy_pooled_ci95[1].toFixed(0)}%`,
  num(r.avg_tokens_mean),
  r.escalate_rate_mean == null ? "-" : pct(r.escalate_rate_mean, 0),
])], { x: 5.3, y: 1.25, w: 4.2, colW: [1.55, 0.95, 0.85, 0.85], fontSize: 11, color: HEX.ink,
  border: { type: "solid", color: HEX.grid, pt: 0.75 }, fill: { color: "FFFFFF" }, rowH: 0.36 });
const vsMas = (m.pooled_paired_tests || []).find((t) => t.method === "GateOrchestra" && t.reference === "Always-MAS");
if (vsMas) text(s, `GateOrchestra vs Always-MAS: ${vsMas.method_only_correct} vs ${vsMas.reference_only_correct} `
  + `discordant tasks, McNemar p = ${vsMas.p_value.toFixed(3)} (pooled over seeds).`,
  { x: 5.3, y: 3.65, w: 4.2, h: 0.8, fontSize: 12, color: HEX.muted });
s.addNotes(story.results_note || "Pooled intervals reuse the same test tasks across seeds, so they are optimistic.");

// 7. Oracle
s = content("The ceiling: a perfect gate", "Results");
const o = a.oracle_gate_calibration;
const gate = row("GateOrchestra");
if (o) {
  stat(s, 0.5, 1.3, 2.85, pct(o.accuracy), "oracle accuracy\n(train + validation)", HEX.ember);
  stat(s, 3.575, 1.3, 2.85, pct(o.escalate_rate), "of questions worth escalating", HEX.navy);
  stat(s, 6.65, 1.3, 2.85, `${o.token_savings_vs_mas_pct.toFixed(0)}%`, "fewer tokens than Always-MAS", HEX.wine);
  text(s, story.oracle_text || "An oracle escalates exactly when MAS is right and CoT-SC is wrong. "
    + "No gate can beat it with these two systems, so it bounds what any learned gate could gain.",
    { x: 0.5, y: 3.1, w: 9, h: 1.0, fontSize: 15, color: HEX.navy });
  if (gate) text(s, `The learned gate on test: ${gate.accuracy_mean.toFixed(1)}% accuracy, `
    + `${pct(gate.escalate_rate_mean, 0)} escalated, ${num(gate.avg_tokens_mean)} tokens per task.`,
    { x: 0.5, y: 4.2, w: 9, h: 0.6, fontSize: 13, color: HEX.muted });
}

// 8. Where MAS helps
s = content("Where MAS helps, and where it hurts", "Results");
if (a.axis) {
  const src = a.axis.source;
  s.addChart(pres.charts.BAR, [{ name: "MAS gain", labels: src.map((r) => r.key.replace(/_/g, " ")),
    values: src.map((r) => r.mas_gain_pts) }], {
    x: 0.5, y: 1.15, w: 5.2, h: 3.9, barDir: "bar", chartColors: [HEX.ember],
    invertedColors: [HEX.wine], showValue: true, dataLabelPosition: "outEnd", catAxisLabelPos: "low",
    dataLabelFormatCode: "+0;-0;0", showLegend: false, showTitle: true,
    title: "MAS minus CoT-SC accuracy by source (pts)", ...AXIS });
  const cells = [...a.axis.depth_parallel].filter((c) => c.n_tasks >= 3)
    .sort((x, y) => y.mas_gain_pts - x.mas_gain_pts);
  const pick = [...cells.slice(0, 3), ...cells.slice(-2)].filter((c, i, arr) => arr.indexOf(c) === i);
  const hdr = ["Depth x Parallel", "Tasks", "Gain"].map((t) => ({ text: t,
    options: { bold: true, color: "FFFFFF", fill: { color: HEX.navy }, fontSize: 11 } }));
  s.addTable([hdr, ...pick.map((c) => [c.key, `${c.n_tasks}`,
    { text: `${c.mas_gain_pts >= 0 ? "+" : ""}${c.mas_gain_pts.toFixed(0)} pts`,
      options: { color: c.mas_gain_pts >= 0 ? "1F7A3A" : HEX.wine, bold: true } }])],
  { x: 6.0, y: 1.25, w: 3.5, colW: [1.6, 0.8, 1.1], fontSize: 11, color: HEX.ink,
    border: { type: "solid", color: HEX.grid, pt: 0.75 }, fill: { color: "FFFFFF" }, rowH: 0.36 });
  text(s, "Cells with at least 3 tasks: the largest gains and losses.",
    { x: 6.0, y: 3.75, w: 3.5, h: 0.5, fontSize: 11, color: HEX.muted });
}

// 9. Ablation
s = content("What the gate relies on", "Results");
const ab = (a.feature_ablation || {}).rows || [];
const cls = ab.some((r) => r.classifier === "logreg") ? "logreg" : (ab[0] || {}).classifier;
if (cls) {
  const mine = ab.filter((r) => r.classifier === cls);
  const base = mine.find((r) => r.dropped === "none");
  const drops = mine.filter((r) => r.dropped !== "none");
  const deltas = drops.map((r) => 100 * (r.escalation_rate - base.escalation_rate));
  const lo = Math.floor(Math.min(0, ...deltas)) - 1;
  const hi = Math.ceil(Math.max(0, ...deltas)) + 1;
  s.addChart(pres.charts.BAR, [{ name: "Change", labels: drops.map((r) => r.dropped.replace(/_/g, " ")),
    values: deltas }], {
    x: 0.5, y: 1.15, w: 5.6, h: 3.9, barDir: "bar", chartColors: [HEX.cyan], invertedColors: [HEX.wine],
    showValue: true, catAxisLabelPos: "low", dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0;0.0",
    valAxisMinVal: lo, valAxisMaxVal: hi, showLegend: false, showTitle: true,
    title: `Escalation-rate change (pts) with one feature removed (${cls})`, ...AXIS });
  stat(s, 6.4, 1.2, 3.1, pct(base.accuracy), `${cls} gate with all 8 features (validation)`, HEX.navy);
  text(s, story.ablation_text || "Each feature is set to a constant in training and validation, "
    + "then the gate is retrained on the real labels.",
    { x: 6.4, y: 2.85, w: 3.1, h: 2.0, fontSize: 13, color: HEX.ink });
}

// 10. Router + budget
s = content("The learned router and the budget cap", "Results");
const calibRouting = (a.routing || {})[String(a.calibration_seed)] || {};
const train = calibRouting.train || {};
const arms = Object.keys(train);
if (arms.length) {
  s.addChart(pres.charts.DOUGHNUT, [{ name: "Runs", labels: arms, values: arms.map((k) => train[k].n) }], {
    x: 0.5, y: 1.15, w: 4.2, h: 3.9, ...AXIS, chartColors: [HEX.navy, HEX.ember, HEX.wine],
    showPercent: true, showValue: false, showLegend: true, legendPos: "b", legendFontFace: "+mn-lt",
    legendColor: HEX.navy, showTitle: true, title: "Sub-agent picked on train runs",
    dataLabelColor: "FFFFFF", dataLabelFontSize: 12 });
  text(s, arms.map((k) => ({ text: `${k}: ${train[k].n} run${train[k].n === 1 ? "" : "s"}, ${pct(train[k].accuracy, 0)} correct`,
    options: { bullet: true, breakLine: true } })),
  { x: 5.0, y: 1.3, w: 4.5, h: 1.4, fontSize: 14, color: HEX.ink, paraSpaceAfter: 4 });
}
if (a.budget) text(s, [
  { text: "Budget cap: ", options: { bold: true } },
  { text: `${a.budget.over_budget} of ${a.budget.n} MAS runs went over k x probe tokens, `
    + `by at most ${a.budget.max_overshoot_pct.toFixed(1)}%. The cap is checked on a chars/4 prompt estimate.` },
], { x: 5.0, y: 3.0, w: 4.5, h: 1.4, fontSize: 13, color: HEX.navy });

// 11. Power
pres.addSection({ title: "Conclusion" });
s = content(`How much ${SIZE.test} test tasks can tell us`, "Conclusion");
const pw = (a.power || []).find((r) => r.method === "GateOrchestra");
if (pw && pw.n_pairs_pooled) {
  const mde = (x) => (x == null ? "n/a" : `${x.toFixed(0)} pts`);
  const perSeed = (a.power || []).map((r) => r.mde_pts_per_seed).filter((x) => x != null);
  const perSeedText = perSeed.length
    ? `${Math.min(...perSeed).toFixed(0)}-${Math.max(...perSeed).toFixed(0)} pts`
    : "n/a";
  stat(s, 0.5, 1.3, 2.85, perSeedText,
    "smallest gap one seed can detect\n(80% power, one seed)", HEX.wine);
  stat(s, 3.575, 1.3, 2.85, mde(pw.mde_pts_pooled), "same, pooled over seeds\n(optimistic)", HEX.navy);
  stat(s, 6.65, 1.3, 2.85, `${pw.observed_diff_pts >= 0 ? "+" : ""}${pw.observed_diff_pts.toFixed(1)} pts`,
    "observed: GateOrchestra minus Always-MAS", HEX.ember);
}
text(s, story.limits || [
  { text: "One 3B model; a larger model may change the premise.", options: { bullet: true, breakLine: true } },
  { text: "The 64 test tasks added late are easier than the original 32 and were reviewed by the assistant, not a second person.", options: { bullet: true, breakLine: true } },
  { text: "Seeds share test tasks, so pooled tests overstate the evidence.", options: { bullet: true } },
], { x: 0.5, y: 3.1, w: 9, h: 1.8, fontSize: 14, color: HEX.ink, paraSpaceAfter: 6 });

// 12. Conclusion
s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Conclusion" });
s.addText(story.conclusion_title || "What we found", { placeholder: "title" });
s.addText(story.conclusion || "See reports/research_report.md for the full write-up.", { placeholder: "body" });
s.addNotes(story.conclusion_note || "All numbers come from results/real/final and reports/final_analysis.json.");

pres.writeFile({ fileName: OUT }).then(async (file) => {
  const skill = process.env.PPTX_SKILL_DIR;
  if (skill) {
    const { applyTheme } = require(path.join(skill, "scripts", "apply_theme.js"));
    await applyTheme(file, THEME);
  }
  console.log(`[OK] wrote ${file}`);
});
