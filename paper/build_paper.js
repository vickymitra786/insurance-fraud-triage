// Builds paper/fraud_triage.docx from ../results/results.json and paper/figures/*.png.
// Author: Vivek Mitra. Every number in the text is read from the results file.
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ImageRun, Footer, PageNumber, BorderStyle, ShadingType } = require("docx");

const R = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "results", "results.json"), "utf8"));
const LR = "logistic_regression", GB = "gradient_boosting";
if (R.decision.primary !== GB) throw new Error("The text below assumes gradient boosting passed the keep rule. Rewrite Sections 4-5.");

const f3 = v => v.toFixed(3), f2 = v => v.toFixed(2);
const p1 = v => (v * 100).toFixed(1) + "%", p0 = v => Math.round(v * 100) + "%";
const int = v => Math.round(v).toLocaleString("en-US");
const usd = v => (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString("en-US");                          // assumption inputs: exact
const usd100 = v => (v < 0 ? "−$" : "$") + (Math.round(Math.abs(v) / 100) * 100).toLocaleString("en-US");  // computed results: nearest $100
const ci = (a, f) => `[${f(a[0])}, ${f(a[1])}]`;
const pts = a => `[${(a[0] * 100).toFixed(1)}, ${(a[1] * 100).toFixed(1)}] pts`;

const P = R.test.point, CI = R.test.ci95, D = R.test.difference_gb_minus_lr, C = R.cost, A = C.assumptions;
const CAL = R.calibration, K = R.checks, FX = R.facts, B = R.tuning.best;
const N = R.split.test_claims, NF = R.split.test_frauds;
const hit = (m, k) => P[m][`capture_top${k}`] * NF / Math.ceil(k / 100 * N);
const yr = Object.fromEntries(R.data.by_year.map(y => [y.year, y]));
const gD = C[GB].depth_chosen_on_1995, lD = C[LR].depth_chosen_on_1995;
const set = name => ({ all_fields: "all fields", no_calendar: "no calendar fields", no_calendar_no_rep: "no calendar fields and no claims representative" })[name];

// ---------- docx helpers ----------
const FONT = "Calibri";
const run = (t, o = {}) => new TextRun({ text: t, font: FONT, size: 22, ...o });
const kids = parts => (Array.isArray(parts) ? parts : [parts]).map(x => (typeof x === "string" ? run(x) : x));
const para = parts => new Paragraph({ children: kids(parts), spacing: { after: 120, line: 276 }, alignment: AlignmentType.LEFT });
const b = t => run(t, { bold: true }), i = t => run(t, { italics: true });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [run(t, { bold: true, size: 28, color: "1F4E8C" })] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 80 }, children: [run(t, { bold: true, size: 24 })] });
const bullet = parts => new Paragraph({ bullet: { level: 0 }, spacing: { after: 60 }, children: kids(parts) });
const caption = t => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [run(t, { italics: true, size: 19 })] });
const figure = (file, w, h, cap) => [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80 }, keepNext: true,
  children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(__dirname, "figures", file)), transformation: { width: w, height: h } })] }), caption(cap)];
const line = { style: BorderStyle.SINGLE, size: 4, color: "BFC5CE" }, none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(head, rows, widths) {
  const tot = widths.reduce((a, x) => a + x, 0);
  const cell = (t, hdr, k) => new TableCell({
    width: { size: Math.round(widths[k] / tot * 100), type: WidthType.PERCENTAGE },
    shading: hdr ? { type: ShadingType.CLEAR, color: "auto", fill: "EEF2F7" } : undefined,
    borders: { top: line, bottom: line, left: none, right: none }, margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: k ? AlignmentType.RIGHT : AlignmentType.LEFT, children: [run(String(t), { size: 18, bold: hdr })] })],
  });
  return new Table({ width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [new TableRow({ tableHeader: true, cantSplit: true, children: head.map((t, k) => cell(t, true, k)) }),
      ...rows.map(r => new TableRow({ cantSplit: true, children: r.map((t, k) => cell(t, false, k)) }))] });
}

// ---------- content ----------
const abstract = `Special investigation units can review only a small share of insurance claims, so the useful question is not "is this claim fraud?" but "which claims should we check first, and how many?". This paper builds a fraud triage model on the public carclaims table of ${int(R.data.rows)} US vehicle claims (1994-1996, ${p1(R.data.fraud_rate)} fraud) and tests it the way it would be used. Settings are chosen on 1994 to 1995, and the models are tested once on the ${int(N)} claims of 1996. Logistic regression is the default model. Gradient boosting is kept only if its gain on 1996 has a 95% bootstrap interval that excludes zero. It passes: ROC-AUC ${f3(P[GB].roc_auc)} ${ci(CI[GB].roc_auc, f3)} against ${f3(P[LR].roc_auc)} ${ci(CI[LR].roc_auc, f3)}, a gain of ${f3(D.roc_auc.point)} ${ci(D.roc_auc.ci95, f3)}. Checking the top 10% of claims finds ${p1(P[GB].capture_top10)} of frauds, against ${p1(P[LR].capture_top10)} for the baseline. A random five-fold split would report an AUC of ${f2(K.random_5fold_all_years[GB].roc_auc)}, overstating performance on a later year. The data has no claim amounts, so a cost model with stated assumptions chooses a review depth on 1995 and applies it to 1996. At that depth the model catches ${int(gD.caught)} of ${NF} frauds while reviewing ${p0(gD.share_checked)} of claims. Each flag carries its three largest TreeSHAP contributions, written as plain sentences. One command reproduces every number.`;

const children = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [run("Cost-aware fraud triage for insurance claims: an honest benchmark", { bold: true, size: 34, color: "1F4E8C" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run("Vivek Mitra", { size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [run("Independent researcher, vivekmitra.com", { size: 20, color: "5A6472" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [run("November 2026", { size: 20, color: "5A6472" })] }),
  H1("Abstract"), para(abstract),
  para([b("Keywords: "), run("insurance fraud, claims triage, special investigation unit, logistic regression, gradient boosting, temporal validation, cost-sensitive decisions, calibration, SHAP")]),

  H1("1. Introduction"),
  para("Claims fraud is a ranking problem under a budget. An insurer cannot investigate every claim, and every investigation costs money and delays honest customers. A fraud model is useful when it puts the claims most worth checking at the top of the list, and when the insurer can say how far down that list to go."),
  para("Statistical fraud detection has a long literature [9], including careful comparisons of classifiers on expert-labelled motor claims [11]. Yet published results on public fraud data often answer an easier question. Models are scored on random train-test splits that mix past and future, on accuracy measures that hide the class imbalance, and with no link to the cost of an investigation. Such numbers tend to be higher than what a model achieves on next year's claims, and they say nothing about how many claims to review."),
  para("This paper sets out a small, honest benchmark for claims triage on a well-known public table. It keeps the method simple on purpose and makes four contributions:"),
  bullet([b("Temporal evaluation. "), run("All choices are made on 1994 and 1995. The models are scored once on 1996, with bootstrap intervals and a paired comparison.")]),
  bullet([b("A simple default and a keep rule. "), run("Logistic regression is the default. A more complex model is kept only if its gain on held-out data has a 95% interval that excludes zero.")]),
  bullet([b("Triage measures and a cost-based depth. "), run("The paper reports the share of frauds found in the top 1%, 5% and 10% of claims. It then chooses a review depth from an expected-value rule with openly stated money assumptions.")]),
  bullet([b("Plain-language reasons. "), run("Each flagged claim carries the three facts that raised its score most, computed exactly and written as sentences an adjuster can read.")]),

  H1("2. Data"),
  para([run("The data is the vehicle insurance claim fraud table known as "), i("carclaims"), run(`, introduced to the research literature by Phua, Alahakoon and Lee [1] and widely shared on Kaggle as "Vehicle Insurance Claim Fraud Detection". It holds ${int(R.data.rows)} US claims from 1994 to 1996, of which ${int(R.data.frauds)} (${p1(R.data.fraud_rate)}) are labelled as fraud found. Each claim has 31 fields. They cover the accident and claim dates (month, week and weekday), the vehicle's make, category, price band and age, the policy and cover type, the deductible, and who was at fault. They also cover the policy holder's age, sex and marital status, past claims, recent address changes, whether a police report was filed or a witness was present, the agent type and the claims representative. There are no claim amounts and no free text.`)]),
  para(`The table was downloaded without a login from a public mirror on the Hugging Face Hub [13], pinned to revision ${R.data.revision.slice(0, 12)} and checked by SHA-256. The mirror stores the rows in two files together with their original row numbers. Re-joining the files gives exactly ${int(R.data.rows)} rows and ${int(R.data.frauds)} frauds.`),
  table(["Year", "Claims", "Frauds", "Fraud rate", "Role"],
    R.data.by_year.map(y => [y.year, int(y.claims), int(y.frauds), p1(y.frauds / y.claims), { 1994: "train; tuning fit", 1995: "train; tuning check", 1996: "test (used once)" }[y.year]]),
    [1, 1, 1, 1, 2]),
  caption("Table 1. Claims by year. The fraud rate falls each year."),
  para([b("Fields never used. "), run("The policy number (a running id), the year (the split variable), and sex and marital status. These two are sensitive attributes, and their use in claims decisions is restricted or contested in many places. Section 4.2 measures the cost of leaving them out.")]),

  H1("3. Method"),
  H2("3.1 Split by time"),
  para(`Each candidate was fitted on the ${int(yr[1994].claims)} claims of 1994 and scored on the ${int(yr[1995].claims)} claims of 1995. The candidate with the best 1995 PR-AUC was kept. The chosen settings were then refitted on 1994 and 1995 together (${int(R.split.train_claims)} claims) and scored once on 1996 (${int(N)} claims, ${NF} frauds). Three field sets were candidates: all fields; all fields except the six calendar fields; and the same without the claims representative. For both models, 1995 preferred the last set ("${set(B[GB].fields)}").`),
  H2("3.2 Models"),
  para([b("Default: logistic regression. "), run(`Every field is one-hot encoded and fed to an L2-regularised logistic regression (scikit-learn [5]). The penalty C was chosen from 0.001, 0.01, 0.1 and 1; 1995 chose C = ${B[LR].C}.`)]),
  para([b("Challenger: gradient boosting. "), run(`LightGBM [2] with learning rate 0.01 and its native handling of categorical fields. Ordered bands, such as vehicle price, are given as ranks. Leaves per tree (4 or 15) and minimum leaf size (20 or 100) were chosen on 1995, with early stopping setting the number of trees. 1995 chose ${B[GB].leaves} leaves, minimum leaf size ${B[GB].min_leaf} and ${B[GB].trees} trees. The ${B[GB].trees} trees were kept fixed when refitting on 1994-95. Neither model uses class weights, so both scores are probabilities.`)]),
  para([b("Keep rule. "), run(`The challenger replaces the default only if the 95% paired bootstrap interval of its ROC-AUC gain on 1996 lies above zero. The rule is applied by the code (results.json, "decision").`)]),
  H2("3.3 Measures and uncertainty"),
  para(`Ranking quality is measured by ROC-AUC and by PR-AUC (average precision), which is more informative when positives are rare [7]. Triage value is capture at k: the share of all test frauds found when checking the top k% of claims, for k = 1, 5 and 10. When scores tie at a cut-off, the tied claims are taken in random order and the expected count is used. Intervals come from ${int(R.test.n_boot)} bootstrap resamples of the 1996 claims [6]. Both models are scored on the same resample each time, which gives a paired interval for their difference. Calibration is checked with the Brier score, the mean predicted against the observed rate, and the calibration error over ten equal-count bins [10]. A small test file checks the core arithmetic against hand calculations: ties, AUC, net saving, and that the per-field reasons add up exactly to the model score.`),
  H2("3.4 Cost model and review depth"),
  para("Let c be the cost of one investigation, L the average loss on a fraudulent claim, and r the share of that loss saved when the fraud is caught. If F(n) of the top n claims are fraud, checking them gives a net saving of"),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [i("Net(n) = F(n) · L · r − n · c.")] }),
  para(`The data has no claim amounts, so L, c and r are stated assumptions: L = ${usd(A.avg_fraud_loss_usd)}, c = ${usd(A.cost_per_check_usd)} and r = ${p0(A.share_recovered)}. They are illustrative, not estimates for any insurer, and the website lets readers change them. Under these values a check pays for itself only when its chance of finding fraud is above c / (L·r) = ${p1(C.break_even_hit_rate)}. The 1996 base rate is ${p1(C.test_base_rate)}. The depth is chosen fairly: each model, fitted on 1994, scores 1995, and the share of claims that maximises Net on 1995 is applied to the 1996 ranking [8]. The best depth chosen with hindsight on 1996 is shown only as an upper bound.`),
  H2("3.5 Reasons for each flag"),
  para("TreeSHAP [3, 4] splits each claim's boosted score (in log-odds) exactly into one contribution per field. The three largest positive contributions become the claim's reasons. Each is written with a fixed template that restates the field value, for example \"Policy holder was at fault\". The templates make no causal claim."),

  H1("4. Results"),
  H2("4.1 Ranking and triage on 1996"),
  table(["Measure (1996 test)", "Logistic regression", "Gradient boosting", "Gain"],
    [["ROC-AUC", "roc_auc", f3], ["PR-AUC", "pr_auc", f3], ["Capture, top 1%", "capture_top1", p1], ["Capture, top 5%", "capture_top5", p1], ["Capture, top 10%", "capture_top10", p1]]
      .map(([lab, k, f]) => [lab, `${f(P[LR][k])} ${ci(CI[LR][k], f)}`, `${f(P[GB][k])} ${ci(CI[GB][k], f)}`,
        f === p1 ? `${(D[k].point * 100).toFixed(1)} pts ${pts(D[k].ci95)}` : `${f3(D[k].point)} ${ci(D[k].ci95, f3)}`]),
    [1.8, 2.2, 2.2, 2.2]),
  caption(`Table 2. Results on ${int(N)} claims from 1996 (${NF} frauds, base rate ${p1(C.test_base_rate)}). Brackets: 95% bootstrap intervals. Gain: observed difference on 1996, gradient boosting minus logistic regression; its interval is the 2.5th to 97.5th percentile of the paired bootstrap differences.`),
  para(`Gradient boosting passes the keep rule: its AUC gain is ${f3(D.roc_auc.point)} with interval ${ci(D.roc_auc.ci95, f3)}. Its PR-AUC gain of ${f3(D.pr_auc.point)} ${ci(D.pr_auc.ci95, f3)} and its gain of ${(D.capture_top10.point * 100).toFixed(1)} pts in the top 10% ${pts(D.capture_top10.ci95)} also exclude zero. In the top 5% the gain is ${(D.capture_top5.point * 100).toFixed(1)} pts and its interval ${pts(D.capture_top5.ci95)} includes zero. Checking the top 10% of claims with the boosted model finds ${p1(P[GB].capture_top10)} of frauds at a hit rate of ${p1(hit(GB, 10))}, about ${(hit(GB, 10) / C.test_base_rate).toFixed(1)} times the base rate.`),
  ...figure("capture.png", 420, 283, "Figure 1. Share of 1996 frauds found against the share of claims checked, highest score first."),
  para(`In absolute terms the ranking is modest. The fields are coarse, and two of them dominate (Figure 4). ${p1(FX.at_fault_non_liability.share_of_frauds)} of all frauds involve a policy holder at fault on collision or all-perils cover. That group is ${p1(FX.at_fault_non_liability.share_of_claims)} of claims and has a fraud rate of ${p1(FX.at_fault_non_liability.fraud_rate_inside)}, against ${p1(FX.at_fault_non_liability.fraud_rate_outside)} outside it. Within the group, honest claims still far outnumber frauds, and the remaining fields separate them only weakly.`),
  H2("4.2 Evaluation choices"),
  table(["Evaluation (ROC-AUC)", "Logistic regression", "Gradient boosting"],
    [["Train 1994-95, test 1996 (this paper)", f3(P[LR].roc_auc), f3(P[GB].roc_auc)],
     ["Random 5-fold split, all years mixed", f3(K.random_5fold_all_years[LR].roc_auc), f3(K.random_5fold_all_years[GB].roc_auc)],
     ["Test 1996, keeping all fields incl. calendar", f3(K.all_fields_incl_calendar[LR].roc_auc), f3(K.all_fields_incl_calendar[GB].roc_auc)],
     ["Test 1996, adding sex and marital status", f3(K.with_sex_and_marital_status[LR].roc_auc), f3(K.with_sex_and_marital_status[GB].roc_auc)]],
    [3, 1.5, 1.5]),
  caption("Table 3. ROC-AUC under other evaluation choices, with the same tuned settings."),
  para(`A random split, which mixes years, reports ${f3(K.random_5fold_all_years[GB].roc_auc)} for boosting and ${f3(K.random_5fold_all_years[LR].roc_auc)} for the baseline. Both are higher than the honest 1996 numbers. Keeping the calendar fields, which the 1995 check rejected, lowers the 1996 AUC to ${f3(K.all_fields_incl_calendar[GB].roc_auc)}, so seasonal patterns did not carry over between years. Adding sex and marital status changes AUC by about ${f3(Math.max(...[LR, GB].map(m => Math.abs(K.with_sex_and_marital_status[m].roc_auc - P[m].roc_auc))))}, well inside the bootstrap noise, so leaving them out costs nothing measurable.`),
  H2("4.3 Calibration"),
  ...figure("calibration.png", 300, 282, "Figure 2. Reliability on 1996: mean predicted fraud probability against the observed rate, by tenth of the score."),
  para(`On 1996 the boosted model predicts a mean fraud probability of ${p1(CAL[GB].mean_predicted)} against an observed ${p1(CAL[GB].observed_rate)}. It over-predicts slightly, as expected given the falling fraud rate in Table 1. Its Brier score is ${f3(CAL[GB].brier)} against ${f3(CAL[GB].brier_base_rate_only)} for always predicting the base rate, and its calibration error is ${p1(CAL[GB].ece_10_bins)}. For logistic regression the figures are ${p1(CAL[LR].mean_predicted)}, ${f3(CAL[LR].brier)} and ${p1(CAL[LR].ece_10_bins)}. A depth expressed as a share of claims does not depend on calibration; a probability threshold would.`),
  H2("4.4 Cost and review depth"),
  table(["Policy on 1996 (assumed money)", "Claims checked", "Frauds caught", "Net saving"],
    [["Check no claims", "0", "0", usd100(0)],
     ["Check every claim", int(N), int(NF), usd100(C.check_everyone.net_saving_usd)],
     ["Logistic regression, depth set on 1995", `${int(lD.checked)} (${p1(lD.share_checked)})`, int(lD.caught), usd100(lD.net_saving_usd)],
     ["Gradient boosting, depth set on 1995", `${int(gD.checked)} (${p1(gD.share_checked)})`, int(gD.caught), `${usd100(gD.net_saving_usd)} ${ci(gD.net_saving_ci95, usd100)}`],
     ["Gradient boosting, best depth in hindsight", `${int(C[GB].hindsight_best_on_1996.checked)} (${p1(C[GB].hindsight_best_on_1996.share_checked)})`, int(C[GB].hindsight_best_on_1996.caught), usd100(C[GB].hindsight_best_on_1996.net_saving_usd)]],
    [3, 1.6, 1.1, 2.2]),
  caption(`Table 4. Review policies on 1996 (L = ${usd(A.avg_fraud_loss_usd)}, c = ${usd(A.cost_per_check_usd)}, r = ${p0(A.share_recovered)}). Frauds caught are measured; money is assumed. Brackets: 95% bootstrap interval.`),
  para(`Checking every claim loses ${usd100(-C.check_everyone.net_saving_usd)}, because the base rate is below the break-even hit rate. The boosted model's depth chosen on 1995, ${p1(gD.share_checked)} of claims, catches ${int(gD.caught)} of ${NF} frauds in 1996 and nets ${usd100(gD.net_saving_usd)}. Choosing the depth with hindsight would net ${usd100(C[GB].hindsight_best_on_1996.net_saving_usd)}.`),
  para([b("An honest caveat. "), run(`At this deep review level the simple baseline does nearly as well: it catches ${int(lD.caught)} frauds by checking ${p1(lD.share_checked)} of claims and nets ${usd100(lD.net_saving_usd)}. With an ${p1(C.break_even_hit_rate)} break-even rate, the best plan is to check most of the at-fault, non-liability group (in 1996, ${int(FX.at_fault_non_liability.test_claims)} claims holding ${FX.at_fault_non_liability.test_frauds} of the ${NF} frauds). Both models find that group easily. The boosted model's advantage is in the order at the top of the list, which matters most when review capacity is tight or each review is expensive.`)]),
  ...figure("net_saving.png", 420, 274, "Figure 3. Net saving on 1996 against the share of claims checked, under the stated money assumptions. Dotted line: the depth set on 1995."),
  table(["Cost per check", "Avg fraud loss", "Depth (set on 1995)", "Frauds caught", "Net saving"],
    C.sensitivity.map(s => [usd(s.cost_per_check_usd), usd(s.avg_fraud_loss_usd), p1(s.share_checked), `${int(s.caught)} of ${NF}`, usd100(s.net_saving_usd)]),
    [1.2, 1.2, 1.4, 1.3, 1.3]),
  caption(`Table 5. Sensitivity of the gradient-boosting policy to the money assumptions (r = ${p0(A.share_recovered)}). Each depth is set on 1995 and applied to 1996.`),
  H2("4.5 What drives the scores, and what a flag says"),
  ...figure("importance.png", 420, 265, "Figure 4. Mean absolute TreeSHAP contribution per field on 1996 test claims (log-odds)."),
  para(`Who was at fault (${f2(R.importance[0].mean_abs_push)}) and the cover type (${f2(R.importance[1].mean_abs_push)}) dominate, followed by policy type, car make and driver age. The highest-scored 1996 claims share the reasons "address changed 2 to 3 years before the claim" and "deductible $500". That pair alone is far from proof. It was fraud in ${FX.address_2to3y_deductible_500.train_frauds} of ${FX.address_2to3y_deductible_500.train_claims} training claims and ${FX.address_2to3y_deductible_500.test_frauds} of ${FX.address_2to3y_deductible_500.test_claims} test claims. The model ranks a claim highest only when the pair is combined with other fields. The reasons describe what the model used. They are not evidence and do not replace an investigator's judgement.`),

  H1("5. Limitations"),
  bullet([b("Old, single-source data. "), run("The claims come from one source in 1994-96. Products, fraud patterns and data capture have changed since.")]),
  bullet([b("Labels are frauds found. "), run("Undetected fraud sits among the negatives, so true capture may be lower, and the model may learn past investigators' habits.")]),
  bullet([b("One test year. "), run(`Only ${NF} test frauds are available. The intervals show the resulting uncertainty, but they cannot cover changes after 1996.`)]),
  bullet([b("No money in the data. "), run("All money figures rest on stated assumptions. Insurers should replace them with their own severities, investigation costs and recovery rates.")]),
  bullet([b("Coarse fields. "), run("Richer data, such as claim amounts, adjuster notes, repair estimates and provider links, would probably change the ranking a great deal.")]),
  bullet([b("Fairness and governance. "), run("Sex and marital status were excluded, but other fields, such as age, may act as proxies. A real deployment needs a fairness review, a human decision on every flag, and documentation of the kind insurance regulators expect [12].")]),

  H1("6. Reproducibility"),
  para([run("The code is released under the GNU Affero General Public License v3.0 at "), i("github.com/vickymitra786/insurance-fraud-triage"), run(". A single command, "), b("python run_all.py"), run(", runs the accuracy checks and downloads and verifies the data. It then fits both models, computes every number and figure in this paper, writes the website files and rebuilds this document.")]),
  bullet(`Software: Python 3.12, scikit-learn ${R.software["scikit-learn"]}, LightGBM ${R.software.lightgbm}, pandas ${R.software.pandas}, NumPy ${R.software.numpy}, and the docx package for Node.js.`),
  bullet("Fixed seed 20261121 for the bootstrap and the random split. Both models are deterministic."),
  bullet(`Data: Hugging Face dataset FeatEng/Data at revision ${R.data.revision}, checked by SHA-256 before use.`),

  H1("References"),
  ...[
    "[1] Phua, C., Alahakoon, D. and Lee, V. (2004). Minority report in fraud detection: classification of skewed data. ACM SIGKDD Explorations Newsletter, 6(1), 50-59. doi:10.1145/1007730.1007738",
    "[2] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems 30.",
    "[3] Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30.",
    "[4] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2, 56-67.",
    "[5] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "[6] Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.",
    "[7] Saito, T. and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS ONE, 10(3), e0118432.",
    "[8] Elkan, C. (2001). The foundations of cost-sensitive learning. Proceedings of the 17th International Joint Conference on Artificial Intelligence, 973-978.",
    "[9] Bolton, R. J. and Hand, D. J. (2002). Statistical fraud detection: A review. Statistical Science, 17(3), 235-255.",
    "[10] Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L. and Steyerberg, E. W. (2019). Calibration: the Achilles heel of predictive analytics. BMC Medicine, 17, 230.",
    "[11] Viaene, S., Derrig, R. A., Baesens, B. and Dedene, G. (2002). A comparison of state-of-the-art classification techniques for expert automobile insurance claim fraud detection. Journal of Risk and Insurance, 69(3), 373-421.",
    "[12] National Association of Insurance Commissioners (2023). Model Bulletin: Use of Artificial Intelligence Systems by Insurers. Adopted December 2023.",
    "[13] FeatEng/Data, folder shivamb__vehicle-claim-fraud-detection. Hugging Face Hub, https://huggingface.co/datasets/FeatEng/Data (a mirror of the Kaggle dataset shivamb/vehicle-claim-fraud-detection).",
  ].map(t => new Paragraph({ spacing: { after: 80 }, indent: { left: 360, hanging: 360 }, children: [run(t, { size: 19 })] })),
];

const doc = new Document({
  creator: "Vivek Mitra", title: "Cost-aware fraud triage for insurance claims: an honest benchmark",
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: { page: { margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run("Vivek Mitra · Cost-aware fraud triage · page ", { size: 16, color: "5A6472" }), new TextRun({ children: [PageNumber.CURRENT], size: 16, color: "5A6472", font: FONT })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(path.join(__dirname, "fraud_triage.docx"), buf); console.log("  wrote paper/fraud_triage.docx"); });
