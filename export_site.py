"""Website data (1996 test claims only), the home-page card and the README results table. Author: Vivek Mitra"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SITE = HERE.parent.parent / "site" / "data"
NAMES = {"gradient_boosting": "Gradient boosting", "logistic_regression": "Logistic regression"}
COL = {"gradient_boosting": "score_gb", "logistic_regression": "score_lr"}
r4 = lambda v: round(float(v), 4)


def main():
    r = json.loads((HERE / "results" / "results.json").read_text())
    t = pd.read_csv(HERE / "results" / "test_scores.csv")
    raw = pd.read_csv(HERE / "data" / "carclaims.csv").set_index("PolicyNumber")
    main_m = r["decision"]["primary"]
    other = next(k for k in NAMES if k != main_m)
    P, CI, D, C = r["test"]["point"], r["test"]["ci95"], r["test"]["difference_gb_minus_lr"], r["cost"]
    t = t.sort_values(COL[main_m], ascending=False, kind="stable").reset_index(drop=True)

    top = [{"rank": i + 1, "score": r4(row[COL[main_m]]), "fraud": int(row.label),
            "car": f"{raw.Make[row.PolicyNumber]} {raw.VehicleCategory[row.PolicyNumber].lower()}",
            "cover": raw.BasePolicy[row.PolicyNumber], "reasons": json.loads(row.reasons)}
           for i, row in t.head(40).iterrows()]
    metric = lambda k: {m: {"v": r4(v), "ci": [r4(x) for x in CI[k][m]]} for m, v in P[k].items()}
    depth = C[main_m]["depth_chosen_on_1995"]
    demo = {
        "model_name": NAMES[main_m], "other_name": NAMES[other],
        "n": len(t), "frauds": int(t.label.sum()), "base_rate": r4(t.label.mean()),
        "assumptions": C["assumptions"], "depth_last_year": r4(depth["share_checked"]),
        "score": [r4(v) for v in t[COL[main_m]]], "label": t.label.tolist(), "score_other": [r4(v) for v in t[COL[other]]],
        "top": top,
        "metrics": {"lr": metric("logistic_regression"), "gb": metric("gradient_boosting"),
                    "gain": {m: {"v": r4(v["point"]), "ci": [r4(x) for x in v["ci95"]]} for m, v in D.items()}},
        "calibration": {k: r4(v) for k, v in r["calibration"][main_m].items() if k != "bins"},
        "caught_at_depth": {"model": r4(depth["caught"]), "other": r4(C[other]["depth_chosen_on_1995"]["caught"])},
        "random_split_auc": r4(r["checks"]["random_5fold_all_years"][main_m]["roc_auc"]),
        "with_calendar_auc": r4(r["checks"]["all_fields_incl_calendar"][main_m]["roc_auc"]),
        "importance": [{"field": x["field"], "v": r4(x["mean_abs_push"])} for x in r["importance"][:8]],
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE / "fraud.json").write_text(json.dumps(demo, separators=(",", ":")))

    m, lo, hi = P[main_m]["roc_auc"], *CI[main_m]["roc_auc"]
    card = {
        "slug": "fraud", "stage": "Claims", "title": "Claims fraud triage: which claims to check first",
        "problem": "Fraud teams can review only a fraction of claims, so they need a ranked list that finds the most fraud per review.",
        "headline": f"{P[main_m]['capture_top10']:.0%} of frauds",
        "headline_note": (f"found by checking the top 10% of claims in a later year the model never saw (1996, {r['split']['test_claims']:,} real claims, "
                          f"{r['split']['test_frauds']} frauds), against 10% by chance. Ranking quality: AUC {m:.2f} (95% interval {lo:.2f} to {hi:.2f})."),
        "href": "fraud.html", "paper": "papers/fraud.pdf",
        "abstract": (f"A fraud triage model trained on 1994-95 public vehicle claims and tested once on 1996. {NAMES[main_m]} ranks claims with "
                     f"AUC {m:.3f} against {P[other]['roc_auc']:.3f} for {NAMES[other].lower()}, and finds {P[main_m]['capture_top5']:.0%} of frauds in "
                     f"the top 5% of claims. A random split would overstate AUC at {demo['random_split_auc']:.2f}. A cost-based review depth, "
                     "plain-language reasons and stated money assumptions turn the scores into a review list."),
    }
    (SITE / "fraud.card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))

    pc, f3 = (lambda v: f"{v:.1%}"), (lambda v: f"{v:.3f}")
    lines = ["| Measure (1996 test) | Logistic regression | Gradient boosting |", "|---|---|---|"]
    for label, k, f in [("ROC-AUC", "roc_auc", f3), ("PR-AUC", "pr_auc", f3), ("Frauds found, top 1% of claims", "capture_top1", pc),
                        ("Frauds found, top 5% of claims", "capture_top5", pc), ("Frauds found, top 10% of claims", "capture_top10", pc)]:
        lines.append("| " + label + " | " + " | ".join(f"{f(P[x][k])} [{f(CI[x][k][0])}, {f(CI[x][k][1])}]" for x in ("logistic_regression", "gradient_boosting")) + " |")
    lines.append(f"\nGain of gradient boosting in AUC: {D['roc_auc']['point']:+.3f} [{D['roc_auc']['ci95'][0]:+.3f}, {D['roc_auc']['ci95'][1]:+.3f}]. "
                 f"Primary model by the keep rule: **{NAMES[main_m].lower()}**.")
    readme = HERE / "README.md"
    text, a, b = readme.read_text(encoding="utf-8"), "<!-- results:start -->", "<!-- results:end -->"
    readme.write_text(text[: text.index(a) + len(a)] + "\n" + "\n".join(lines) + "\n" + text[text.index(b):], encoding="utf-8")
    print("  wrote site/data/fraud.json, site/data/fraud.card.json and the README results table")


if __name__ == "__main__":
    main()
