"""Fraud triage on public vehicle claims. Author: Vivek Mitra

Split by time: choose settings by fitting on 1994 and checking on 1995, refit on 1994-95, test once on 1996.
Primary model: logistic regression. Gradient boosting is kept only if its AUC gain on 1996 has a
95% bootstrap interval that excludes zero. Writes results/results.json and results/test_scores.csv.
"""
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import features as F

HERE = Path(__file__).resolve().parent
SEED = 20261121
N_BOOT = 2000
# The data has no claim amounts, so money is a stated assumption, not a measurement.
MONEY = {"avg_fraud_loss_usd": 5000, "cost_per_check_usd": 200, "share_recovered": 0.5}
LR, GB = "logistic_regression", "gradient_boosting"


# ---------- the maths ----------
def cum_caught(y, s):
    """Frauds found after checking the n highest-scored claims, for n = 0..N.
    Tied scores are checked in random order, so inside a tie the expected count is used."""
    order = np.argsort(-s, kind="stable")
    ss, c = s[order], np.concatenate([[0.0], np.cumsum(y[order])])
    out = c.copy()
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(ss)]):
        if b - a > 1:
            out[a:b + 1] = c[a] + np.arange(b - a + 1) * (c[b] - c[a]) / (b - a)
    return out


def capture(y, s, k):
    """Share of all frauds found when checking the top k share of claims."""
    return float(cum_caught(y, s)[math.ceil(k * len(y))] / y.sum())


def metrics(y, s):
    return {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "capture_top1": capture(y, s, 0.01), "capture_top5": capture(y, s, 0.05), "capture_top10": capture(y, s, 0.10)}


def net_saving(y, s, money):
    """Net saving for checking the top n claims, n = 0..N: frauds found x loss x share recovered - checks x cost."""
    n = np.arange(len(y) + 1)
    return cum_caught(y, s) * money["avg_fraud_loss_usd"] * money["share_recovered"] - n * money["cost_per_check_usd"]


def plan(y, s, share, money):
    n = round(share * len(y))
    caught = float(cum_caught(y, s)[n])
    return {"share_checked": n / len(y), "checked": n, "caught": caught, "missed": float(y.sum() - caught),
            "net_saving_usd": float(net_saving(y, s, money)[n])}


def ci95(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def calibration(y, p):
    bins = np.array_split(np.argsort(p), 10)
    return {"brier": brier_score_loss(y, p), "brier_base_rate_only": brier_score_loss(y, np.full(len(y), y.mean())),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
            "ece_10_bins": float(sum(len(b) / len(y) * abs(p[b].mean() - y[b].mean()) for b in bins)),
            "bins": [{"mean_pred": float(p[b].mean()), "observed": float(y[b].mean())} for b in bins]}


# ---------- the two models ----------
def fit_lr(d, cols, C):
    m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(C=C, max_iter=5000))
    return m.fit(F.lr_x(d, cols), d[F.TARGET])


def fit_gb(d, cols, leaves, min_leaf, trees):
    m = lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.01, num_leaves=leaves, min_child_samples=min_leaf,
                           random_state=SEED, verbose=-1)
    return m.fit(F.gbm_x(d, cols, d), d[F.TARGET])


def score(name, m, d, cols, ref):
    X = F.lr_x(d, cols) if name == LR else F.gbm_x(d, cols, ref)
    return m.predict_proba(X)[:, 1]


def tune(a, b):
    """Pick the field set and settings with the best PR-AUC on 1995 for models fitted on 1994."""
    ya, yb, rows = a[F.TARGET], b[F.TARGET], []
    for set_name in F.SETS:
        cols = F.columns(set_name)
        for C in [0.001, 0.01, 0.1, 1.0]:
            s = score(LR, fit_lr(a, cols, C), b, cols, a)
            rows.append({"model": LR, "fields": set_name, "C": C, "valid_pr_auc": average_precision_score(yb, s)})
        for leaves in [4, 15]:
            for min_leaf in [20, 100]:
                m = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.01, num_leaves=leaves, min_child_samples=min_leaf,
                                       random_state=SEED, verbose=-1)
                m.fit(F.gbm_x(a, cols, a), ya, eval_set=[(F.gbm_x(b, cols, a), yb)], callbacks=[lgb.early_stopping(300, verbose=False)])
                s = m.predict_proba(F.gbm_x(b, cols, a), num_iteration=m.best_iteration_)[:, 1]
                rows.append({"model": GB, "fields": set_name, "leaves": leaves, "min_leaf": min_leaf,
                             "trees": int(m.best_iteration_), "valid_pr_auc": average_precision_score(yb, s)})
    best = {k: max((r for r in rows if r["model"] == k), key=lambda r: r["valid_pr_auc"]) for k in (LR, GB)}
    return best, rows


def fit(name, best, d, sensitive=False, fields=None):
    cols = F.columns(fields or best["fields"], sensitive)
    m = fit_lr(d, cols, best["C"]) if name == LR else fit_gb(d, cols, best["leaves"], best["min_leaf"], best["trees"])
    return m, cols


# ---------- main ----------
def main():
    d = pd.read_csv(HERE / "data" / "carclaims.csv")
    train, test = d[d.Year <= 1995].reset_index(drop=True), d[d.Year == 1996].reset_index(drop=True)
    y = test[F.TARGET].to_numpy()
    print(f"  train {len(train):,} claims (1994-95), test {len(test):,} claims (1996), {y.sum()} test frauds")

    best, grid = tune(d[d.Year == 1994], d[d.Year == 1995])
    models = {k: fit(k, best[k], train) for k in (LR, GB)}
    s = {k: score(k, m, test, cols, train) for k, (m, cols) in models.items()}
    point = {k: metrics(y, s[k]) for k in s}

    # Bootstrap: resample the test claims, score both models on the same resample.
    rng = np.random.default_rng(SEED)
    draws = {LR: [], GB: []}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        for k in draws:
            draws[k].append(metrics(y[i], s[k][i]))
    boot = {k: {m: ci95([x[m] for x in draws[k]]) for m in point[k]} for k in draws}
    # Gain = observed difference on 1996; interval = percentiles of the paired bootstrap differences.
    diff = {m: {"point": float(point[GB][m] - point[LR][m]),
                "ci95": ci95([g[m] - l[m] for g, l in zip(draws[GB], draws[LR])])} for m in point[LR]}
    keep_gb = diff["roc_auc"]["ci95"][0] > 0
    primary = GB if keep_gb else LR
    print(f"  AUC {LR} {point[LR]['roc_auc']:.3f} {GB} {point[GB]['roc_auc']:.3f}, gain {diff['roc_auc']['ci95']} -> primary: {primary}")

    # Review depth: choose the best share of claims on 1995 (models fitted on 1994), apply it to 1996.
    y95 = d.loc[d.Year == 1995, F.TARGET].to_numpy()
    cost = {"assumptions": MONEY, "break_even_hit_rate": MONEY["cost_per_check_usd"] / (MONEY["avg_fraud_loss_usd"] * MONEY["share_recovered"]),
            "test_base_rate": float(y.mean()), "check_everyone": plan(y, s[LR], 1.0, MONEY)}
    s95 = {}
    for k in (LR, GB):
        m94, cols = fit(k, best[k], d[d.Year == 1994])
        s95[k] = score(k, m94, d[d.Year == 1995], cols, d[d.Year == 1994])
        share = int(np.argmax(net_saving(y95, s95[k], MONEY))) / len(y95)
        cost[k] = {"depth_chosen_on_1995": plan(y, s[k], share, MONEY),
                   "hindsight_best_on_1996": plan(y, s[k], int(np.argmax(net_saving(y, s[k], MONEY))) / len(y), MONEY)}
    share_p = cost[primary]["depth_chosen_on_1995"]["share_checked"]
    cost[primary]["depth_chosen_on_1995"]["net_saving_ci95"] = ci95(
        [plan(y[i], s[primary][i], share_p, MONEY)["net_saving_usd"] for i in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))])
    cost["sensitivity"] = []
    for c in [50, 100, 200, 400]:
        for loss in [2000, 5000, 10000]:
            mo = {**MONEY, "cost_per_check_usd": c, "avg_fraud_loss_usd": loss}
            share = int(np.argmax(net_saving(y95, s95[primary], mo))) / len(y95)
            cost["sensitivity"].append({"cost_per_check_usd": c, "avg_fraud_loss_usd": loss, **plan(y, s[primary], share, mo)})

    # Reasons: each claim's score split into one push per field (exact for both model types).
    m, cols = models[primary]
    if primary == GB:
        contrib = m.predict(F.gbm_x(test, cols, train), pred_contrib=True)[:, :-1]
    else:  # linear model: push of a field = sum over its 0/1 columns of weight x (value - training average)
        enc, lr = m.named_steps["onehotencoder"], m.named_steps["logisticregression"]
        Xtr, Xte = enc.transform(F.lr_x(train, cols)).toarray(), enc.transform(F.lr_x(test, cols)).toarray()
        per_col = lr.coef_[0] * (Xte - Xtr.mean(axis=0))
        field = np.repeat(np.arange(len(cols)), [len(c) for c in enc.categories_])
        contrib = np.stack([per_col[:, field == j].sum(axis=1) for j in range(len(cols))], axis=1)
    reasons = [[F.reason(cols[j], test.loc[i, cols[j]]) for j in np.argsort(-contrib[i])[:3] if contrib[i, j] > 0] for i in range(len(test))]
    importance = sorted(({"field": c, "mean_abs_push": float(np.abs(contrib[:, j]).mean())} for j, c in enumerate(cols)), key=lambda r: -r["mean_abs_push"])

    # Checks on other evaluation choices, for both models.
    checks = {"with_sex_and_marital_status": {}, "all_fields_incl_calendar": {}, "random_5fold_all_years": {}}
    for k in (LR, GB):
        mk, ck = fit(k, best[k], train, sensitive=True)
        checks["with_sex_and_marital_status"][k] = metrics(y, score(k, mk, test, ck, train))
        mk, ck = fit(k, best[k], train, fields="all_fields")
        checks["all_fields_incl_calendar"][k] = metrics(y, score(k, mk, test, ck, train))
        oof = np.zeros(len(d))
        for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=SEED).split(d, d[F.TARGET]):
            mk, ck = fit(k, best[k], d.iloc[tr_i])
            oof[te_i] = score(k, mk, d.iloc[te_i], ck, d.iloc[tr_i])
        checks["random_5fold_all_years"][k] = metrics(d[F.TARGET].to_numpy(), oof)

    group = (d.Fault == "Policy Holder") & (d.BasePolicy != "Liability")
    tg = (test.Fault == "Policy Holder") & (test.BasePolicy != "Liability")
    combo = lambda x: x[(x.AddressChange_Claim == "2 to 3 years") & (x.Deductible == 500)][F.TARGET]
    facts = {"age_recorded_zero": int((d.Age == 0).sum()),
             "at_fault_non_liability": {"share_of_frauds": float(d[F.TARGET][group].sum() / d[F.TARGET].sum()),
                                        "share_of_claims": float(group.mean()), "fraud_rate_inside": float(d[F.TARGET][group].mean()),
                                        "fraud_rate_outside": float(d[F.TARGET][~group].mean()),
                                        "test_claims": int(tg.sum()), "test_frauds": int(test[F.TARGET][tg].sum())},
             "address_2to3y_deductible_500": {"train_claims": len(combo(train)), "train_frauds": int(combo(train).sum()),
                                              "test_claims": len(combo(test)), "test_frauds": int(combo(test).sum())}}

    res = {
        "data": {"rows": len(d), "frauds": int(d[F.TARGET].sum()), "fraud_rate": float(d[F.TARGET].mean()),
                 "by_year": [{"year": int(yr), "claims": int(len(g)), "frauds": int(g[F.TARGET].sum())} for yr, g in d.groupby("Year")],
                 "revision": "eb82fee0a992bccbf63cc667f2b044cd396c9a5c"},
        "split": {"train_claims": len(train), "test_claims": len(test), "test_frauds": int(y.sum())},
        "tuning": {"best": best, "grid": grid},
        "fields": {k: cols for k, (_, cols) in models.items()},
        "test": {"point": point, "ci95": boot, "difference_gb_minus_lr": diff, "n_boot": N_BOOT},
        "decision": {"keep_gradient_boosting": bool(keep_gb), "primary": primary,
                     "rule": "keep gradient boosting only if the 95% interval of its AUC gain over logistic regression on 1996 excludes zero"},
        "calibration": {k: calibration(y, s[k]) for k in s},
        "cost": cost, "importance": importance, "checks": checks, "facts": facts,
        "software": {"lightgbm": lgb.__version__, "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "results.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame({"PolicyNumber": test.PolicyNumber, "label": y, "score_lr": s[LR], "score_gb": s[GB],
                  "reasons": [json.dumps(r) for r in reasons]}).to_csv(HERE / "results" / "test_scores.csv", index=False)
    print("  wrote results/results.json and results/test_scores.csv")


if __name__ == "__main__":
    main()
