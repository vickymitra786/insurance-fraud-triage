"""Tiny accuracy checks against hand calculations and library references. Author: Vivek Mitra
Run: python test_core.py   (run_all.py runs it first and stops if any check fails)
"""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

from model import capture, cum_caught, net_saving

# Four claims: scores 0.9, 0.5, 0.5, 0.1; the first and one of the tied pair are fraud.
y = np.array([1, 0, 1, 0])
s = np.array([0.9, 0.5, 0.5, 0.1])


def test_cum_caught_by_hand():
    # Top 1 finds 1 fraud. Top 2 takes one of the tied pair at random: expected 1 + 0.5. Top 3 has both.
    assert np.allclose(cum_caught(y, s), [0, 1, 1.5, 2, 2])
    assert capture(y, s, 0.5) == 0.75          # top 2 of 4 claims -> 1.5 of 2 frauds


def test_auc_by_hand():
    # Pairs (fraud, genuine): 0.9>0.5, 0.9>0.1, 0.5=0.5 (half), 0.5>0.1 -> 3.5 of 4.
    assert roc_auc_score(y, s) == 3.5 / 4


def test_net_saving_by_hand():
    money = {"avg_fraud_loss_usd": 10, "share_recovered": 0.5, "cost_per_check_usd": 1}
    # n checks: caught x 10 x 0.5 - n x 1
    assert np.allclose(net_saving(y, s, money), [0, 4, 5.5, 7, 6])


def test_tree_reasons_add_up():
    # The per-field pushes (TreeSHAP) plus the base value must equal the model's raw score for every claim.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 4))
    t = (X[:, 0] + rng.normal(size=500) > 1).astype(int)
    m = lgb.LGBMClassifier(n_estimators=50, num_leaves=7, verbose=-1).fit(X, t)
    contrib = m.predict(X, pred_contrib=True)
    assert np.allclose(contrib.sum(axis=1), m.predict(X, raw_score=True))


def main():
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"  ok  {name}")


if __name__ == "__main__":
    main()
