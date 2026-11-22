"""Check that numbers printed in the paper's tables match results/results.json. Author: Vivek Mitra"""
import json
import re
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tables(docx):
    """Every table in the .docx as a list of rows of cell text."""
    xml = zipfile.ZipFile(docx).read("word/document.xml").decode("utf-8")
    text = lambda frag: "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", frag))
    return [[[text(c) for c in re.findall(r"<w:tc>.*?</w:tc>", row, re.S)] for row in re.findall(r"<w:tr[ >].*?</w:tr>", t, re.S)]
            for t in re.findall(r"<w:tbl>.*?</w:tbl>", xml, re.S)]


def main():
    r = json.loads((HERE / "results" / "results.json").read_text())
    tabs = {t[0][0]: t[1:] for t in tables(HERE / "paper" / "fraud_triage.docx")}

    # Table 2: point values and intervals, and the gain as observed difference.
    P, CI, D = r["test"]["point"], r["test"]["ci95"], r["test"]["difference_gb_minus_lr"]
    f3, pc, pt = (lambda v: f"{v:.3f}"), (lambda v: f"{v * 100:.1f}%"), (lambda v: f"{v * 100:.1f}")
    for row, (m, f) in zip(tabs["Measure (1996 test)"], [("roc_auc", f3), ("pr_auc", f3), ("capture_top1", pc), ("capture_top5", pc), ("capture_top10", pc)]):
        want = [f"{f(P[k][m])} [{f(CI[k][m][0])}, {f(CI[k][m][1])}]" for k in ("logistic_regression", "gradient_boosting")]
        want.append(f"{f3(D[m]['point'])} [{f3(D[m]['ci95'][0])}, {f3(D[m]['ci95'][1])}]" if f is f3 else
                    f"{pt(D[m]['point'])} pts [{pt(D[m]['ci95'][0])}, {pt(D[m]['ci95'][1])}] pts")
        assert row[1:] == want, (m, row, want)

    # Table 5: the assumption inputs must print exactly as the grid set in model.py, never rounded.
    rows = tabs["Cost per check"]
    grid = [(f"${s['cost_per_check_usd']:,}", f"${s['avg_fraud_loss_usd']:,}") for s in r["cost"]["sensitivity"]]
    assert [(row[0], row[1]) for row in rows] == grid, ([(row[0], row[1]) for row in rows], grid)
    print(f"  ok  paper tables match results.json ({len(tabs['Measure (1996 test)'])} rows in Table 2, {len(rows)} in Table 5)")


if __name__ == "__main__":
    main()
