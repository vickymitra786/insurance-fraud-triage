# Insurance fraud triage: an honest benchmark

Which claims should a special investigation team check first, and how many? This repository builds a claims fraud triage model on real, public vehicle insurance claims. It tests the model the way it would be used: trained on earlier years, scored once on a later year it never saw, compared against a baseline, with uncertainty shown and the cost of each review taken into account.

- **Live demo:** <https://vivekmitra.com/fraud.html>
- **Paper:** [Cost-aware fraud triage for insurance claims: an honest benchmark (PDF)](https://vivekmitra.com/papers/fraud.pdf)

## Results on the 1996 test year

Trained on 1994–95 claims and tested once on all 1996 claims. Brackets show 95% bootstrap intervals (2,000 resamples of the test claims).

<!-- results:start -->
| Measure (1996 test) | Logistic regression | Gradient boosting |
|---|---|---|
| ROC-AUC | 0.766 [0.742, 0.789] | 0.789 [0.766, 0.810] |
| PR-AUC | 0.112 [0.095, 0.136] | 0.162 [0.127, 0.202] |
| Frauds found, top 1% of claims | 0.9% [0.0%, 3.3%] | 4.2% [2.3%, 7.4%] |
| Frauds found, top 5% of claims | 12.7% [8.1%, 16.7%] | 15.5% [10.9%, 20.1%] |
| Frauds found, top 10% of claims | 22.5% [16.7%, 27.5%] | 29.1% [22.8%, 34.7%] |

Gain of gradient boosting in AUC: +0.023 [+0.008, +0.039]. Primary model by the keep rule: **gradient boosting**.
<!-- results:end -->

Money figures in the demo and paper rest on **stated assumptions**, because the public data has no claim amounts. Frauds caught and missed are measured outcomes.

## What is in here

| File | What it does |
|---|---|
| `fetch_data.py` | Downloads the data from a public mirror (no login), pinned to one revision and checked by SHA-256 |
| `features.py` | Which fields each model uses, how they are encoded, and plain-language reason templates |
| `model.py` | Time split, tuning on 1994 → 1995, final fit, test metrics with bootstrap intervals, the keep rule, calibration, cost model, per-claim reasons and checks. Writes `results/results.json` |
| `test_core.py` | Tiny accuracy checks against hand calculations and library references |
| `figures.py` | The four paper figures |
| `export_site.py` | Website data (1996 test claims only), the home-page card and this README's results table |
| `paper/` | `build_paper.js` builds the paper with Node.js; `to_pdf.ps1` converts it to PDF with Word on Windows |
| `run_all.py` | Runs everything in order |

## Reproduce

Needs Python 3.12 and Node.js 18 or newer. The PDF step needs Microsoft Word on Windows.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

### Data download

`run_all.py` fetches the data by itself. To fetch it on its own, run `python fetch_data.py`. The table is the public "Vehicle Insurance Claim Fraud Detection" data (the Oracle *carclaims* table, Phua, Alahakoon and Lee, 2004): 15,420 US vehicle claims from 1994–1996, 923 of them fraud. It is downloaded from the public Hugging Face mirror [`FeatEng/Data`](https://huggingface.co/datasets/FeatEng/Data) (folder `shivamb__vehicle-claim-fraud-detection`, revision `eb82fee0a992bccbf63cc667f2b044cd396c9a5c`). No account is needed. Both files are checked against fixed SHA-256 hashes, and the script stops if they differ. The data is not stored in this repository.

## Method in brief

- **Split by time:** settings are chosen by fitting on 1994 and checking on 1995, then refit on 1994–95 and tested once on 1996.
- **Default model:** logistic regression on one-hot fields. **Challenger:** LightGBM gradient boosting. The challenger is kept only if its AUC gain on 1996 has a 95% bootstrap interval that excludes zero. It does: +0.023 [+0.008, +0.039].
- **Left out:** sex and marital status (sensitive), the policy number (a running id) and the year. Calendar fields and the claims representative were dropped because the 1995 check preferred the smaller field set.
- **Measures:** ROC-AUC, PR-AUC, the share of frauds found in the top 1%, 5% and 10% of claims, and calibration, each with bootstrap intervals.
- **Review depth:** the share of claims that maximises net saving on 1995 is applied to 1996.
- **Reasons:** each claim's score is split exactly into one push per field (TreeSHAP), and the three biggest pushes are written as plain sentences.

## Licence

Copyright (C) 2026 Vivek Mitra. Licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).

The claims data belongs to its original publishers and is not covered by this licence.
