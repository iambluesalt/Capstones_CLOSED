# Fraud Detection — Version 3

Version 3 rebuilds the project from the raw `MyBank.csv` with a leak-safe pipeline and asks a question V1/V2 never did:
**is there any signal in these features at all?**

## Bottom line

1. **V1 and V2's headline scores (≈98% F1 / 99.7% recall) are evaluation artifacts.** Reproduced with their own protocols, then redone correctly on the same data, test AUC drops from 0.97–0.99 to **0.49** (chance).
2. **`TransactionKey` is a perfect label leak.** All 3,963 frauds have key ≥ 134,790; every non-fraud has key ≤ 134,789 (AUC = 1.000). V1/V2 dropped it, so their features were clean of it, but it is the reason a naive model "works".
3. **The features carry no detectable fraud signal.** 66 engineered features, 190 univariate tests, 14 entity-clustering tests and a tuned gradient-boosting model all land on chance. No model beats a constant predictor on a locked test set.
4. **So there is nothing to "improve" yet.** The honest deliverable is the evidence above plus a reusable pipeline that will evaluate real signal correctly once better data exists.

## Evidence

| # | Claim | Result | Script / figure |
|---|---|---|---|
| 1 | Key is a leak | fraud keys 134,790–138,752, non-fraud 1–134,789, AUC(key) = 1.000 | `01_audit.py`, `fig1` |
| 2 | V1/V2 protocols leak | see table below | `05_leakage_demo.py`, `fig4` |
| 3 | No single-feature signal | 56 numeric features: AUCs 0.491–0.512, **0 pass FDR** (chance expects ~3 outside ±0.009) | `03_deep_tests.py`, `fig2` |
| 4 | No categorical signal | 134 levels (n ≥ 200): 6 nominal p<0.05 (chance ≈ 7), **0 pass FDR**; best column chi² p = 0.027, q = 0.27 | `03_deep_tests.py` |
| 5 | Tests behave like pure noise | 8 of 190 p-values < 0.05 (chance ≈ 10), roughly uniform histogram | `fig3` |
| 6 | No fraud clustering by account | 14 keys tested (same address/email timestamp, same coordinates, …) permutation p = 0.12–0.98 | `03_deep_tests.py` |
| 7 | No multivariate signal | LightGBM 5-fold CV AUC 0.499–0.502 vs shuffled-label null 0.498 ± 0.008 (p = 0.38–0.57); PR-AUC 0.0288 vs base rate 0.0286 | `02_signal_test.py` |
| 8 | The pipeline *can* see signal | planted signals recovered from oracle AUC ≈ 0.55 upward (0.558 → 0.530, 0.615 → 0.589) | `04_power_check.py`, `fig5` |
| 9 | Tuning doesn't help | Optuna, 40 trials: locked-test AUC 0.506 on real labels vs 0.503 on shuffled labels (SE ≈ 0.010) | `06_tuning_noise.py`, `fig6` |

### V1 / V2 protocols vs the correct one (same data, same XGBoost)

| Protocol | Test AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| A — V2: SMOTE on all rows → split | 0.986 | 0.9997 | 0.970 | 0.985 |
| B — V1: RandomOverSampler on all rows → test on original 20% | 0.972 | 0.180 | 0.938 | 0.302 |
| C — correct: split → SMOTE on train only | 0.494 | 0.000 | 0.000 | 0.000 |
| D — correct: split → `scale_pos_weight` | 0.486 | 0.029 | 0.129 | 0.048 |
| E — `TransactionKey` left in | 1.000 | 0.949 | 0.976 | 0.962 |
| E′ — same, key removed | 0.499 | 0.029 | 0.340 | 0.053 |

A's test set is balanced by SMOTE (half positives), so its precision/F1 are not comparable to the others; AUC is the fair column.
V2's own "normal data" cells already showed the truth (test recall ≈ 0% for every model); the SMOTE/ADASYN cells hid it.

### Leak-safe final evaluation (locked 20% test, 793 frauds; 5-fold OOF CV on the rest)

| Model | CV AUC | Test AUC [95% CI] | Test PR-AUC (base rate 0.0286) |
|---|---|---|---|
| constant (base rate) | 0.500 | 0.500 | 0.0286 |
| logistic L2 | 0.493 | 0.497 [0.474, 0.517] | 0.0290 |
| LightGBM (regularised) | 0.495 | 0.490 [0.468, 0.511] | 0.0282 |
| XGBoost (`scale_pos_weight`) | 0.496 | 0.496 [0.475, 0.518] | 0.0287 |

Every CI contains 0.5. CV AUC a hair under 0.5 is the known bias of stratified CV when there is no signal.
Top-1% "lift" is noise at this size: the constant-score dummy shows 1.38× because a 1% queue is only 278 rows.

## How far the conclusion goes

* **Detection limits.** One feature at a time, an AUC deviation ≥ 0.009 is detectable at α = 0.05 (≥ 0.013 at 80% power). The multivariate model reliably sees signal from oracle AUC ≈ 0.55. Effects smaller than that could exist, but at AUC 0.51–0.53 they would be operationally worthless.
* **Not proof of zero, but strong.** The pipeline was validated by planting signal and by shuffling labels, so the null is a property of the data, not a bug in the modelling.
* **Hypothesis (not verified):** the labels look assigned independently of the features (frauds are exactly the last 3,963 keys, and every feature is independent of the label). That points to a synthetic or shuffled dataset. Only the data owner can confirm.
* **Missing ingredients for real fraud modelling:** no transaction amount, merchant, card/account/device ID, or label delay. `AddressUpdateDate` has a 9,820-row value (2004-01-20 03:26:16) that looks like a system default rather than an account.

## What would actually improve the model

1. Ask the data owner how `Fraud` and `TransactionKey` were generated, and whether an account/device ID and amount exist.
2. With a real account ID: `GroupKFold` by account, a time-based hold-out, and velocity features (counts and amounts per account/device over rolling windows).
3. Keep the protocol in `src/fd`: split first, sampling only inside training folds, PR-AUC and precision@k not accuracy, threshold chosen from review cost (a random review queue pays off only if average fraud loss / review cost > 35).
4. Present the project as a **data-validity case study**: it is a stronger story than a 98% score that collapses under a correct split.

## Data-quality notes (from `01_audit.py`)

* All 138,752 rows are kept. V1/V2 ended with 138,372. The 380 lost rows are 354 with State `unknown`/`none` (86 + 268, of which 13 fraud) plus 26 with an unparseable `EmailUpdateDate` (0 fraud), so 13 of 3,963 frauds were dropped; reproduced from the raw file.
* 97.4% of rows fall in 12 days (2013-05-22 to 06-02); 3,622 stragglers run to Oct 22. A temporal split is not meaningful.
* `WebSessionRetail` is constant 0 (or missing); `TimeZone` has a 999 sentinel (4 rows); 1,036 address and 1,132 email updates are dated after the transaction; `CurrentLat` is missing for 24,379 rows while `CurrentLong` is never missing.
* Missingness is not informative: for every column with > 4,000 missing rows the fraud rate is 2.4–3.2% whether present or not (the two columns with only 12 and 26 missing rows are too small to read).

## Layout and reproduction

**Start with `Fraud_Detection_V3.ipynb`**: the readable walkthrough with all outputs and figures embedded. Light checks run live; heavy experiments load from `reports/`. It imports `src/fd`, so no logic is duplicated.

```
Version 3/
  Fraud_Detection_V3.ipynb            narrative walkthrough (executed, outputs saved)
  src/fd/{data,features,metrics}.py   label-free features, honest metrics (no accuracy)
  scripts/01_audit … 08_figures.py    one script per experiment; numbered in run order
  reports/tables/*.csv|txt|log        results behind every number above
  reports/figures/fig1…fig6.png
```

```bash
cd "Fraud-Detection/Version 3"
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
for s in 01_audit 02_signal_test 03_deep_tests 04_power_check 05_leakage_demo 07_final_evaluation 08_figures; do
  .venv/Scripts/python.exe scripts/$s.py; done
.venv/Scripts/python.exe scripts/02_signal_test.py 20     # adds the 20-shuffle permutation null
.venv/Scripts/python.exe scripts/06_tuning_noise.py 40    # slowest step, ~3–4 minutes on 20 cores
```

Raw data is read from `../Version 1/MyBank.csv` (override with `FD_RAW_CSV`); `05_leakage_demo.py` also needs `../Version 2/FD-Dataset-2.csv`.
Root `.gitignore` excludes `*.csv`, so data is never committed; the small result tables in `reports/tables` are whitelisted.
