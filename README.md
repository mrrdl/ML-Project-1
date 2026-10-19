# Industrial Machine Condition Classification

Machine Learning Course Project-I (Kaggle competition).
Team: **[team name]** - [member 1], [member 2], [member 3], [member 4]

Task: classify an industrial machine as **0 = Normal**, **1 = Thermal Failure** or
**2 = Mechanical Failure** from 10 sensor readings and the machine type.
Metric: classification accuracy. Training set: 16,000 rows; test set: 4,000 rows.

Models used: logistic regression (plain and degree-3 polynomial softmax), random forest, XGBoost, CatBoost.

## Repository layout

```
data/                  put the three competition CSVs here (not committed, see below)
src/
  features.py          shared engineered features (used from step 5 onward)
  01_eda.py            exploratory data analysis, saves plots to eda_plots/
  02_baseline_logreg.py    logistic regression baseline           -> submission_01
  03_random_forest.py      random forest + randomized search      -> submission_02
  04_xgboost.py            XGBoost + randomized search            -> submission_03
  05_feature_engineering.py  feature-group ablation (LR and RF)
  06_xgboost_fe.py         XGBoost with engineered features       -> submission_04
  07_ensemble.py           RF + XGBoost (+ LR) ensemble           -> submission_05
  08_softmax.py            degree-3 polynomial softmax regression -> submission_07
notebooks/
  catboost-2.ipynb         CatBoost with randomized search        -> submission_06
  softmax.ipynb            Kaggle notebook version of step 08
submissions/           every CSV uploaded to Kaggle
eda_plots/             plots produced by 01_eda.py
results.md             full results log (CV and leaderboard score of every run)
outputs.md             console output of every step (EDA, tuning tables, confusion matrices)
```

## How to reproduce

1. Download `student_dataset.csv`, `test.csv` and `sample_submission.csv` from the
   competition page and put them in `data/`. (They are not committed because the
   data belongs to the competition.)
2. `pip install -r requirements.txt`
3. From the repository root, run the scripts in order, e.g. `python src/04_xgboost.py`.
   Each script writes its submission file to `submissions/` and appends a row to `results.md`.

**Reproducibility:** every random component uses `random_state=42` (models, CV splits,
randomized searches), so reruns give the same numbers in the same environment.
Small differences are possible across different library versions.

## Results

| Step | Model | 5-fold CV accuracy | Kaggle public score |
|------|-------|--------------------|---------------------|
| 01 | Logistic Regression | 0.8255 | 0.82250 |
| 02 | Random Forest | 0.8786 | 0.88562 |
| 03 | XGBoost | 0.8862 | 0.89125 |
| 04 | XGBoost + engineered features | 0.8867 | 0.88500 |
| 05 | Ensemble (mean of RF + XGBoost probabilities) | 0.8857 | **0.89250** |
| 06 | CatBoost | 0.8863 | 0.89062 |
| 07 | Softmax regression (degree-3 polynomial) | **0.8926** | 0.88562 |

Best public leaderboard score: **0.89250** (`submissions/submission_05_ensemble.csv`).

Main findings: tree models clearly beat plain logistic regression; engineered features help
logistic regression (+0.024) but not the tree models; polynomial softmax reaches the best CV
accuracy (0.8926); all other strong models plateau at about 0.886 CV accuracy, so leaderboard differences of a few thousandths are within noise
(about +-0.005 on 4,000 test rows). Details are in `results.md` and the project report.
