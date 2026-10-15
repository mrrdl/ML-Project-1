"""Step 5b: XGBoost with engineered features.

Run from the repo root:  python src/06_xgboost_fe.py
Requires: pip install xgboost

Uses the best XGBoost parameters found in step 4 (fixed, not re-tuned) and
compares raw features against engineered feature sets with 3x repeated 5-fold
CV (seed 42). The feature set with the best mean CV accuracy is used to fit on
all training data and write submissions/submission_04_xgb_fe.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from features import CAT, FEATURE_SETS, RAW_NUM, add_features  # noqa: E402

SEED = 42
np.random.seed(SEED)

SUB_DIR = "submissions"
os.makedirs(SUB_DIR, exist_ok=True)

train = pd.read_csv(os.path.join("data", "student_dataset.csv"))
test = pd.read_csv(os.path.join("data", "test.csv"))
TARGET, ID = "condition", "unit_id"
y = train[TARGET]

# Best parameters from step 4
BEST = dict(n_estimators=600, learning_rate=0.1, max_depth=3, min_child_weight=1,
            subsample=0.85, colsample_bytree=0.8, reg_lambda=1, gamma=0.1)


def build(extra):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), RAW_NUM + extra),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    clf = XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
                        eval_metric="mlogloss", random_state=SEED, n_jobs=-1, **BEST)
    return Pipeline([("pre", pre), ("clf", clf)])


FEATURE_SETS["thermal3+mech4"] = FEATURE_SETS["thermal3"] + FEATURE_SETS["mech4"]
candidates = ["raw", "thermal3", "thermal3+mech4", "all9"]

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
scores = {}
for name in candidates:
    names = FEATURE_SETS[name]
    X = add_features(train.drop(columns=[TARGET]), names)
    s = cross_val_score(build(names), X, y, cv=cv, scoring="accuracy")
    scores[name] = s.mean()
    print(f"{name:<16} CV accuracy = {s.mean():.4f} (+-{s.std():.4f})", flush=True)

best_name = max(scores, key=scores.get)
print(f"\nBest feature set: {best_name} ({scores[best_name]:.4f}); "
      f"raw was {scores['raw']:.4f}")

names = FEATURE_SETS[best_name]
model = build(names).fit(add_features(train.drop(columns=[TARGET]), names), y)
pred = model.predict(add_features(test.drop(columns=[ID]), names))

sub = pd.DataFrame({ID: test[ID], TARGET: pred.astype(int)})
path = os.path.join(SUB_DIR, "submission_04_xgb_fe.csv")
sub.to_csv(path, index=False)
print("Saved", path, "| class counts:", sub[TARGET].value_counts().sort_index().to_dict())

with open("results.md", "a") as f:
    f.write(f"| 04 | XGBoost + features ({best_name}) | step-4 params | "
            f"{scores[best_name]:.4f} (3x5 CV) | (fill in) |\n")
