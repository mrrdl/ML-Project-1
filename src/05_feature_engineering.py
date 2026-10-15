"""Step 5: Which engineered features help? (repeated-CV ablation with LR and RF)

Run from the repo root:  python src/05_feature_engineering.py
Random forest uses fixed parameters from step 3 (not re-tuned), logistic
regression uses C=10 from step 2, so differences come from the features only.
3x repeated 5-fold stratified CV (seed 42) is used to reduce noise.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from features import CAT, FEATURE_SETS, RAW_NUM, add_features  # noqa: E402

SEED = 42
np.random.seed(SEED)

train = pd.read_csv(os.path.join("data", "student_dataset.csv"))
y = train["condition"]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)


def build(model_name, extra):
    num = RAW_NUM + extra
    if model_name == "LR":
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler())])
        clf = LogisticRegression(C=10, max_iter=3000, random_state=SEED)
    else:
        num_pipe = SimpleImputer(strategy="median")
        clf = RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5,
                                     min_samples_leaf=1, max_features="log2",
                                     random_state=SEED, n_jobs=-1)
    pre = ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", clf)])


def score(model_name, names):
    X = add_features(train.drop(columns=["condition"]), names)
    s = cross_val_score(build(model_name, names), X, y, cv=cv, scoring="accuracy")
    return s.mean(), s.std()


sets = ["raw", "thermal3", "thermal5", "mech4", "all9"]
# also the two sides combined without the weak thermal extras
FEATURE_SETS["thermal3+mech4"] = FEATURE_SETS["thermal3"] + FEATURE_SETS["mech4"]
sets.append("thermal3+mech4")

results = {}
for s in sets:
    lr_m, lr_s = score("LR", FEATURE_SETS[s])
    rf_m, rf_s = score("RF", FEATURE_SETS[s])
    results[s] = (lr_m, rf_m)
    print(f"{s:<16} LR={lr_m:.4f} (+-{lr_s:.4f})   RF={rf_m:.4f} (+-{rf_s:.4f})", flush=True)

b_lr, b_rf = results["raw"]
print("\nChange vs raw features:")
for s in sets[1:]:
    print(f"{s:<16} LR {results[s][0] - b_lr:+.4f}   RF {results[s][1] - b_rf:+.4f}")
