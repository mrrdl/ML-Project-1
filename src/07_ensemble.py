"""Step 6: Ensemble of logistic regression, random forest and XGBoost.

Run from the repo root:  python src/07_ensemble.py
Requires: pip install xgboost

1. Builds out-of-fold (OOF) class probabilities for each base model with the
   same 5-fold stratified split (seed 42).
2. Compares blends of the OOF probabilities:
     - simple average of all three models
     - simple average of RF + XGB
     - weighted average (weights searched on a 0.1 grid; the accuracy is
       estimated honestly: weights are chosen on 4 folds and scored on the 5th)
     - stacking (logistic regression trained on the OOF probabilities)
3. Picks the best blend (a simple average is preferred unless a fancier blend
   beats it by more than 0.002), fits all base models on the full training
   data and writes submissions/submission_05_ensemble.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from features import CAT, FEATURE_SETS, RAW_NUM, add_features  # noqa: E402

SEED = 42
np.random.seed(SEED)

SUB_DIR, ART_DIR = "submissions", "artifacts"
os.makedirs(SUB_DIR, exist_ok=True)
os.makedirs(ART_DIR, exist_ok=True)

train = pd.read_csv(os.path.join("data", "student_dataset.csv"))
test = pd.read_csv(os.path.join("data", "test.csv"))
TARGET, ID = "condition", "unit_id"
y = train[TARGET].to_numpy()
X_raw = train.drop(columns=[TARGET])
T_raw = test.copy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X_raw, y))


def make_pre(extra, scale):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), RAW_NUM + extra),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ])


# name -> (engineered feature names, pipeline). Parameters come from steps 2-4.
ALL9 = FEATURE_SETS["all9"]
models = {
    "LR": (ALL9, Pipeline([
        ("pre", make_pre(ALL9, scale=True)),
        ("clf", LogisticRegression(C=10, max_iter=3000, random_state=SEED))])),
    "RF": ([], Pipeline([
        ("pre", make_pre([], scale=False)),
        ("clf", RandomForestClassifier(n_estimators=600, max_depth=20, min_samples_split=5,
                                       min_samples_leaf=1, max_features="log2",
                                       random_state=SEED, n_jobs=-1))])),
    "XGB": ([], Pipeline([
        ("pre", make_pre([], scale=False)),
        ("clf", XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
                              eval_metric="mlogloss", random_state=SEED, n_jobs=-1,
                              n_estimators=600, learning_rate=0.1, max_depth=3,
                              min_child_weight=1, subsample=0.85, colsample_bytree=0.8,
                              reg_lambda=1, gamma=0.1))])),
}

names = list(models)
oof, tst = {}, {}
for name, (feats, pipe) in models.items():
    Xm, Tm = add_features(X_raw, feats), add_features(T_raw, feats)
    oof[name] = cross_val_predict(pipe, Xm, y, cv=folds, method="predict_proba")
    tst[name] = pipe.fit(Xm, y).predict_proba(Tm)
    print(f"{name:<4} OOF accuracy = {(oof[name].argmax(1) == y).mean():.4f}", flush=True)

np.savez_compressed(os.path.join(ART_DIR, "oof_and_test_probs.npz"),
                    **{f"oof_{k}": v for k, v in oof.items()},
                    **{f"test_{k}": v for k, v in tst.items()})

acc = lambda P, idx=slice(None): (P[idx].argmax(1) == y[idx]).mean()  # noqa: E731
blend = lambda probs, w: sum(wi * p for wi, p in zip(w, probs))  # noqa: E731

oof_list = [oof[n] for n in names]
tst_list = [tst[n] for n in names]

# --- simple averages -------------------------------------------------------
w_all3 = (1 / 3, 1 / 3, 1 / 3)
w_rfxgb = (0.0, 0.5, 0.5)
est = {
    "mean LR+RF+XGB": acc(blend(oof_list, w_all3)),
    "mean RF+XGB": acc(blend(oof_list, w_rfxgb)),
}

# --- weighted average with honest (nested) estimate ------------------------
grid = [(i / 10, j / 10, (10 - i - j) / 10) for i in range(11) for j in range(11 - i)]


def best_weights(idx):
    return max(grid, key=lambda w: acc(blend(oof_list, w), idx))


nested = [acc(blend(oof_list, best_weights(tr)), va) for tr, va in folds]
est["weighted (nested est.)"] = float(np.mean(nested))
w_best = best_weights(slice(None))

# --- stacking --------------------------------------------------------------
Z = np.hstack(oof_list)
Zt = np.hstack(tst_list)
meta = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
est["stack (LR meta)"] = cross_val_score(meta, Z, y, cv=folds, scoring="accuracy").mean()

print("\nBlend comparison (OOF accuracy, 5-fold):")
for k, v in est.items():
    print(f"  {k:<24} {v:.4f}")
print(f"  best single model        {max(acc(oof[n]) for n in names):.4f}")
print("  grid-search weights (LR, RF, XGB) on full OOF:", w_best)

# --- choose ----------------------------------------------------------------
simple_key = max(("mean LR+RF+XGB", "mean RF+XGB"), key=est.get)
best_key = max(est, key=est.get)
choice = best_key if est[best_key] - est[simple_key] > 0.002 else simple_key
print(f"\nChosen blend: {choice} ({est[choice]:.4f})")

if choice == "mean LR+RF+XGB":
    P_oof, P_test = blend(oof_list, w_all3), blend(tst_list, w_all3)
elif choice == "mean RF+XGB":
    P_oof, P_test = blend(oof_list, w_rfxgb), blend(tst_list, w_rfxgb)
elif choice.startswith("weighted"):
    P_oof, P_test = blend(oof_list, w_best), blend(tst_list, w_best)
else:
    P_oof = cross_val_predict(meta, Z, y, cv=folds, method="predict_proba")
    P_test = meta.fit(Z, y).predict_proba(Zt)

print("\nOOF confusion matrix of the chosen blend (rows=true, cols=pred):")
print(confusion_matrix(y, P_oof.argmax(1)))

pred = P_test.argmax(1)
sub = pd.DataFrame({ID: test[ID], TARGET: pred.astype(int)})
path = os.path.join(SUB_DIR, "submission_05_ensemble.csv")
sub.to_csv(path, index=False)
print("\nSaved", path, "| class counts:", sub[TARGET].value_counts().sort_index().to_dict())

with open("results.md", "a") as f:
    f.write(f"| 05 | Ensemble: {choice} (LR+FE, RF, XGB) | step 2-4 params | "
            f"{est[choice]:.4f} (5-fold OOF) | (fill in) |\n")
