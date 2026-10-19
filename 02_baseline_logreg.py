"""Step 2: Baseline logistic regression.

Run from the repo root:  python src/02_baseline_logreg.py
Writes submissions/submission_01_logreg.csv and appends to results.md.
"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data"
SUB_DIR = "submissions"
os.makedirs(SUB_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "student_dataset.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET = "condition"
ID = "unit_id"
CAT = ["unit_type"]
NUM = [c for c in train.columns if c not in (ID, TARGET) + tuple(CAT)]

X = train[CAT + NUM]
y = train[TARGET]
X_test = test[CAT + NUM]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Small grid over regularisation strength (documented for the report)
param_grid = {"clf__C": [0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
gs.fit(X, y)

print("C grid results (mean CV accuracy):")
for c, s in zip(param_grid["clf__C"], gs.cv_results_["mean_test_score"]):
    print(f"  C={c:<6} -> {s:.4f}")
print(f"\nBest: {gs.best_params_}  CV accuracy = {gs.best_score_:.4f}")

# Confusion matrix from out-of-fold predictions of the best model
oof = cross_val_predict(gs.best_estimator_, X, y, cv=cv)
print("\nOut-of-fold confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y, oof))

# Final fit on all training data and predict test
final = gs.best_estimator_.fit(X, y)
pred = final.predict(X_test)

sub = pd.DataFrame({ID: test[ID], TARGET: pred.astype(int)})
path = os.path.join(SUB_DIR, "submission_01_logreg.csv")
sub.to_csv(path, index=False)
print("\nSaved", path, "| predicted class counts:",
      sub[TARGET].value_counts().sort_index().to_dict())

with open("results.md", "a") as f:
    f.write(f"| 01 | Logistic Regression | {gs.best_params_} | "
            f"{gs.best_score_:.4f} | (fill in Kaggle score) |\n")
