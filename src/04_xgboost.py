"""Step 4: XGBoost with randomized hyperparameter search (raw features only).

Run from the repo root:  python src/04_xgboost.py
Requires: pip install xgboost
Writes submissions/submission_03_xgb.csv and appends to results.md.
"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data"
SUB_DIR = "submissions"
os.makedirs(SUB_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "student_dataset.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

TARGET, ID = "condition", "unit_id"
CAT = ["unit_type"]
NUM = [c for c in train.columns if c not in (ID, TARGET) + tuple(CAT)]

X, y = train[CAT + NUM], train[TARGET]
X_test = test[CAT + NUM]

# Same preprocessing as the random forest so the comparison is fair
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", XGBClassifier(objective="multi:softprob", num_class=3,
                          tree_method="hist", eval_metric="mlogloss",
                          random_state=SEED, n_jobs=2)),
])

param_dist = {
    "clf__n_estimators": [200, 400, 600],
    "clf__learning_rate": [0.03, 0.05, 0.1],
    "clf__max_depth": [3, 4, 6, 8],
    "clf__min_child_weight": [1, 3, 5],
    "clf__subsample": [0.7, 0.85, 1.0],
    "clf__colsample_bytree": [0.6, 0.8, 1.0],
    "clf__reg_lambda": [1, 5, 10],
    "clf__gamma": [0, 0.1, 0.3],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(pipe, param_dist, n_iter=25, cv=cv,
                            scoring="accuracy", random_state=SEED,
                            n_jobs=1, refit=True)
search.fit(X, y)

res = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
print("Top 5 configurations:")
cols = [c for c in res.columns if c.startswith("param_")] + ["mean_test_score"]
print(res[cols].head(5).to_string(index=False))
print(f"\nBest: {search.best_params_}\nCV accuracy = {search.best_score_:.4f}")

oof = cross_val_predict(search.best_estimator_, X, y, cv=cv)
print("\nOut-of-fold confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y, oof))

pre_fit = search.best_estimator_.named_steps["pre"]
names = NUM + list(pre_fit.named_transformers_["cat"].named_steps["oh"]
                   .get_feature_names_out(CAT))
imp = pd.Series(search.best_estimator_.named_steps["clf"].feature_importances_,
                index=names).sort_values(ascending=False)
print("\nFeature importances:")
print(imp.round(4).to_string())

pred = search.best_estimator_.predict(X_test)
sub = pd.DataFrame({ID: test[ID], TARGET: pred.astype(int)})
path = os.path.join(SUB_DIR, "submission_03_xgb.csv")
sub.to_csv(path, index=False)
print("\nSaved", path, "| class counts:", sub[TARGET].value_counts().sort_index().to_dict())

short = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
with open("results.md", "a") as f:
    f.write(f"| 03 | XGBoost | {short} | {search.best_score_:.4f} | (fill in) |\n")
