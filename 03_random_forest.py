"""Step 3: Random forest with randomized hyperparameter search.

Run from the repo root:  python src/03_random_forest.py
Writes submissions/submission_02_rf.csv and appends to results.md.
"""
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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

# Trees do not need scaling, only imputation + encoding
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), CAT),
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
])

param_dist = {
    "clf__n_estimators": [200, 400, 600],
    "clf__max_depth": [8, 12, 16, 20, None],
    "clf__min_samples_leaf": [1, 2, 4, 8],
    "clf__min_samples_split": [2, 5, 10],
    "clf__max_features": ["sqrt", "log2", 0.5],
    "clf__class_weight": [None, "balanced"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
search = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=cv,
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

# Feature importances of the final model
rf = search.best_estimator_.named_steps["clf"]
names = NUM + list(search.best_estimator_.named_steps["pre"]
                   .named_transformers_["cat"].named_steps["oh"].get_feature_names_out(CAT))
imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
print("\nFeature importances:")
print(imp.round(4).to_string())

pred = search.best_estimator_.predict(X_test)
sub = pd.DataFrame({ID: test[ID], TARGET: pred.astype(int)})
path = os.path.join(SUB_DIR, "submission_02_rf.csv")
sub.to_csv(path, index=False)
print("\nSaved", path, "| class counts:", sub[TARGET].value_counts().sort_index().to_dict())

short = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
with open("results.md", "a") as f:
    f.write(f"| 02 | Random Forest | {short} | {search.best_score_:.4f} | (fill in) |\n")
