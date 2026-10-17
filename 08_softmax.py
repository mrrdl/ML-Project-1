"""Step 8: Polynomial softmax (multinomial logistic) regression.

Run from the repo root:  python src/08_softmax.py
Writes submissions/submission_07_softmax.csv and appends to results.md.

Idea: plain logistic regression is weak (step 2) because the class boundaries are
curved. Seven engineered features + one-hot unit_type are expanded with degree-3
polynomial terms, so the linear softmax scores act on products and powers of the
sensors. Strong L2 regularisation (C = 0.1) controls the many cubic terms.
"""
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data"
SUB_DIR = "submissions"
os.makedirs(SUB_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "student_dataset.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

TARGET, ID = "condition", "unit_id"
y = train[TARGET]


def fe(df):
    """Seven engineered features + one-hot unit_type (all float)."""
    df = df.drop(columns=[ID, TARGET], errors="ignore").copy()
    df["temp_diff"] = df.process_temp - df.ambient_temp
    df["pw"] = df.drive_torque * df.shaft_speed
    df["wear_torque"] = df.component_wear * df.drive_torque
    df["cool_ratio"] = df.temp_diff / (df.coolant_flow + 1e-6)
    df["power_resid"] = df.power_load - df.pw
    df["wear_vib"] = df.component_wear * df.vibration_index
    df["ts_cool"] = df.thermal_stress / (df.coolant_flow + 1e-6)
    return pd.get_dummies(df, columns=["unit_type"]).astype(float)


X = fe(train)
X_test = fe(test).reindex(columns=X.columns, fill_value=0.0)

DEGREE, C = 3, 0.1
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    PolynomialFeatures(DEGREE),
    StandardScaler(),            # puts polynomial terms on one scale for the L2 penalty
    LogisticRegression(C=C, max_iter=5000),
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Out-of-fold predictions give both the CV accuracy and the confusion matrix
oof = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
cv_acc = (oof == y).mean()
print(f"Degree-{DEGREE} softmax, C={C}: OOF accuracy = {cv_acc:.4f}")
print("\nOut-of-fold confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y, oof))

# Final fit on all training data and predict test
model.fit(X, y)
pred = model.predict(X_test)

sub = pd.DataFrame({ID: test[ID], TARGET: pred.astype(int)})
sub = sample[[ID]].merge(sub, on=ID, how="left")
assert sub[TARGET].notna().all() and len(sub) == len(sample)
sub[TARGET] = sub[TARGET].astype(int)
path = os.path.join(SUB_DIR, "submission_07_softmax.csv")
sub.to_csv(path, index=False)
print("\nSaved", path, "| predicted class counts:",
      sub[TARGET].value_counts().sort_index().to_dict())

with open("results.md", "a") as f:
    f.write(f"| 07 | Softmax (polynomial) | {{'degree': {DEGREE}, 'C': {C}}} | "
            f"{cv_acc:.4f} | (fill in Kaggle score) |\n")
