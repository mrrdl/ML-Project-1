"""Step 1: Exploratory data analysis.

Run from the repo root:  python src/01_eda.py
Saves plots to eda_plots/ and prints key statistics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

DATA_DIR = "data"
OUT_DIR = "eda_plots"
os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "student_dataset.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print("Train shape:", train.shape, "| Test shape:", test.shape)
print("\n--- dtypes ---")
print(train.dtypes)

print("\n--- missing values (train) ---")
print(train.isna().sum()[train.isna().sum() > 0])
print("--- missing values (test) ---")
print(test.isna().sum()[test.isna().sum() > 0])

print("\n--- duplicates ---")
print("duplicate rows (excluding id):",
      train.drop(columns=["unit_id"]).duplicated().sum())
print("duplicate unit_id:", train["unit_id"].duplicated().sum())

print("\n--- class balance ---")
print(train["condition"].value_counts().sort_index())
print(train["condition"].value_counts(normalize=True).sort_index().round(4))

print("\n--- unit_type ---")
print(train["unit_type"].value_counts())
print(pd.crosstab(train["unit_type"], train["condition"], normalize="index").round(3))
print("test unit_type:", test["unit_type"].value_counts().to_dict())

num_cols = [c for c in train.columns
            if c not in ("unit_id", "unit_type", "condition")]

print("\n--- describe ---")
print(train[num_cols].describe().T.round(3))

print("\n--- mean of each feature by class ---")
print(train.groupby("condition")[num_cols].mean().T.round(3))

print("\n--- correlation with target ---")
print(train[num_cols + ["condition"]].corr()["condition"].drop("condition")
      .sort_values().round(3))

print("\n--- train vs test mean (distribution shift check) ---")
cmp = pd.DataFrame({"train": train[num_cols].mean(), "test": test[num_cols].mean()})
cmp["diff_%"] = ((cmp["test"] - cmp["train"]) / cmp["train"].abs() * 100).round(2)
print(cmp.round(3))

# ---- plots ----
fig, axes = plt.subplots(3, 4, figsize=(18, 11))
for ax, col in zip(axes.ravel(), num_cols):
    for k in sorted(train["condition"].unique()):
        ax.hist(train.loc[train["condition"] == k, col], bins=40,
                alpha=0.5, label=f"class {k}", density=True)
    ax.set_title(col)
axes.ravel()[0].legend()
for ax in axes.ravel()[len(num_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_hist_by_class.png"), dpi=110)
plt.close()

corr = train[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
fig.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"), dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(4, 4))
train["condition"].value_counts().sort_index().plot(kind="bar", ax=ax)
ax.set_title("Class distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_balance.png"), dpi=110)
plt.close()
print("\nPlots saved to", OUT_DIR)
