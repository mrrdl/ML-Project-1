# Results log

| Step | Model | Best params | 5-fold CV accuracy | Kaggle public score |
|------|-------|-------------|--------------------|---------------------|
| 01 | Logistic Regression | {'clf__C': 10} | 0.8255 | 0.8225 |
| 02 | Random Forest | {'n_estimators': 600, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': 20, 'class_weight': None} | 0.8786 | 0.88562 |
| 03 | XGBoost | {'n_estimators': 600, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.85, 'colsample_bytree': 0.8, 'reg_lambda': 1, 'gamma': 0.1} | 0.8862 | 0.89125 |
| 04 | XGBoost + engineered features (all9), step-4 params | same as 03 | 0.8867 (3x5 CV; raw 0.8856, thermal3 0.8863, thermal3+mech4 0.8864) | 0.88500 |
| 05 | CatBoost (depth 6, lr 0.05, 800 iterations, l2_leaf_reg 5, SqrtBalanced class weights) | see catboost-2.ipynb | 0.8863 | 0.89062 |
| 06 | Softmax regression, degree-3 polynomial (C = 0.1) | {'degree': 3, 'C': 0.1} | 0.8926 | 0.88562 |
| 07 | Ensemble: simple mean of RF + XGBoost (probabilities; LR+RF+XGB mean 0.8853, weighted 0.8856, stack 0.8854 also tried) | step 3 + step 4 params | 0.8857 (5-fold OOF) | 0.89250 |


## Step 5 - feature engineering ablation (3x repeated 5-fold CV, seed 42, parameters fixed)

| Feature set | Logistic Regression | Random Forest |
|-------------|---------------------|---------------|
| raw | 0.8251 | 0.8782 |
| thermal3 (ts_per_cool, ts_per_tdiff, tdiff_per_cool) | 0.8424 | 0.8804 |
| thermal5 (+ temp_diff, power_per_cool) | 0.8432 | 0.8782 |
| mech4 (wear_x_vib, wear_x_load, vib_per_speed, ml_per_torque) | 0.8373 | 0.8790 |
| all9 | 0.8492 | 0.8780 |
| thermal3 + mech4 | 0.8491 | 0.8800 |

Dropped after analysis (single-feature AUC ~0.5-0.6, no CV gain): power_gap, n_missing, temp_ratio, torque_per_speed, calc_power.
