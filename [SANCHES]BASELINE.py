# ==========================================================
#  Rainfall Prediction (aligned with Sanches et al., 2023)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# -----------------------------
# 1. Load and basic cleaning
# -----------------------------
df = pd.read_csv("Austin-2019-01-01-to-2023-07-22.csv")
cols = ["datetime","tempmax","tempmin","temp","humidity",
        "windspeed","sealevelpressure","solarradiation","precip"]
df = df[cols].dropna().copy()

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["doy"] = df["datetime"].dt.dayofyear
df["rain"] = (df["precip"] > 0).astype(int)

# -----------------------------
# 2. Key rainfall features
# -----------------------------
df["precip_prev1"] = df["precip"].shift(1).fillna(0)
df["humidity_prev1"] = df["humidity"].shift(1).fillna(df["humidity"].mean())
df["humidity_roll3"] = df["humidity"].rolling(3).mean().shift(1).fillna(df["humidity"].mean())
df["sin_doy"] = np.sin(2 * np.pi * df["doy"] / 365.25)
df["cos_doy"] = np.cos(2 * np.pi * df["doy"] / 365.25)

features = [
    "tempmax","tempmin","temp","humidity","humidity_prev1","humidity_roll3",
    "windspeed","sealevelpressure","solarradiation",
    "precip_prev1","sin_doy","cos_doy","month","day"
]

# -----------------------------
# 3. Split train / test
# -----------------------------
train_df = df[df["year"] < 2023]
test_df  = df[df["year"] == 2023]

X_train, X_test = train_df[features], test_df[features]
y_train_cls, y_test_cls = train_df["rain"], test_df["rain"]

# ==========================================================
# 4. CLASSIFICATION  (Rain occurrence)
# ==========================================================
clf = XGBClassifier(
    learning_rate=0.05,
    max_depth=10,
    n_estimators=5000,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train_cls)
y_pred_cls = clf.predict(X_test)

acc = accuracy_score(y_test_cls, y_pred_cls)
prec = precision_score(y_test_cls, y_pred_cls)
rec = recall_score(y_test_cls, y_pred_cls)
print(f"\n=== Rain Occurrence (Classification) ===")
print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

# ==========================================================
# 5. REGRESSION  (Rain amount on rainy days only)
# ==========================================================
# Train only on days with measurable rain
train_rain = train_df[train_df["precip"] > 0.1].copy()
test_rain  = test_df[test_df["precip"] > 0.1].copy()

Xr_train, Xr_test = train_rain[features], test_rain[features]
y_train_reg = np.log1p(train_rain["precip"])   # log-transform
y_test_true = test_rain["precip"]

reg = XGBRegressor(
    learning_rate=0.03,
    max_depth=10,
    n_estimators=5000,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    gamma=0.1,
    random_state=42,
    n_jobs=-1
)
reg.fit(Xr_train, y_train_reg)
y_pred_log = reg.predict(Xr_test)
y_pred = np.expm1(y_pred_log)

# --- regression metrics (on rainy days only) ---
mae = mean_absolute_error(y_test_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
r2 = r2_score(y_test_true, y_pred)
mape = np.mean(np.abs((y_test_true - y_pred) /
                      np.maximum(y_test_true, 1e-3))) * 100

print(f"\n=== Rainfall Amount (Regression, rainy days) ===")
print(f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}  MAPE={mape:.1f}%")

# ==========================================================
# 6. OUTLIER FILTER (Generic Adjusted Model)
# ==========================================================
q_hi = df["precip"].quantile(0.995)
train_adj = train_rain[train_rain["precip"] <= q_hi]
test_adj  = test_rain[test_rain["precip"] <= q_hi]
if len(train_adj) > 30:
    reg_adj = XGBRegressor(
        learning_rate=0.03, max_depth=10, n_estimators=5000,
        subsample=0.9, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    reg_adj.fit(train_adj[features], np.log1p(train_adj["precip"]))
    y_pred_adj = np.expm1(reg_adj.predict(test_adj[features]))
    mae_a = mean_absolute_error(test_adj["precip"], y_pred_adj)
    rmse_a = np.sqrt(mean_squared_error(test_adj["precip"], y_pred_adj))
    r2_a = r2_score(test_adj["precip"], y_pred_adj)
    print(f"\n=== Generic Adjusted Model (no outliers) ===")
    print(f"MAE={mae_a:.3f}  RMSE={rmse_a:.3f}  R²={r2_a:.3f}")

# ==========================================================
# 7. Feature Importance Plot
# ==========================================================
plt.figure(figsize=(8,6))
plot_importance(clf, importance_type='weight', max_num_features=10)
plt.title("Feature Importance – Rain Occurrence")
plt.tight_layout()
plt.show()
