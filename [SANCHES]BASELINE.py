# ==========================================================
# Rainfall Prediction (aligned with Sanches et al., 2023)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, XGBRegressor, plot_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# Define the path to the dataset subfolder
DATASET_FOLDER = 'dataset/'

# -----------------------------
# 1. Load and basic cleaning
# -----------------------------
print("Loading and Merging DataFiles...")

try:
    # Load all weather station and pressure data
    naia_daily = pd.read_csv(DATASET_FOLDER + "NAIA Daily Data.csv")
    port_daily = pd.read_csv(DATASET_FOLDER + "Port Area Daily Data.csv")
    science_daily = pd.read_csv(DATASET_FOLDER + "Science Garden Daily Data.csv")
    pressure_daily = pd.read_csv(DATASET_FOLDER + "Mean Sea Level Pressure Daily Data.csv", skiprows=[1])
except FileNotFoundError as e:
    print(f"ERROR: Data file not found: {e}")
    print(f"Please ensure the required CSV files are inside the '{DATASET_FOLDER}' directory.")
    # Exit or raise error if files are missing
    raise

# Standardize datetime columns
for df_ in [naia_daily, port_daily, science_daily]:
    df_['datetime'] = pd.to_datetime(df_[['YEAR', 'MONTH', 'DAY']])
    
pressure_daily['datetime'] = pd.to_datetime(pressure_daily['Date(UTC)'])
pressure_daily = pressure_daily.drop('Date(UTC)', axis=1)

# Function to clean and standardize column names
def clean_weather_data(df_):
    df_ = df_.replace(-999.0, np.nan)
    if 'RAINFALL' in df_.columns:
        df_['RAINFALL'] = df_['RAINFALL'].replace(-1.0, 0.05)
        df_.loc[df_['RAINFALL'] < 0, 'RAINFALL'] = 0
    if 'RH' in df_.columns:
        df_['RH'] = np.clip(df_['RH'], 0, 100)
    if 'WIND_SPEED' in df_.columns:
        df_.loc[df_['WIND_SPEED'] < 0, 'WIND_SPEED'] = 0
    return df_

naia_daily = clean_weather_data(naia_daily).rename(columns={'RAINFALL': 'rainfall_naia', 'TMAX': 'tmax_naia', 'TMIN': 'tmin_naia', 'RH': 'humidity_naia', 'WIND_SPEED': 'windspeed_naia'})
port_daily = clean_weather_data(port_daily).rename(columns={'RAINFALL': 'rainfall_port', 'TMAX': 'tmax_port', 'TMIN': 'tmin_min', 'RH': 'humidity_port', 'WIND_SPEED': 'windspeed_port'})
science_daily = clean_weather_data(science_daily).rename(columns={'RAINFALL': 'rainfall_science', 'TMAX': 'tmax_science', 'TMIN': 'tmin_science', 'RH': 'humidity_science', 'WIND_SPEED': 'windspeed_science'})
pressure_daily = pressure_daily.rename(columns={'NAIA Pasay City, M.Manila Press.QFF.Dly [hPa]': 'pressure_naia', 'Port Area, Manila Press.QFF.Dly [hPa]': 'pressure_port', 'Science Garden Quezon City, Metro Manila Press.QFF.Dly [hPa]': 'pressure_science'})

# --- Merge DataFrames ---
df = naia_daily[['datetime', 'rainfall_naia', 'tmax_naia', 'tmin_naia', 'RH', 'windspeed_naia']].copy()
df = df.merge(port_daily[['datetime', 'rainfall_port', 'tmax_port', 'TMIN', 'RH', 'windspeed_port']], on='datetime', how='outer', suffixes=('_naia', '_port'))
df = df.merge(science_daily[['datetime', 'rainfall_science', 'tmax_science', 'tmin_science', 'RH', 'windspeed_science']], on='datetime', how='outer', suffixes=('_port', '_science'))
df = df.merge(pressure_daily, on='datetime', how='outer')
df = df.sort_values('datetime').reset_index(drop=True)

# --- Aggregate Features (Mean for Temp/Humidity/Wind, Max for Precip) ---
df['tempmax'] = df.filter(regex='tmax_').mean(axis=1)
df['tempmin'] = df.filter(regex='tmin_|TMIN_port').mean(axis=1) # Need to handle TMIN_port explicitly if not renamed earlier
df['humidity'] = df.filter(regex='RH').mean(axis=1)
df['windspeed'] = df.filter(regex='windspeed_').mean(axis=1)
df['sealevelpressure'] = df.filter(regex='pressure_').mean(axis=1)
df['precip'] = df.filter(regex='rainfall_').max(axis=1).fillna(0) # Use max rainfall across stations
df['temp'] = (df['tempmax'] + df['tempmin']) / 2 # Calculate 'temp' as avg of max/min

df = df.dropna(subset=['tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure']).copy()
df['doy'] = df["datetime"].dt.dayofyear
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["rain"] = (df["precip"] > 0).astype(int)

# Note: 'solarradiation' column is missing in the new dataset, so it must be removed from the feature list.

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
    "windspeed","sealevelpressure", # Removed "solarradiation"
    "precip_prev1","sin_doy","cos_doy","month","day"
]

# -----------------------------
# 3. Split train / test
# -----------------------------
# You must adjust the splitting logic since the new data covers a different time range.
# We will split at the end of the 2020 data, similar to the previous pipeline's 80/20 split.
# Using 2021 as the test set for a similar time split approach.
train_df = df[df["year"] < 2021]
test_df  = df[df["year"] >= 2021]

X_train, X_test = train_df[features], test_df[features]
y_train_cls, y_test_cls = train_df["rain"], test_df["rain"]
print(f"Train/Test split: {len(X_train)} training days, {len(X_test)} testing days.")

# ==========================================================
# 4. CLASSIFICATION (Rain occurrence)
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
prec = precision_score(y_test_cls, y_pred_cls, zero_division=0)
rec = recall_score(y_test_cls, y_pred_cls, zero_division=0)
print(f"\n=== Rain Occurrence (Classification) ===")
print(f"Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}")

# ==========================================================
# 5. REGRESSION (Rain amount on rainy days only)
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