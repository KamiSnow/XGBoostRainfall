import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Austin-2019-01-01-to-2023-07-22.csv")

# DATA CLEANING
print("Step 1: Data Cleaning — Preparing dataset for modeling...")

# Select relevant columns
cols = ["datetime", "tempmax", "tempmin", "temp", "humidity",
        "windspeed", "sealevelpressure", "solarradiation", "precip"]
df = df[cols].copy()

# Remove duplicate records (if any)
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f" - Removed {duplicates} duplicate rows.")

# Check for and remove missing values
missing_count = df.isna().sum().sum()
if missing_count > 0:
    df = df.dropna()
    print(f" - Removed {missing_count} rows with missing values.")
else:
    print(" - No missing values found.")

# Remove or fix invalid entries
df["precip"] = np.where(df["precip"] < 0, 0, df["precip"])   # No negative rainfall
df["humidity"] = np.clip(df["humidity"], 0, 100)             # Valid humidity range
df["windspeed"] = np.clip(df["windspeed"], 0, None)          # No negative windspeed

# Reset index after cleaning
df = df.reset_index(drop=True)

print(f"Data cleaning complete. Final dataset shape: {df.shape}\n")

# Select relevant columns
cols = ["datetime", "tempmax", "tempmin", "temp", "humidity",
        "windspeed", "sealevelpressure", "solarradiation", "precip"]
df = df[cols].copy()
df = df.dropna()

# Parse datetime and create features
df["datetime"] = pd.to_datetime(df["datetime"])
df["rain"] = (df["precip"] > 0).astype(int)
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year
df["day_of_year"] = df["datetime"].dt.dayofyear

# Add lagged features (previous day's weather)
df = df.sort_values("datetime")
for col in ["precip", "humidity", "sealevelpressure"]:
    df[f"{col}_lag1"] = df[col].shift(1)
df = df.dropna()

# Create rolling features (3-day moving averages)
df["precip_roll3"] = df["precip"].rolling(window=3, min_periods=1).mean()
df["humidity_roll3"] = df["humidity"].rolling(window=3, min_periods=1).mean()

# Improved features (remove multicollinearity, add temporal patterns)
features = [
    "tempmax", "tempmin",  # Remove 'temp' to reduce multicollinearity
    "humidity", "windspeed", "sealevelpressure", "solarradiation",
    "month", "day_of_year",  # Better temporal features
    "precip_lag1", "humidity_lag1", "sealevelpressure_lag1",  # Lagged features
    "precip_roll3", "humidity_roll3"  # Rolling features
]

X = df[features]
y_class = df["rain"]
y_reg = df["precip"]

# Temporal split (train: 2019-2022, test: 2023)
train_mask = df["year"] < 2023
test_mask = df["year"] == 2023

X_train, y_train_class, y_train_reg = X[train_mask], y_class[train_mask], y_reg[train_mask]
X_test, y_test_class, y_test_reg = X[test_mask], y_class[test_mask], y_reg[test_mask]

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Rain days in train: {y_train_class.sum()} ({y_train_class.mean()*100:.1f}%)")
print(f"Rain days in test: {y_test_class.sum()} ({y_test_class.mean()*100:.1f}%)\n")

# =============================================================================
# CLASSIFICATION MODEL - Predict Rain Occurrence
# =============================================================================
clf = XGBClassifier(
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    colsample_bytree=0.8,
    subsample=0.8,
    scale_pos_weight=len(y_train_class[y_train_class==0]) / len(y_train_class[y_train_class==1]),  # Handle imbalance
    eval_metric='logloss',
    random_state=42
)
clf.fit(X_train, y_train_class)
y_pred_class = clf.predict(X_test)

print("="*60)
print("CLASSIFICATION RESULTS - Rain Occurrence (Yes/No)")
print("="*60)
print(classification_report(y_test_class, y_pred_class, target_names=["No Rain", "Rain"]))
print(f"F1 Score: {f1_score(y_test_class, y_pred_class):.3f}")

# =============================================================================
# REGRESSION MODEL - Predict Rainfall Amount (Only on Rainy Days)
# =============================================================================
# Filter to only rainy days for regression
rainy_train = y_train_reg > 0
rainy_test = y_test_reg > 0

X_train_rainy = X_train[rainy_train]
y_train_rainy = y_train_reg[rainy_train]
X_test_rainy = X_test[rainy_test]
y_test_rainy = y_test_reg[rainy_test]

print(f"\nRegression on rainy days only:")
print(f"Train rainy samples: {len(X_train_rainy)}, Test rainy samples: {len(X_test_rainy)}\n")

reg = XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    n_estimators=500,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42
)
reg.fit(X_train_rainy, y_train_rainy)
y_pred_rainy = reg.predict(X_test_rainy)

# Ensure non-negative predictions
y_pred_rainy = np.maximum(y_pred_rainy, 0)

print("="*60)
print("REGRESSION RESULTS - Rainfall Amount (on rainy days)")
print("="*60)
mae = mean_absolute_error(y_test_rainy, y_pred_rainy)
rmse = np.sqrt(mean_squared_error(y_test_rainy, y_pred_rainy))
r2 = r2_score(y_test_rainy, y_pred_rainy)

print(f"MAE:  {mae:.3f} inches")
print(f"RMSE: {rmse:.3f} inches")
print(f"R²:   {r2:.3f}")

# Calculate MAPE only for values > 0.01 to avoid division issues
valid_mape_mask = y_test_rainy > 0.01
if valid_mape_mask.sum() > 0:
    mape = np.mean(np.abs((y_test_rainy[valid_mape_mask] - y_pred_rainy[valid_mape_mask]) / 
                           y_test_rainy[valid_mape_mask])) * 100
    print(f"MAPE: {mape:.2f}% (calculated on {valid_mape_mask.sum()} samples with precip > 0.01)")

# =============================================================================
# VISUALIZATIONS
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Feature importance - Classification
import xgboost as xgb
xgb.plot_importance(clf, importance_type='gain', max_num_features=10, ax=axes[0, 0])
axes[0, 0].set_title("Feature Importance - Rain Occurrence (Classification)")

# 2. Feature importance - Regression
xgb.plot_importance(reg, importance_type='gain', max_num_features=10, ax=axes[0, 1])
axes[0, 1].set_title("Feature Importance - Rainfall Amount (Regression)")

# 3. Predicted vs Actual - Regression
axes[1, 0].scatter(y_test_rainy, y_pred_rainy, alpha=0.5)
axes[1, 0].plot([0, y_test_rainy.max()], [0, y_test_rainy.max()], 'r--', lw=2)
axes[1, 0].set_xlabel("Actual Rainfall (inches)")
axes[1, 0].set_ylabel("Predicted Rainfall (inches)")
axes[1, 0].set_title(f"Predicted vs Actual Rainfall\n(R² = {r2:.3f})")
axes[1, 0].grid(alpha=0.3)

# 4. Residuals distribution
residuals = y_test_rainy - y_pred_rainy
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel("Residual (Actual - Predicted)")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title(f"Residuals Distribution\n(MAE = {mae:.3f} inches)")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY IMPROVEMENTS MADE:")
print("="*60)
print("1. Added lagged features (previous day's weather)")
print("2. Added rolling averages (3-day patterns)")
print("3. Removed multicollinear features (kept tempmax/tempmin, removed temp)")
print("4. Used scale_pos_weight to handle class imbalance")
print("5. Split regression to predict only on rainy days")
print("6. Improved temporal features (day_of_year instead of day)")
print("7. Fixed MAPE calculation to avoid division by near-zero values")