import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Import XGBoost with the specific callback for early stopping
from xgboost import XGBClassifier, XGBRegressor
from xgboost.callback import EarlyStopping # NEW IMPORT for modern XGBoost early stopping
from sklearn.ensemble import IsolationForest 
from sklearn.model_selection import StratifiedKFold 
import matplotlib.pyplot as plt
import xgboost as xgb
import shap 
import warnings

# Suppress harmless warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # Suppress the date parsing warning

# Load data
df = pd.read_csv("Austin-2019-01-01-to-2023-07-22.csv")

# DATA CLEANING - SOP 1: Sensitivity to Data Quality and Outliers
print("Step 1: Data Cleaning & Outlier Handling — Preparing dataset for modeling...")

# Select relevant columns
cols = ["datetime", "tempmax", "tempmin", "temp", "humidity",
        "windspeed", "sealevelpressure", "solarradiation", "precip"]
df = df[cols].copy()

# Remove duplicate records (if any)
duplicates = df.duplicated().sum()
if duplicates > 0:
    df = df.drop_duplicates()
    print(f" - Removed {duplicates} duplicate rows.")

# Handle missing values
missing_count = df.isna().sum().sum()
if missing_count > 0:
    df = df.dropna()
    print(f" - Removed rows with missing values.")
else:
    print(" - No missing values found.")

# Remove or fix invalid entries
df["precip"] = np.where(df["precip"] < 0, 0, df["precip"])      # No negative rainfall
df["humidity"] = np.clip(df["humidity"], 0, 100)                # Valid humidity range
df["windspeed"] = np.clip(df["windspeed"], 0, None)             # No negative windspeed

# --- New Method: IQR-based Capping for Outliers (SOP 1) ---
def cap_outliers_iqr(series, factor=1.5):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return np.clip(series, lower_bound, upper_bound)

for col in ["tempmax", "tempmin"]:
    # Using a higher factor (3.0) for weather data to be less aggressive
    df[col] = cap_outliers_iqr(df[col], factor=3.0) 

# Reset index after cleaning
df = df.reset_index(drop=True)
print(f"Data cleaning complete. Final dataset shape: {df.shape}\n")

# FEATURE ENGINEERING
# Parse datetime and create features
# Suppress warning by specifying the format if known, or letting pandas infer it once
df["datetime"] = pd.to_datetime(df["datetime"], errors='coerce') 
df = df.dropna(subset=['datetime']) # Drop rows where datetime couldn't be parsed

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

# Improved features
features = [
    "tempmax", "tempmin", 
    "humidity", "windspeed", "sealevelpressure", "solarradiation",
    "month", "day_of_year", 
    "precip_lag1", "humidity_lag1", "sealevelpressure_lag1", 
    "precip_roll3", "humidity_roll3"
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
# CLASSIFICATION MODEL - Predict Rain Occurrence (SOP 2: Fix TypeError and add stability)
# =============================================================================
print("Step 2: Training Classification Model with Model Stability Enhancements...")

# Define Stratified K-Fold for robust early stopping evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Using one fold as a validation set for early stopping
# Check if the split is possible (need at least 2 samples per class in each fold)
try:
    train_idx, val_idx = next(skf.split(X_train, y_train_class))
except ValueError as e:
    print(f"Warning: Could not perform stratified split. Falling back to simple split. Error: {e}")
    # Fallback to a simpler, non-stratified split if the above fails (e.g., if one class is too small)
    from sklearn.model_selection import train_test_split
    X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
        X_train, y_train_class, test_size=0.2, random_state=42, shuffle=True
    )
else:
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train_class.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train_class.iloc[val_idx]

# Define the EarlyStopping callback (Fixes the TypeError)
early_stop = EarlyStopping(
    rounds=50, 
    maximize=False, # Minimize 'logloss'
    save_best=True,
    verbose=False
)

clf = XGBClassifier(
    learning_rate=0.1, 
    max_depth=6,
    n_estimators=1000, 
    colsample_bytree=0.8,
    subsample=0.8,
    scale_pos_weight=len(y_train_class[y_train_class==0]) / len(y_train_class[y_train_class==1]),
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False 
)

# Fit the model using the 'callbacks' parameter (Fixes the TypeError)
clf.fit(
    X_train_fold, y_train_fold,
    eval_set=[(X_val_fold, y_val_fold)],
    callbacks=[early_stop] # FIXED: Using callbacks instead of early_stopping_rounds
)
y_pred_class = clf.predict(X_test)

print("="*60)
print("CLASSIFICATION RESULTS - Rain Occurrence (Yes/No)")
print("="*60)
print(classification_report(y_test_class, y_pred_class, target_names=["No Rain", "Rain"]))
print(f"F1 Score: {f1_score(y_test_class, y_pred_class):.3f}")

# =============================================================================
# REGRESSION MODEL - Predict Rainfall Amount (SOP 1: Implement Huber Loss)
# =============================================================================
# Filter to only rainy days for regression
rainy_train = y_train_reg > 0
rainy_test = y_test_reg > 0

X_train_rainy = X_train[rainy_train]
y_train_rainy = y_train_reg[rainy_train]
X_test_rainy = X_test[rainy_test]
y_test_rainy = y_test_reg[rainy_test]

print(f"\nStep 3: Training Regression Model on rainy days only:")
print(f"Train rainy samples: {len(X_train_rainy)}, Test rainy samples: {len(X_test_rainy)}\n")

# --- Implement Huber Loss (Robust Loss Function) ---
reg = XGBRegressor(
    learning_rate=0.05,
    max_depth=5,
    n_estimators=500,
    colsample_bytree=0.8,
    subsample=0.8,
    random_state=42,
    # FIXED: Using 'reg:huber' (Robust Loss Function) as suggested in SOP 1
    objective='reg:huber' 
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

valid_mape_mask = y_test_rainy > 0.01
if valid_mape_mask.sum() > 0:
    mape = np.mean(np.abs((y_test_rainy[valid_mape_mask] - y_pred_rainy[valid_mape_mask]) / 
                          y_test_rainy[valid_mape_mask])) * 100
    print(f"MAPE: {mape:.2f}% (calculated on {valid_mape_mask.sum()} samples with precip > 0.01)")

# =============================================================================
# VISUALIZATIONS & INTERPRETABILITY (SOP 3: SHAP-Based Feature Evaluation)
# =============================================================================
print("\nStep 4: Model Interpretability (SHAP) and Visualizations...")

# Initialize SHAP Explainer
# Check if SHAP is available before running plots
if 'shap' in globals() and len(X_test) > 0 and len(X_test_rainy) > 0:
    try:
        explainer_clf = shap.TreeExplainer(clf)
        # Note: shap_values for binary classification returns a list of two arrays. We typically use the 1st index (class 1: Rain)
        shap_values_clf = explainer_clf.shap_values(X_test)[1] 
        explainer_reg = shap.TreeExplainer(reg)
        shap_values_reg = explainer_reg.shap_values(X_test_rainy)

        fig, axes = plt.subplots(3, 2, figsize=(18, 18))

        # 1. Classification SHAP Summary Plot (Feature Importance)
        shap.summary_plot(shap_values_clf, X_test, plot_type="bar", show=False, ax=axes[0, 0])
        axes[0, 0].set_title("SHAP Feature Importance (Classification - Rain Occurrence)", fontsize=14) 

        # 2. Regression SHAP Summary Plot (Feature Importance)
        shap.summary_plot(shap_values_reg, X_test_rainy, plot_type="bar", show=False, ax=axes[0, 1])
        axes[0, 1].set_title("SHAP Feature Importance (Regression - Rainfall Amount)", fontsize=14)

        # 3. Predicted vs Actual - Regression (Original Plot)
        axes[1, 0].scatter(y_test_rainy, y_pred_rainy, alpha=0.5)
        axes[1, 0].plot([0, y_test_rainy.max()], [0, y_test_rainy.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel("Actual Rainfall (inches)")
        axes[1, 0].set_ylabel("Predicted Rainfall (inches)")
        axes[1, 0].set_title(f"Predicted vs Actual Rainfall\n(R² = {r2:.3f})", fontsize=14)
        axes[1, 0].grid(alpha=0.3)

        # 4. Residuals distribution (Original Plot)
        residuals = y_test_rainy - y_pred_rainy
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel("Residual (Actual - Predicted)")
        axes[1, 1].set_ylabel("Frequency")
        axes[1, 1].set_title(f"Residuals Distribution\n(MAE = {mae:.3f} inches)", fontsize=14)
        axes[1, 1].grid(alpha=0.3)

        # 5. Classification SHAP Dependence Plot (Example: Top Feature)
        # Find the index of the top feature for the plot
        top_feature_index_clf = np.argsort(np.abs(shap_values_clf).mean(0))[::-1][0]
        shap.dependence_plot(top_feature_index_clf, shap_values_clf, X_test, 
                             ax=axes[2, 0], show=False, interaction_index="auto")
        axes[2, 0].set_title(f"SHAP Dependence Plot (Classification) - {X.columns[top_feature_index_clf]}", fontsize=14)

        # 6. Regression SHAP Dependence Plot (Example: Top Feature)
        top_feature_index_reg = np.argsort(np.abs(shap_values_reg).mean(0))[::-1][0]
        shap.dependence_plot(top_feature_index_reg, shap_values_reg, X_test_rainy, 
                             ax=axes[2, 1], show=False, interaction_index="auto")
        axes[2, 1].set_title(f"SHAP Dependence Plot (Regression) - {X.columns[top_feature_index_reg]}", fontsize=14)

        plt.tight_layout()
        plt.show() 
    except Exception as e:
        print(f"SHAP/Plotting Error: {e}. Ensure all libraries are installed correctly.")
else:
    print("\nSkipping SHAP plotting due to missing dependency or empty test set.")


print("\n" + "="*60)
print("SUMMARY OF FIXES AND IMPROVEMENTS:")
print("="*60)
print("1. **CRITICAL FIX:** Replaced `early_stopping_rounds` with the modern **`callbacks=[EarlyStopping(...)]`** method to resolve the `TypeError` in XGBoost.")
print("2. **SOP 1 Enhancement:** Implemented **`objective='reg:huber'`** for the Regression model to use a **Robust Loss Function**, reducing sensitivity to rainfall outliers.")
print("3. **SOP 2 Enhancement:** Stratified K-Fold logic improved with better error handling.")
print("4. **SOP 3 Enhancement:** SHAP plotting logic updated for binary classification output.")