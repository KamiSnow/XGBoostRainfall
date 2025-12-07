import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor
import matplotlib.pyplot as plt

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
print("Step 1: Loading Manila weather datasets...")

# Load daily data from each station
naia_daily = pd.read_csv("NAIA Daily Data.csv")
port_daily = pd.read_csv("Port Area Daily Data.csv")
science_daily = pd.read_csv("Science Garden Daily Data.csv")
# Skip the "Source" row in pressure data (row 1)
pressure_daily = pd.read_csv("Mean Sea Level Pressure Daily Data.csv", skiprows=[1])

# Create datetime column for each daily dataset
for df in [naia_daily, port_daily, science_daily]:
    df['datetime'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])

# Process pressure data
pressure_daily['datetime'] = pd.to_datetime(pressure_daily['Date(UTC)'])
pressure_daily = pressure_daily.drop('Date(UTC)', axis=1)

print(f"NAIA Daily: {naia_daily.shape}")
print(f"Port Area Daily: {port_daily.shape}")
print(f"Science Garden Daily: {science_daily.shape}")
print(f"Pressure Daily: {pressure_daily.shape}\n")

# =============================================================================
# DATA CLEANING
# =============================================================================
print("Step 2: Data Cleaning...")

def clean_weather_data(df, station_name):
    """Clean individual station data"""
    # Replace missing values (-999.0) with NaN
    df = df.replace(-999.0, np.nan)
    
    # Replace trace rainfall (-1.0) with 0.05mm (small but non-zero)
    if 'RAINFALL' in df.columns:
        df['RAINFALL'] = df['RAINFALL'].replace(-1.0, 0.05)
        # Ensure non-negative rainfall
        df.loc[df['RAINFALL'] < 0, 'RAINFALL'] = 0
    
    # Clip humidity to valid range
    if 'RH' in df.columns:
        df['RH'] = np.clip(df['RH'], 0, 100)
    
    # Ensure non-negative wind speed
    if 'WIND_SPEED' in df.columns:
        df.loc[df['WIND_SPEED'] < 0, 'WIND_SPEED'] = 0
    
    print(f" - Cleaned {station_name} data")
    return df

naia_daily = clean_weather_data(naia_daily, "NAIA")
port_daily = clean_weather_data(port_daily, "Port Area")
science_daily = clean_weather_data(science_daily, "Science Garden")

# Clean pressure data
pressure_daily = pressure_daily.replace(-999.0, np.nan)

# =============================================================================
# MERGE AND AGGREGATE DATA FROM MULTIPLE STATIONS
# =============================================================================
print("\nStep 3: Merging data from multiple stations...")

# Rename columns to include station identifier
naia_daily = naia_daily.rename(columns={
    'RAINFALL': 'rainfall_naia', 'TMAX': 'tmax_naia', 'TMIN': 'tmin_naia',
    'RH': 'humidity_naia', 'WIND_SPEED': 'windspeed_naia', 
    'WIND_DIRECTION': 'winddir_naia'
})

port_daily = port_daily.rename(columns={
    'RAINFALL': 'rainfall_port', 'TMAX': 'tmax_port', 'TMIN': 'tmin_port',
    'RH': 'humidity_port', 'WIND_SPEED': 'windspeed_port',
    'WIND_DIRECTION': 'winddir_port'
})

science_daily = science_daily.rename(columns={
    'RAINFALL': 'rainfall_science', 'TMAX': 'tmax_science', 'TMIN': 'tmin_science',
    'RH': 'humidity_science', 'WIND_SPEED': 'windspeed_science',
    'WIND_DIRECTION': 'winddir_science'
})

# Merge all datasets on datetime
df = naia_daily[['datetime', 'rainfall_naia', 'tmax_naia', 'tmin_naia', 
                  'humidity_naia', 'windspeed_naia', 'winddir_naia']].copy()

df = df.merge(port_daily[['datetime', 'rainfall_port', 'tmax_port', 'tmin_port',
                           'humidity_port', 'windspeed_port', 'winddir_port']], 
              on='datetime', how='outer')

df = df.merge(science_daily[['datetime', 'rainfall_science', 'tmax_science', 'tmin_science',
                              'humidity_science', 'windspeed_science', 'winddir_science']], 
              on='datetime', how='outer')

df = df.merge(pressure_daily, on='datetime', how='outer')

# Rename pressure columns for simplicity
df = df.rename(columns={
    'NAIA Pasay City, M.Manila Press.QFF.Dly [hPa]': 'pressure_naia',
    'Port Area, Manila Press.QFF.Dly [hPa]': 'pressure_port',
    'Science Garden Quezon City, Metro Manila Press.QFF.Dly [hPa]': 'pressure_science'
})

# Sort by date
df = df.sort_values('datetime').reset_index(drop=True)

print(f"Merged dataset shape: {df.shape}")

# =============================================================================
# CREATE AGGREGATED FEATURES (Average across stations)
# =============================================================================
print("\nStep 4: Creating aggregated features...")

# Average temperature across stations
df['tempmax'] = df[['tmax_naia', 'tmax_port', 'tmax_science']].mean(axis=1)
df['tempmin'] = df[['tmin_naia', 'tmin_port', 'tmin_science']].mean(axis=1)

# Average humidity across stations
df['humidity'] = df[['humidity_naia', 'humidity_port', 'humidity_science']].mean(axis=1)

# Average wind speed across stations
df['windspeed'] = df[['windspeed_naia', 'windspeed_port', 'windspeed_science']].mean(axis=1)

# Average pressure across stations
df['sealevelpressure'] = df[['pressure_naia', 'pressure_port', 'pressure_science']].mean(axis=1)

# Total rainfall (use maximum across stations to capture localized rain)
df['precip'] = df[['rainfall_naia', 'rainfall_port', 'rainfall_science']].max(axis=1)

# Remove rows where all key features are missing
df = df.dropna(subset=['tempmax', 'tempmin', 'humidity', 'precip'], how='all')

print(f"Dataset after aggregation: {df.shape}")
print(f"Remaining missing values: {df[['tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure', 'precip']].isna().sum().sum()}")

# Forward fill small gaps (up to 2 days) for weather continuity
df = df.set_index('datetime')
df = df.ffill(limit=2)
df = df.reset_index()

# Drop remaining rows with missing values in key columns
df = df.dropna(subset=['tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure', 'precip'])

print(f"Final dataset after handling missing values: {df.shape}\n")

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
print("Step 5: Feature Engineering...")

# Binary rain indicator
df['rain'] = (df['precip'] > 0).astype(int)

# Temporal features
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
df['day_of_year'] = df['datetime'].dt.dayofyear

# Lagged features (previous day's weather)
for col in ['precip', 'humidity', 'sealevelpressure']:
    df[f'{col}_lag1'] = df[col].shift(1)

# Rolling features (3-day moving averages)
df['precip_roll3'] = df['precip'].rolling(window=3, min_periods=1).mean()
df['humidity_roll3'] = df['humidity'].rolling(window=3, min_periods=1).mean()

# Drop rows with NaN from lagged features (only the first row should have NaN)
df = df.dropna(subset=['precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1'])

print(f"Dataset with engineered features: {df.shape}")

# Check if we have enough data
if len(df) == 0:
    print("\nERROR: No data remaining after feature engineering!")
    print("This might be due to too many missing values in the source data.")
    exit(1)

# =============================================================================
# PREPARE FEATURES AND SPLIT DATA
# =============================================================================
features = [
    'tempmax', 'tempmin',
    'humidity', 'windspeed', 'sealevelpressure',
    'month', 'day_of_year',
    'precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1',
    'precip_roll3', 'humidity_roll3'
]

X = df[features]
y_class = df['rain']
y_reg = df['precip']

# Temporal split (last 20% for testing)
split_idx = int(len(df) * 0.8)
split_date = df.iloc[split_idx]['datetime']

train_mask = df.index < split_idx
test_mask = df.index >= split_idx

X_train, y_train_class, y_train_reg = X[train_mask], y_class[train_mask], y_reg[train_mask]
X_test, y_test_class, y_test_reg = X[test_mask], y_class[test_mask], y_reg[test_mask]

print(f"\nTrain-Test Split:")
print(f"Training period: {df.iloc[0]['datetime'].date()} to {df.iloc[split_idx-1]['datetime'].date()}")
print(f"Testing period: {split_date.date()} to {df.iloc[-1]['datetime'].date()}")
print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Rain days in train: {y_train_class.sum()} ({y_train_class.mean()*100:.1f}%)")
print(f"Rain days in test: {y_test_class.sum()} ({y_test_class.mean()*100:.1f}%)\n")

# =============================================================================
# ADVANCED HYPERPARAMETER TUNING - Classification Model (2-Stage)
# =============================================================================
print("="*60)
print("ADVANCED HYPERPARAMETER TUNING - CLASSIFICATION MODEL")
print("="*60)

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# STAGE 1: Broad Random Search (40 iterations)
print("\n[STAGE 1] Broad random search with 40 iterations...")
param_grid_class_broad = {
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'n_estimators': [300, 500, 700, 1000],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 4, 5, 7],
    'gamma': [0, 0.05, 0.1, 0.15, 0.2, 0.3],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1],  # L1 regularization
    'reg_lambda': [0.5, 1, 1.5, 2, 3]      # L2 regularization
}

base_clf = XGBClassifier(
    scale_pos_weight=len(y_train_class[y_train_class==0]) / len(y_train_class[y_train_class==1]),
    eval_metric='logloss',
    random_state=42,
    tree_method='hist'  # Faster training
)

random_search_clf = RandomizedSearchCV(
    base_clf,
    param_distributions=param_grid_class_broad,
    n_iter=40,
    scoring='f1',
    cv=5,  # 5-fold CV for better evaluation
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_clf.fit(X_train, y_train_class)
best_params_stage1 = random_search_clf.best_params_

print(f"\nStage 1 Best Parameters: {best_params_stage1}")
print(f"Stage 1 Best CV F1 Score: {random_search_clf.best_score_:.3f}")

# STAGE 2: Fine-tune around best parameters
print("\n[STAGE 2] Fine-tuning around best parameters with grid search...")
param_grid_class_fine = {
    'learning_rate': [max(0.005, best_params_stage1['learning_rate']*0.5), 
                      best_params_stage1['learning_rate'],
                      min(0.2, best_params_stage1['learning_rate']*1.5)],
    'max_depth': [max(3, best_params_stage1['max_depth']-1),
                  best_params_stage1['max_depth'],
                  min(12, best_params_stage1['max_depth']+1)],
    'n_estimators': [best_params_stage1['n_estimators'], 
                     best_params_stage1['n_estimators'] + 200],
    'colsample_bytree': [max(0.5, best_params_stage1['colsample_bytree']-0.1),
                         best_params_stage1['colsample_bytree'],
                         min(1.0, best_params_stage1['colsample_bytree']+0.1)],
    'subsample': [max(0.5, best_params_stage1['subsample']-0.1),
                  best_params_stage1['subsample'],
                  min(1.0, best_params_stage1['subsample']+0.1)]
}

# Keep other best parameters fixed
base_clf_fine = XGBClassifier(
    scale_pos_weight=len(y_train_class[y_train_class==0]) / len(y_train_class[y_train_class==1]),
    eval_metric='logloss',
    random_state=42,
    tree_method='hist',
    min_child_weight=best_params_stage1['min_child_weight'],
    gamma=best_params_stage1['gamma'],
    reg_alpha=best_params_stage1['reg_alpha'],
    reg_lambda=best_params_stage1['reg_lambda']
)

grid_search_clf = GridSearchCV(
    base_clf_fine,
    param_grid=param_grid_class_fine,
    scoring='f1',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_clf.fit(X_train, y_train_class)

print(f"\nFinal Best Classification Parameters: {grid_search_clf.best_params_}")
print(f"Final Best CV F1 Score: {grid_search_clf.best_score_:.3f}")

# Use the best model
clf = grid_search_clf.best_estimator_
y_pred_class = clf.predict(X_test)

print("\n" + "="*60)
print("CLASSIFICATION RESULTS - Rain Occurrence (Yes/No)")
print("="*60)
print(classification_report(y_test_class, y_pred_class, target_names=["No Rain", "Rain"]))
print(f"F1 Score: {f1_score(y_test_class, y_pred_class):.3f}")

# =============================================================================
# REGRESSION MODEL - Predict Rainfall Amount (Only on Rainy Days)
# =============================================================================
print("\n" + "="*60)
print("PREPARING REGRESSION DATA")
print("="*60)

# Filter to only rainy days for regression
rainy_train = y_train_reg > 0
rainy_test = y_test_reg > 0

X_train_rainy = X_train[rainy_train]
y_train_rainy = y_train_reg[rainy_train]
X_test_rainy = X_test[rainy_test]
y_test_rainy = y_test_reg[rainy_test]

print(f"Train rainy samples: {len(X_train_rainy)}, Test rainy samples: {len(X_test_rainy)}")

# =============================================================================
# ADVANCED HYPERPARAMETER TUNING - Regression Model (2-Stage)
# =============================================================================
print("\n" + "="*60)
print("ADVANCED HYPERPARAMETER TUNING - REGRESSION MODEL")
print("="*60)

# STAGE 1: Broad Random Search (40 iterations)
print("\n[STAGE 1] Broad random search with 40 iterations...")
param_grid_reg_broad = {
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1, 0.15],
    'max_depth': [2, 3, 4, 5, 6, 7, 8],
    'n_estimators': [300, 500, 700, 1000, 1200],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 4, 5, 7, 10],
    'gamma': [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
    'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5, 1, 2],  # L1 regularization
    'reg_lambda': [0.5, 1, 1.5, 2, 3, 5]           # L2 regularization
}

base_reg = XGBRegressor(
    random_state=42,
    tree_method='hist'
)

random_search_reg = RandomizedSearchCV(
    base_reg,
    param_distributions=param_grid_reg_broad,
    n_iter=40,
    scoring='neg_mean_absolute_error',
    cv=5,  # 5-fold CV
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search_reg.fit(X_train_rainy, y_train_rainy)
best_params_reg_stage1 = random_search_reg.best_params_

print(f"\nStage 1 Best Parameters: {best_params_reg_stage1}")
print(f"Stage 1 Best CV MAE: {-random_search_reg.best_score_:.3f} mm")

# STAGE 2: Fine-tune around best parameters
print("\n[STAGE 2] Fine-tuning around best parameters with grid search...")
param_grid_reg_fine = {
    'learning_rate': [max(0.005, best_params_reg_stage1['learning_rate']*0.5),
                      best_params_reg_stage1['learning_rate'],
                      min(0.2, best_params_reg_stage1['learning_rate']*1.5)],
    'max_depth': [max(2, best_params_reg_stage1['max_depth']-1),
                  best_params_reg_stage1['max_depth'],
                  min(10, best_params_reg_stage1['max_depth']+1)],
    'n_estimators': [best_params_reg_stage1['n_estimators'],
                     best_params_reg_stage1['n_estimators'] + 200,
                     best_params_reg_stage1['n_estimators'] + 400],
    'colsample_bytree': [max(0.5, best_params_reg_stage1['colsample_bytree']-0.1),
                         best_params_reg_stage1['colsample_bytree'],
                         min(1.0, best_params_reg_stage1['colsample_bytree']+0.1)],
    'subsample': [max(0.5, best_params_reg_stage1['subsample']-0.1),
                  best_params_reg_stage1['subsample'],
                  min(1.0, best_params_reg_stage1['subsample']+0.1)]
}

base_reg_fine = XGBRegressor(
    random_state=42,
    tree_method='hist',
    min_child_weight=best_params_reg_stage1['min_child_weight'],
    gamma=best_params_reg_stage1['gamma'],
    reg_alpha=best_params_reg_stage1['reg_alpha'],
    reg_lambda=best_params_reg_stage1['reg_lambda']
)

grid_search_reg = GridSearchCV(
    base_reg_fine,
    param_grid=param_grid_reg_fine,
    scoring='neg_mean_absolute_error',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search_reg.fit(X_train_rainy, y_train_rainy)

print(f"\nFinal Best Regression Parameters: {grid_search_reg.best_params_}")
print(f"Final Best CV MAE: {-grid_search_reg.best_score_:.3f} mm")

# Use the best model
reg = grid_search_reg.best_estimator_
y_pred_rainy = reg.predict(X_test_rainy)

# Ensure non-negative predictions
y_pred_rainy = np.maximum(y_pred_rainy, 0)

print("\n" + "="*60)
print("REGRESSION RESULTS - Rainfall Amount (on rainy days)")
print("="*60)
mae = mean_absolute_error(y_test_rainy, y_pred_rainy)
rmse = np.sqrt(mean_squared_error(y_test_rainy, y_pred_rainy))
r2 = r2_score(y_test_rainy, y_pred_rainy)

print(f"MAE:  {mae:.3f} mm")
print(f"RMSE: {rmse:.3f} mm")
print(f"R²:   {r2:.3f}")

# Calculate MAPE only for values > 0.1 to avoid division issues
valid_mape_mask = y_test_rainy > 0.1
if valid_mape_mask.sum() > 0:
    mape = np.mean(np.abs((y_test_rainy[valid_mape_mask] - y_pred_rainy[valid_mape_mask]) / 
                           y_test_rainy[valid_mape_mask])) * 100
    print(f"MAPE: {mape:.2f}% (calculated on {valid_mape_mask.sum()} samples with precip > 0.1mm)")

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
axes[1, 0].set_xlabel("Actual Rainfall (mm)")
axes[1, 0].set_ylabel("Predicted Rainfall (mm)")
axes[1, 0].set_title(f"Predicted vs Actual Rainfall\n(R² = {r2:.3f})")
axes[1, 0].grid(alpha=0.3)

# 4. Residuals distribution
residuals = y_test_rainy - y_pred_rainy
axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel("Residual (Actual - Predicted)")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].set_title(f"Residuals Distribution\n(MAE = {mae:.3f} mm)")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("KEY ADAPTATIONS FOR MANILA DATA:")
print("="*60)
print("1. Loaded and merged data from 3 weather stations (NAIA, Port Area, Science Garden)")
print("2. Handled missing values (-999.0) and trace rainfall (-1.0)")
print("3. Created aggregated features across multiple stations")
print("4. Used maximum rainfall across stations to capture localized rain")
print("5. Changed units from inches to millimeters (mm)")
print("6. Removed solar radiation (not available in Manila datasets)")
print("7. Forward-filled small gaps (up to 2 days) for continuity")
print("8. Used temporal split (80% train, 20% test)")
print("9. Maintained lagged and rolling features for temporal patterns")
print("10. Handled Manila's tropical climate characteristics")
print("\n" + "="*60)
print("HYPERPARAMETER TUNING SUMMARY:")
print("="*60)
print("✓ 2-STAGE TUNING PROCESS:")
print("  - Stage 1: Broad RandomizedSearchCV (40 iterations, 5-fold CV)")
print("  - Stage 2: Fine GridSearchCV around best parameters")
print("\n✓ CLASSIFICATION MODEL:")
print("  - Tested: learning_rate, max_depth, n_estimators, subsample, colsample")
print("  - Added: L1 (reg_alpha) and L2 (reg_lambda) regularization")
print("  - Total combinations tested: ~40 (stage 1) + ~81 (stage 2)")
print("\n✓ REGRESSION MODEL:")
print("  - Tested: Same parameters + extended ranges")
print("  - More aggressive regularization options")
print("  - Total combinations tested: ~40 (stage 1) + ~243 (stage 2)")
print("\n✓ OPTIMIZATIONS:")
print("  - Used 'hist' tree method for faster training")
print("  - 5-fold cross-validation for robust evaluation")
print("  - All CPU cores utilized (n_jobs=-1)")
print("\nBest parameters automatically applied to final models!")