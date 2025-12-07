# ================================================================
# MENU-BASED SIMULATIONS FOR DAILY RAINFALL PREDICTION (SOP 1–3)
# ================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
import os

# ------------------------------------------------------------
# LOAD AND PREPARE DATA - PHILIPPINE WEATHER DATASETS
# ------------------------------------------------------------
print("Loading and preparing dataset...")

def load_and_preprocess_hourly_data():
    """
    Load and preprocess 6-hourly data file
    """
    print("Loading 6-Hourly Data.csv...")
    
    # Read the file with proper handling of the "Source" row
    df_hourly = pd.read_csv("6-Hourly Data.csv")
    
    # Remove the "Source" row (second row) which contains metadata
    if df_hourly.iloc[0, 0] == "Source":
        # If first row is "Source", drop it
        df_hourly = df_hourly.iloc[1:].reset_index(drop=True)
    elif df_hourly.iloc[1, 0] == "Source":
        # If second row is "Source", drop it
        df_hourly = df_hourly.drop(0).reset_index(drop=True)
    
    # Convert Date column to datetime with explicit format
    # First, let's check what the date format looks like
    print(f"Sample dates: {df_hourly.iloc[0, 0]}, {df_hourly.iloc[1, 0]}")
    
    # Try different date formats
    date_formats = [
        '%Y-%m-%d %H:%M:%S', 
        '%Y/%m/%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%d',
        '%d-%m-%Y',
        '%m-%d-%Y'
    ]
    
    for fmt in date_formats:
        try:
            df_hourly["Date"] = pd.to_datetime(df_hourly.iloc[:, 0], format=fmt)
            print(f"Successfully parsed dates with format: {fmt}")
            break
        except:
            continue
    else:
        # If no format works, use default parsing
        df_hourly["Date"] = pd.to_datetime(df_hourly.iloc[:, 0], errors='coerce')
    
    # Drop rows where date parsing failed
    initial_len = len(df_hourly)
    df_hourly = df_hourly.dropna(subset=["Date"])
    print(f"Dropped {initial_len - len(df_hourly)} rows with invalid dates")
    
    # Extract just the date part for daily aggregation
    df_hourly["Date_Day"] = df_hourly["Date"].dt.normalize()
    
    # Convert all numeric columns (except date columns) to float, handling missing values
    numeric_cols = df_hourly.columns[1:]  # Skip the first column (original date)
    
    for col in numeric_cols:
        if col not in ["Date", "Date_Day"]:
            # Replace -999.0 with NaN
            df_hourly[col] = pd.to_numeric(df_hourly[col], errors='coerce')
            df_hourly.loc[df_hourly[col] == -999.0, col] = np.nan
    
    # Convert trace rainfall values (-1.0) to 0
    for col in df_hourly.columns:
        if 'Prec.Period.Amount' in str(col):
            df_hourly.loc[df_hourly[col] == -1.0, col] = 0
    
    print(f"6-hourly data loaded: {len(df_hourly)} rows")
    return df_hourly

def load_and_process_daily_data(station_name):
    """
    Load daily data for a specific station
    """
    filename_map = {
        'NAIA': 'NAIA Daily Data.csv',
        'Port Area': 'Port Area Daily Data.csv', 
        'Science Garden': 'Science Garden Daily Data.csv'
    }
    
    if station_name not in filename_map:
        raise ValueError(f"Unknown station: {station_name}")
    
    filename = filename_map[station_name]
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
    
    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(
        df[['YEAR', 'MONTH', 'DAY']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d'
    )
    
    # Handle missing values (-999.0)
    for col in ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED', 'WIND_DIRECTION']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] == -999.0, col] = np.nan
    
    # Convert trace rainfall values (-1.0) to 0
    if 'RAINFALL' in df.columns:
        df.loc[df['RAINFALL'] == -1.0, 'RAINFALL'] = 0
    
    # Rename columns for consistency
    column_rename = {
        'RAINFALL': 'precip',
        'TMAX': 'tempmax',
        'TMIN': 'tempmin',
        'RH': 'humidity',
        'WIND_SPEED': 'windspeed',
        'WIND_DIRECTION': 'winddir'
    }
    
    for old_col, new_col in column_rename.items():
        if old_col in df.columns:
            df = df.rename(columns={old_col: new_col})
    
    # Add station identifier
    df['station'] = station_name
    
    return df

def create_combined_dataset():
    """
    Create combined dataset from all available sources
    """
    print("\nCreating combined dataset...")
    
    # Load daily data for all stations
    stations = ['NAIA', 'Port Area', 'Science Garden']
    daily_dfs = []
    
    for station in stations:
        df_station = load_and_process_daily_data(station)
        if df_station is not None:
            daily_dfs.append(df_station)
    
    if not daily_dfs:
        print("ERROR: No daily data files found!")
        return None
    
    # Combine daily data
    combined_df = pd.concat(daily_dfs, ignore_index=True)
    print(f"Combined daily data: {len(combined_df)} rows")
    
    # Load and merge mean sea level pressure data
    if os.path.exists('Mean Sea Level Pressure Daily Data.csv'):
        print("Loading mean sea level pressure data...")
        pressure_df = pd.read_csv('Mean Sea Level Pressure Daily Data.csv')
        
        # Remove "Source" row if present
        if pressure_df.iloc[0, 0] == "Source":
            pressure_df = pressure_df.iloc[1:].reset_index(drop=True)
        
        # Parse date column
        date_col = pressure_df.columns[0]
        pressure_df['datetime'] = pd.to_datetime(pressure_df[date_col], errors='coerce')
        pressure_df = pressure_df.dropna(subset=['datetime'])
        
        # Handle missing values
        for col in pressure_df.columns:
            if col != 'datetime' and col != date_col:
                pressure_df[col] = pd.to_numeric(pressure_df[col], errors='coerce')
                pressure_df.loc[pressure_df[col] == -999.0, col] = np.nan
        
        # Rename columns for easier merging
        pressure_columns = {}
        for col in pressure_df.columns:
            if 'NAIA' in col:
                pressure_columns[col] = 'NAIA_sealevelpressure'
            elif 'Port Area' in col:
                pressure_columns[col] = 'Port Area_sealevelpressure'
            elif 'Science Garden' in col:
                pressure_columns[col] = 'Science Garden_sealevelpressure'
        
        pressure_df = pressure_df.rename(columns=pressure_columns)
        
        # Merge pressure data with combined data
        for station in stations:
            pressure_col = f'{station}_sealevelpressure'
            if pressure_col in pressure_df.columns:
                # Create a mapping of date to pressure for this station
                pressure_map = pressure_df.set_index('datetime')[pressure_col].to_dict()
                
                # Add pressure to the combined dataframe
                combined_df[f'{station}_pressure'] = combined_df['datetime'].map(pressure_map)
                
                # For rows of this station, use their pressure value
                station_mask = combined_df['station'] == station
                combined_df.loc[station_mask, 'sealevelpressure'] = combined_df.loc[station_mask, f'{station}_pressure']
        
        # Drop temporary columns
        for station in stations:
            if f'{station}_pressure' in combined_df.columns:
                combined_df = combined_df.drop(columns=[f'{station}_pressure'])
    
    # Create binary rain target
    combined_df['rain'] = (combined_df['precip'] > 0).astype(int)
    
    # Add temporal features
    combined_df['month'] = combined_df['datetime'].dt.month
    combined_df['day_of_year'] = combined_df['datetime'].dt.dayofyear
    combined_df['year'] = combined_df['datetime'].dt.year
    combined_df['day_of_week'] = combined_df['datetime'].dt.dayofweek
    
    # Add station-specific features (one-hot encoding)
    combined_df = pd.get_dummies(combined_df, columns=['station'], prefix='station')
    
    # Sort by datetime
    combined_df = combined_df.sort_values('datetime')
    
    print(f"Final combined dataset: {len(combined_df)} rows")
    print(f"Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
    
    return combined_df

# Load the data
try:
    df = create_combined_dataset()
    
    if df is None:
        print("ERROR: Could not create dataset. Please check your data files.")
        print("\nRequired files:")
        print("1. NAIA Daily Data.csv (or Port Area Daily Data.csv, or Science Garden Daily Data.csv)")
        print("2. Mean Sea Level Pressure Daily Data.csv (optional)")
        print("3. 6-Hourly Data.csv (optional)")
        exit()
    
    # Define features based on available columns
    base_features = ["tempmax", "tempmin", "humidity", "windspeed", "month", "day_of_year"]
    
    # Add pressure if available
    if 'sealevelpressure' in df.columns:
        base_features.append("sealevelpressure")
    
    # Add station features
    station_features = [col for col in df.columns if col.startswith('station_')]
    
    # Combine all features
    features = base_features + station_features
    
    # Check which features are actually available
    available_features = [f for f in features if f in df.columns]
    print(f"\nAvailable features: {available_features}")
    
    # Remove rows with missing values in features or target
    required_cols = available_features + ['rain', 'precip']
    df_clean = df[required_cols].dropna()
    
    print(f"After cleaning: {len(df_clean)} rows")
    
    # Check if we have enough data
    if len(df_clean) < 100:
        print(f"WARNING: Very little data available ({len(df_clean)} rows)")
        print("Consider using fewer features or checking data quality.")
    
    # Split data - use most recent 20% as test
    train_size = int(0.8 * len(df_clean))
    train_mask = np.arange(len(df_clean)) < train_size
    test_mask = np.arange(len(df_clean)) >= train_size
    
    X_train = df_clean.loc[train_mask, available_features].copy()
    X_test = df_clean.loc[test_mask, available_features].copy()
    y_train_class = df_clean.loc[train_mask, "rain"]
    y_test_class = df_clean.loc[test_mask, "rain"]
    y_train_reg = df_clean.loc[train_mask, "precip"]
    y_test_reg = df_clean.loc[test_mask, "precip"]
    
    print(f"\nData split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Rain days in training: {y_train_class.sum()} ({y_train_class.mean()*100:.1f}%)")
    print(f"Rain days in test: {y_test_class.sum()} ({y_test_class.mean()*100:.1f}%)")
    
except Exception as e:
    print(f"\nERROR during data processing: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Try a simpler approach with just NAIA data
    print("\nTrying simpler approach with NAIA data only...")
    try:
        df = pd.read_csv("NAIA Daily Data.csv")
        print(f"NAIA data columns: {df.columns.tolist()}")
        
        # Basic processing
        df['datetime'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']].astype(str).agg('-'.join, axis=1))
        
        # Handle missing values
        for col in ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND_SPEED']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df.loc[df[col] == -999.0, col] = np.nan
        
        # Convert trace rainfall
        if 'RAINFALL' in df.columns:
            df.loc[df['RAINFALL'] == -1.0, 'RAINFALL'] = 0
        
        # Rename
        rename_map = {
            'RAINFALL': 'precip',
            'TMAX': 'tempmax',
            'TMIN': 'tempmin',
            'RH': 'humidity',
            'WIND_SPEED': 'windspeed'
        }
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        df = df.dropna()
        df["rain"] = (df["precip"] > 0).astype(int)
        df["month"] = df["datetime"].dt.month
        df["day_of_year"] = df["datetime"].dt.dayofyear
        
        features = ["tempmax", "tempmin", "humidity", "windspeed", "month", "day_of_year"]
        features = [f for f in features if f in df.columns]
        
        train_size = int(0.8 * len(df))
        train_mask = np.arange(len(df)) < train_size
        test_mask = np.arange(len(df)) >= train_size
        
        X_train = df.loc[train_mask, features].copy()
        X_test = df.loc[test_mask, features].copy()
        y_train_class = df.loc[train_mask, "rain"]
        y_test_class = df.loc[test_mask, "rain"]
        y_train_reg = df.loc[train_mask, "precip"]
        y_test_reg = df.loc[test_mask, "precip"]
        
        print(f"Loaded NAIA data: {len(df)} rows")
        
    except Exception as e2:
        print(f"Could not load data: {e2}")
        print("\nPlease ensure at least one of these files exists:")
        print("1. NAIA Daily Data.csv")
        print("2. Port Area Daily Data.csv")
        print("3. Science Garden Daily Data.csv")
        exit()

# ================================================================
# DEFINE SIMULATION FUNCTIONS (UNCHANGED)
# ================================================================
def sop1_precision_recall_curve(X_train, y_train, X_test, y_test):
    """SOP 1 — Noisy Data Sensitivity → Precision–Recall Curve"""
    print("\nRunning SOP 1: Precision–Recall Curve Simulation...")
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
    
    # Inject noise (simulate satellite data distortion)
    X_train_noisy = X_train.copy()
    noise_idx = np.random.choice(X_train_noisy.index, 
                                size=int(0.1 * len(X_train_noisy)), 
                                replace=False)
    X_train_noisy.loc[noise_idx, "humidity"] *= np.random.uniform(1.5, 3.0, len(noise_idx))

    model_clean = XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=300, random_state=42)
    model_noisy = XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=300, random_state=42)

    model_clean.fit(X_train, y_train)
    model_noisy.fit(X_train_noisy, y_train)

    y_scores_clean = model_clean.predict_proba(X_test)[:, 1]
    y_scores_noisy = model_noisy.predict_proba(X_test)[:, 1]

    precision_clean, recall_clean, _ = precision_recall_curve(y_test, y_scores_clean)
    precision_noisy, recall_noisy, _ = precision_recall_curve(y_test, y_scores_noisy)
    ap_clean = average_precision_score(y_test, y_scores_clean)
    ap_noisy = average_precision_score(y_test, y_scores_noisy)

    plt.figure(figsize=(8,6))
    plt.plot(recall_clean, precision_clean, label=f"Clean Data (AP={ap_clean:.3f})")
    plt.plot(recall_noisy, precision_noisy, linestyle='--', label=f"Noisy Data (AP={ap_noisy:.3f})")
    plt.title("SOP 1: Precision–Recall Curve under Noisy Satellite Data")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def sop2_learning_curve(X_train, y_train, X_test, y_test):
    """SOP 2 — Learning Curve: Effect of Learning Rate"""
    print("\nRunning SOP 2: Learning Curve Simulation...")

    lr_low = 0.05
    lr_high = 0.3

    model_low = XGBRegressor(
        learning_rate=lr_low,
        n_estimators=200,
        max_depth=5,
        random_state=42,
        eval_metric="rmse"
    )
    model_high = XGBRegressor(
        learning_rate=lr_high,
        n_estimators=200,
        max_depth=5,
        random_state=42,
        eval_metric="rmse"
    )

    model_low.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model_high.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    rmse_low = model_low.evals_result()["validation_0"]["rmse"]
    rmse_high = model_high.evals_result()["validation_0"]["rmse"]

    plt.figure(figsize=(8,6))
    plt.plot(rmse_low, label=f"Low Learning Rate ({lr_low})", linewidth=2)
    plt.plot(rmse_high, linestyle="--", label=f"High Learning Rate ({lr_high})", linewidth=2)
    plt.title("SOP 2: Learning Curve - Effect of Learning Rate on Convergence")
    plt.xlabel("Boosting Rounds")
    plt.ylabel("Validation RMSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def sop3_feature_importance_heatmap(model, feature_names):
    """SOP 3: Feature Importance Table (XGBoost Gain Heatmap)"""
    importance_dict = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': [importance_dict.get(f, 0) for f in feature_names]
    }).sort_values(by='Importance', ascending=False)

    importance_df['Normalized'] = importance_df['Importance'] / importance_df['Importance'].sum()

    plt.figure(figsize=(10, 6))
    heatmap_data = importance_df.set_index('Feature').T
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={'label': 'Normalized Importance'},
        linewidths=0.5
    )

    plt.title("SOP 3: Feature Importance Table (XGBoost Gain)", fontsize=14)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ================================================================
# MENU SYSTEM
# ================================================================
def main_menu(reg, X_train):
    while True:
        print("\n" + "="*50)
        print("PHILIPPINE WEATHER DATA SIMULATIONS")
        print("="*50)
        print("Select an option:")
        print("1. SOP 1: Precision-Recall Curve")
        print("2. SOP 2: Learning Curve")
        print("3. SOP 3: Feature Importance Heatmap")
        print("4. Show Data Summary")
        print("5. Exit")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            if len(X_train) == 0 or len(X_test) == 0:
                print("ERROR: Not enough data to run simulation")
            else:
                sop1_precision_recall_curve(X_train, y_train_class, X_test, y_test_class)
        elif choice == "2":
            if len(X_train) == 0 or len(X_test) == 0:
                print("ERROR: Not enough data to run simulation")
            else:
                sop2_learning_curve(X_train, y_train_reg, X_test, y_test_reg)
        elif choice == "3":
            if len(X_train) == 0:
                print("ERROR: No training data available")
            else:
                print("\nRunning SOP 3: Feature Importance Simulation (Heatmap Style)...\n")
                sop3_feature_importance_heatmap(reg, X_train.columns.tolist())
        elif choice == "4":
            print("\n" + "="*50)
            print("DATA SUMMARY")
            print("="*50)
            print(f"Total samples: {len(X_train) + len(X_test)}")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            if len(y_train_class) > 0:
                print(f"Rain days in training: {y_train_class.sum()} ({y_train_class.mean()*100:.1f}%)")
            if len(y_test_class) > 0:
                print(f"Rain days in test: {y_test_class.sum()} ({y_test_class.mean()*100:.1f}%)")
            print(f"\nFeatures used: {list(X_train.columns)}")
            if len(X_train) > 0:
                print("\nTraining set statistics:")
                print(X_train.describe())
        elif choice == "5":
            print("Exiting simulation.")
            break
        else:
            print("Invalid choice. Try again.")


# ------------------------------------------------------------
# TRAIN MODELS BEFORE MENU
# ------------------------------------------------------------
print("\nTraining baseline models...")

try:
    # Classification model
    clf = XGBClassifier(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=300,
        random_state=42,
        eval_metric="logloss"
    )
    clf.fit(X_train, y_train_class)

    # Regression model
    reg = XGBRegressor(
        learning_rate=0.05,
        max_depth=5,
        n_estimators=300,
        random_state=42,
        eval_metric="rmse"
    )
    reg.fit(X_train, y_train_reg)

    print("Models trained successfully!")

except Exception as e:
    print(f"Error training models: {e}")
    print("Continuing with menu, but some options may not work...")
    # Create dummy models for the menu
    reg = None

# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main_menu(reg, X_train)