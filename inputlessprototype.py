import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score,
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)
from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
import warnings
import json
import time

# --- CONFIGURATION ---
try:
    warnings.filterwarnings('ignore', category=pd.SettingWithCopyWarning)
except AttributeError:
    pass

app = Flask(__name__)
CORS(app) 

CLASSIFIER_MODEL_PATH = 'xgb_classifier_rain_occurrence.json'
REGRESSOR_MODEL_PATH = 'xgb_regressor_rain_amount.json'

clf_booster = None 
reg_booster = None 
GLOBAL_METRICS = {}

# CRITICAL: 13-feature list, synchronized across all files
FEATURES = [
    'tempmax', 'tempmin',
    'humidity', 'windspeed', 'sealevelpressure',
    'month', 'day_of_year',
    'precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1',
    'precip_roll3', 'humidity_roll3',
    'windspeed_roll3'
]

# =============================================================================
# CORE UTILITIES
# =============================================================================

def load_models():
    """Load the trained XGBoost models from disk."""
    global clf_booster, reg_booster
    try:
        clf_booster = xgb.Booster()
        clf_booster.load_model(CLASSIFIER_MODEL_PATH)
        reg_booster = xgb.Booster()
        reg_booster.load_model(REGRESSOR_MODEL_PATH)
        print("Models loaded successfully.")
        return True
    except xgb.core.XGBoostError as e:
        print(f"Error loading XGBoost model files: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return False

def clean_weather_data(df_):
    """Helper function to clean individual station data."""
    df_ = df_.replace(-999.0, np.nan)
    if 'RAINFALL' in df_.columns:
        df_['RAINFALL'] = df_['RAINFALL'].replace(-1.0, 0.05)
        df_.loc[df_['RAINFALL'] < 0, 'RAINFALL'] = 0
    if 'RH' in df_.columns:
        df_['RH'] = np.clip(df_['RH'], 0, 100)
    if 'WIND_SPEED' in df_.columns:
        df_.loc[df_['WIND_SPEED'] < 0, 'WIND_SPEED'] = 0
    return df_

def iqr_capping(df, column):
    """Cap outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df.copy()

def load_and_preprocess_data():
    """Loads, merges, cleans, imputes, and engineers features."""
    DATASET_FOLDER = 'dataset/'
    try:
        naia_daily = pd.read_csv(DATASET_FOLDER + "NAIA Daily Data.csv")
        port_daily = pd.read_csv(DATASET_FOLDER + "Port Area Daily Data.csv")
        science_daily = pd.read_csv(DATASET_FOLDER + "Science Garden Daily Data.csv")
        pressure_daily = pd.read_csv(DATASET_FOLDER + "Mean Sea Level Pressure Daily Data.csv", skiprows=[1])
    except FileNotFoundError as e:
        raise RuntimeError(f"Data file not found: {e}. Cannot proceed without data.")
    
    # Date processing
    for df_ in [naia_daily, port_daily, science_daily]:
        df_['datetime'] = pd.to_datetime(df_[['YEAR', 'MONTH', 'DAY']])
    
    pressure_daily['datetime'] = pd.to_datetime(pressure_daily['Date(UTC)'])
    pressure_daily = pressure_daily.drop('Date(UTC)', axis=1)
    
    # Clean and rename
    naia_daily = clean_weather_data(naia_daily).rename(columns={
        'RAINFALL': 'rainfall_naia', 'TMAX': 'tmax_naia', 'TMIN': 'tmin_naia', 
        'RH': 'humidity_naia', 'WIND_SPEED': 'windspeed_naia'
    })
    port_daily = clean_weather_data(port_daily).rename(columns={
        'RAINFALL': 'rainfall_port', 'TMAX': 'tmax_port', 'TMIN': 'tmin_port', 
        'RH': 'humidity_port', 'WIND_SPEED': 'windspeed_port'
    })
    science_daily = clean_weather_data(science_daily).rename(columns={
        'RAINFALL': 'rainfall_science', 'TMAX': 'tmax_science', 'TMIN': 'tmin_science', 
        'RH': 'humidity_science', 'WIND_SPEED': 'windspeed_science'
    })
    pressure_daily = pressure_daily.replace(-999.0, np.nan).rename(columns={
        'NAIA Pasay City, M.Manila Press.QFF.Dly [hPa]': 'pressure_naia', 
        'Port Area, Manila Press.QFF.Dly [hPa]': 'pressure_port', 
        'Science Garden Quezon City, Metro Manila Press.QFF.Dly [hPa]': 'pressure_science'
    })
    
    # Merge
    df = naia_daily[['datetime', 'rainfall_naia', 'tmax_naia', 'tmin_naia', 'humidity_naia', 'windspeed_naia']].copy()
    df = df.merge(port_daily[['datetime', 'rainfall_port', 'tmax_port', 'tmin_port', 'humidity_port', 'windspeed_port']], 
                  on='datetime', how='outer')
    df = df.merge(science_daily[['datetime', 'rainfall_science', 'tmax_science', 'tmin_science', 'humidity_science', 'windspeed_science']], 
                  on='datetime', how='outer')
    df = df.merge(pressure_daily[['datetime', 'pressure_naia', 'pressure_port', 'pressure_science']], 
                  on='datetime', how='outer')
    df = df.sort_values('datetime').reset_index(drop=True)
    
    # Aggregate features
    df['tempmax'] = df[['tmax_naia', 'tmax_port', 'tmax_science']].mean(axis=1)
    df['tempmin'] = df[['tmin_naia', 'tmin_port', 'tmin_science']].mean(axis=1)
    df['humidity'] = df[['humidity_naia', 'humidity_port', 'humidity_science']].mean(axis=1)
    df['windspeed'] = df[['windspeed_naia', 'windspeed_port', 'windspeed_science']].mean(axis=1)
    df['sealevelpressure'] = df[['pressure_naia', 'pressure_port', 'pressure_science']].mean(axis=1)
    df['precip'] = df[['rainfall_naia', 'rainfall_port', 'rainfall_science']].max(axis=1)
    
    # KNN Imputation
    impute_cols = ['tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure', 'precip']
    imputer = KNNImputer(n_neighbors=5)
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

    # IQR Outlier Capping
    df = iqr_capping(df, 'precip')

    # Feature Engineering
    df['rain'] = (df['precip'] > 0).astype(int)
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Lagged features
    for col in ['precip', 'humidity', 'sealevelpressure']:
        df[f'{col}_lag1'] = df[col].shift(1) 
        
    # Rolling features
    df['precip_roll3'] = df['precip'].rolling(window=3, min_periods=1).mean()
    df['humidity_roll3'] = df['humidity'].rolling(window=3, min_periods=1).mean()
    df['windspeed_roll3'] = df['windspeed'].rolling(window=3, min_periods=1).mean()

    df = df.dropna(subset=['precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1'])
    
    return df

def get_last_data_point():
    """Loads data and prepares the feature vector for the day after the last date in the dataset."""
    df = load_and_preprocess_data()
    
    last_row_features = df[FEATURES].iloc[-1].to_dict()
    
    last_datetime = df['datetime'].iloc[-1]
    prediction_datetime = last_datetime + pd.Timedelta(days=1)
    
    last_row_features['month'] = prediction_datetime.month
    last_row_features['day_of_year'] = prediction_datetime.dayofyear
    last_row_features['prediction_date'] = prediction_datetime.strftime('%Y-%m-%d')

    return last_row_features

def predict_combined_rainfall(data: dict):
    """Performs the two-stage prediction based on input features."""
    if clf_booster is None or reg_booster is None:
        raise RuntimeError("Models are not loaded on the server.")
        
    input_df = pd.DataFrame([data], columns=FEATURES)
    input_Dmatrix = xgb.DMatrix(input_df)
    
    # Stage 1: Classification
    rain_prob = clf_booster.predict(input_Dmatrix)[0]
    rain_occurrence = 1 if rain_prob > 0.5 else 0
    predicted_amount = 0.0

    # Stage 2: Regression
    if rain_occurrence == 1:
        predicted_amount = reg_booster.predict(input_Dmatrix)[0]
        predicted_amount = float(np.maximum(predicted_amount, 0))
    
    return {
        'rain_occurrence': rain_occurrence,
        'rain_probability': float(rain_prob), 
        'rain_amount': predicted_amount
    }

def calculate_adaptive_base_score(y_train_reg):
    """Calculates the mean rainfall for non-zero days to use as base_score."""
    wet_season_rainfall = y_train_reg[y_train_reg > 0]
    if len(wet_season_rainfall) > 0:
        return wet_season_rainfall.mean()
    return 0.5

def train_and_save_models():
    """Execute the full ML pipeline to train and save the models with full evaluation."""
    global GLOBAL_METRICS
    print("="*60)
    print("EXECUTING FULL ML PIPELINE: TRAINING AND SAVING MODELS (ENHANCED)")
    print("="*60)
    
    try:
        df = load_and_preprocess_data()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return False
    
    print(f"Dataset loaded: {len(df)} samples from {df['datetime'].min()} to {df['datetime'].max()}")
    
    X = df[FEATURES]
    y_class = df['rain']
    y_reg = df['precip']

    # Temporal split: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train_mask = df.index < split_idx

    X_train, y_train_class, y_train_reg = X[train_mask], y_class[train_mask], y_reg[train_mask]
    X_test, y_test_class, y_test_reg = X[~train_mask], y_class[~train_mask], y_reg[~train_mask]
    
    print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Train rain days: {y_train_class.sum()} ({y_train_class.mean()*100:.1f}%)")
    print(f"Test rain days: {y_test_class.sum()} ({y_test_class.mean()*100:.1f}%)")
    
    # Isolate rainy days
    rainy_train = y_train_reg > 0
    X_train_rainy = X_train[rainy_train]
    y_train_rainy = y_train_reg[rainy_train]
    
    # Feature Correlation Analysis
    print("\n--- Feature Correlation with Rainfall ---")
    print(X_train.corrwith(y_train_reg).sort_values(ascending=False))
    
    # ========== CLASSIFICATION MODEL ==========
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*60)
    
    train_start_time = time.time()
    
    param_grid_class = {
        'learning_rate': [0.005, 0.01, 0.05, 0.1], 
        'max_depth': [5, 7, 9],
        'n_estimators': [500, 1000], 
        'colsample_bytree': [0.7, 0.9],
        'subsample': [0.7, 0.9], 
        'min_child_weight': [1, 3], 
        'gamma': [0.1, 0.5, 1.0], 
        'reg_alpha': [0, 0.1], 
        'reg_lambda': [1, 2, 5] 
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scale_pos_weight = len(y_train_class[y_train_class==0]) / len(y_train_class[y_train_class==1])
    
    base_clf = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', 
        random_state=42, 
        tree_method='hist'
    )
    
    random_search_clf = RandomizedSearchCV(
        base_clf, param_distributions=param_grid_class, n_iter=20, 
        scoring='f1', cv=skf, verbose=0, random_state=42, n_jobs=-1
    )
    
    print("Running hyperparameter tuning for classifier...")
    random_search_clf.fit(X_train, y_train_class)
    clf_final = random_search_clf.best_estimator_
    
    # OBJECTIVE 1: Calculate F1 stability across CV folds
    cv_f1_scores = cross_val_score(clf_final, X_train, y_train_class, cv=skf, scoring='f1')
    f1_std_dev = float(np.std(cv_f1_scores))
    
    print(f"Best CV F1 Score: {random_search_clf.best_score_:.3f}")
    print(f"F1 Std Dev (Stability): ±{f1_std_dev:.4f}")
    print(f"Best parameters: {random_search_clf.best_params_}")
    
    # Feature Importance
    print("\n--- Classifier Feature Importance ---")
    feature_importance_df = pd.Series(clf_final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(feature_importance_df)
    
    # ========== CLASSIFICATION EVALUATION ==========
    print("\n" + "="*60)
    print("CLASSIFICATION EVALUATION (Test Set)")
    print("="*60)
    
    y_pred_class = clf_final.predict(X_test)
    
    acc = accuracy_score(y_test_class, y_pred_class)
    prec = precision_score(y_test_class, y_pred_class, zero_division=0)
    rec = recall_score(y_test_class, y_pred_class, zero_division=0)
    f1_test = f1_score(y_test_class, y_pred_class, zero_division=0)
    
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-Score:  {f1_test:.3f}")
    
    # Save classifier
    clf_final.save_model(CLASSIFIER_MODEL_PATH)
    print(f"\nClassifier saved to {CLASSIFIER_MODEL_PATH}")
    
    # ========== REGRESSION MODEL ==========
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODEL")
    print("="*60)
    
    param_grid_reg = {
        'learning_rate': [0.005, 0.01, 0.05, 0.1], 
        'max_depth': [4, 6, 8],
        'n_estimators': [500, 1000], 
        'colsample_bytree': [0.7, 0.9],
        'subsample': [0.7, 0.9], 
        'min_child_weight': [1, 3], 
        'gamma': [0.1, 0.5, 1.0], 
        'reg_alpha': [0, 0.1], 
        'reg_lambda': [1, 2, 5] 
    }
    
    regressor_base_score = calculate_adaptive_base_score(y_train_reg)
    print(f"Adaptive base score for regressor: {regressor_base_score:.3f}")
    
    base_reg = XGBRegressor(
        random_state=42, 
        tree_method='hist',
        objective='reg:squarederror',
        base_score=regressor_base_score
    )
    
    random_search_reg = RandomizedSearchCV(
        base_reg, param_distributions=param_grid_reg, n_iter=20, 
        scoring='neg_mean_absolute_error', cv=3, verbose=0, random_state=42, n_jobs=-1
    )
    
    print(f"Training regressor on {len(X_train_rainy)} rainy days...")
    random_search_reg.fit(X_train_rainy, y_train_rainy)
    reg_final = random_search_reg.best_estimator_
    
    # OBJECTIVE 1: Calculate R² stability across CV folds
    cv_r2_scores = cross_val_score(reg_final, X_train_rainy, y_train_rainy, cv=3, scoring='r2')
    r2_std_dev = float(np.std(cv_r2_scores))
    
    print(f"Best CV MAE: {-random_search_reg.best_score_:.3f} mm")
    print(f"R² Std Dev (Stability): ±{r2_std_dev:.4f}")
    print(f"Best parameters: {random_search_reg.best_params_}")
    
    # OBJECTIVE 2: Track convergence using validation set during training
    # Simulate incremental training to show MAE trend
    mae_trend = []
    n_estimators_final = reg_final.n_estimators
    test_intervals = [50, 100, 200, 300, 400, 500, 700, n_estimators_final]
    test_intervals = [n for n in test_intervals if n <= n_estimators_final]
    
    print("\n--- Tracking Convergence (MAE Trend) ---")
    for n_est in test_intervals:
        temp_reg = XGBRegressor(
            **{k: v for k, v in reg_final.get_params().items() if k != 'n_estimators'},
            n_estimators=n_est
        )
        temp_reg.fit(X_train_rainy, y_train_rainy)
        
        # Evaluate on validation subset (use last 20% of training data)
        val_split = int(len(X_train_rainy) * 0.8)
        X_val = X_train_rainy.iloc[val_split:]
        y_val = y_train_rainy.iloc[val_split:]
        
        y_pred_val = temp_reg.predict(X_val)
        y_pred_val = np.maximum(y_pred_val, 0)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        mae_trend.append(float(mae_val))
        print(f"  n_estimators={n_est}: MAE={mae_val:.3f} mm")
    
    # OBJECTIVE 3: Feature Importance from regressor
    print("\n--- Regressor Feature Importance ---")
    feature_importance_df_reg = pd.Series(reg_final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(feature_importance_df_reg)
    
    # Get top 5 predictors with normalized scores
    top_5_features = feature_importance_df_reg.head(5)
    top_predictors = [
        {'name': feat, 'score': float(score / feature_importance_df_reg.sum())}
        for feat, score in top_5_features.items()
    ]
    
    train_end_time = time.time()
    training_time = train_end_time - train_start_time
    
    # ========== REGRESSION EVALUATION ==========
    print("\n" + "="*60)
    print("REGRESSION EVALUATION (Test Set - Rainy Days)")
    print("="*60)
    
    rainy_test = y_test_reg > 0
    X_test_rainy = X_test[rainy_test]
    y_test_rainy = y_test_reg[rainy_test]
    
    print(f"Evaluating on {len(X_test_rainy)} rainy test days...")
    
    y_pred_rainy = reg_final.predict(X_test_rainy)
    y_pred_rainy = np.maximum(y_pred_rainy, 0)
    
    mae_test = mean_absolute_error(y_test_rainy, y_pred_rainy)
    rmse_test = np.sqrt(mean_squared_error(y_test_rainy, y_pred_rainy))
    r2_test = r2_score(y_test_rainy, y_pred_rainy)
    
    print(f"MAE:  {mae_test:.3f} mm")
    print(f"RMSE: {rmse_test:.3f} mm")
    print(f"R²:   {r2_test:.3f}")
    
    # Save regressor
    reg_final.save_model(REGRESSOR_MODEL_PATH)
    print(f"\nRegressor saved to {REGRESSOR_MODEL_PATH}")
    
    # ========== POPULATE GLOBAL_METRICS ==========
    GLOBAL_METRICS = {
        # Core metrics
        'f1_score': float(f1_test),
        'mae': float(mae_test),
        'rmse': float(rmse_test),
        'r2': float(r2_test),
        
        # OBJECTIVE 1: Stability metrics
        'f1_std_dev': f1_std_dev,
        'r2_std_dev': r2_std_dev,
        
        # OBJECTIVE 2: Convergence metrics
        'train_time': float(training_time),
        'mae_trend': mae_trend,
        
        # OBJECTIVE 3: Feature importance
        'top_predictors': top_predictors
    }
    
    print("\n" + "="*60)
    print("GLOBAL_METRICS POPULATED SUCCESSFULLY")
    print("="*60)
    print(json.dumps(GLOBAL_METRICS, indent=2))
    
    return True

# =============================================================================
# FLASK API ROUTES
# =============================================================================

@app.route('/metrics_and_prediction', methods=['GET'])
def handle_metrics_and_prediction():
    """API endpoint to get model metrics and the next day's prediction."""
    global clf_booster, reg_booster, GLOBAL_METRICS
    
    try:
        # Ensure models are trained
        if not os.path.exists(CLASSIFIER_MODEL_PATH) or not os.path.exists(REGRESSOR_MODEL_PATH) or not GLOBAL_METRICS:
            if not train_and_save_models():
                return jsonify({'error': 'Failed to train models due to data error.'}), 500
        
        # Load models for prediction
        if clf_booster is None or reg_booster is None:
            if not load_models():
                return jsonify({'error': 'Failed to load trained models.'}), 500

        # Get next day's input data
        input_data = get_last_data_point()

        # Predict
        model_features = {k: input_data[k] for k in FEATURES}
        prediction_result = predict_combined_rainfall(model_features)

        # Combine results
        response = {
            'prediction': prediction_result,
            'metrics': GLOBAL_METRICS,
            'prediction_date': input_data['prediction_date']
        }
        return jsonify(response)

    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return jsonify({'error': 'An unknown internal error occurred.'}), 500

@app.route('/predict_rain', methods=['POST'])
def handle_prediction_request():
    """API endpoint for manual input prediction."""
    if not request.is_json:
        return jsonify({'error': 'Missing JSON data in request body.'}), 400

    try:
        data = request.get_json(force=True) 
        prediction_result = predict_combined_rainfall(data)
        return jsonify(prediction_result)

    except ValueError as e:
        return jsonify({'error': str(e), 'expected_features': FEATURES}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return jsonify({'error': 'An unknown internal error occurred.'}), 500

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Check if retraining is needed
    if not os.path.exists(CLASSIFIER_MODEL_PATH) or not os.path.exists(REGRESSOR_MODEL_PATH) or not GLOBAL_METRICS:
        print("Training models for the first time...")
        if not train_and_save_models():
            exit(1)
    
    # Load models
    if load_models():
        print("\nStarting Flask API Server...")
        print("API available at:")
        print("  - http://127.0.0.1:5000/metrics_and_prediction (GET - Metrics + Next Day Prediction)")
        print("  - http://127.0.0.1:5000/predict_rain (POST - Manual Input)")
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)