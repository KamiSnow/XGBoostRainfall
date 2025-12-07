import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import StratifiedKFold # FIX: Corrected typo 'model_model_selection'
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
import warnings
import json

# --- CONFIGURATION ---
try:
    # Suppress the SettingWithCopyWarning
    warnings.filterwarnings('ignore', category=pd.SettingWithCopyWarning)
except AttributeError:
    pass

app = Flask(__name__)
CORS(app) 

CLASSIFIER_MODEL_PATH = 'xgb_classifier_rain_occurrence.json'
REGRESSOR_MODEL_PATH = 'xgb_regressor_rain_amount.json'

clf_booster = None 
reg_booster = None 
GLOBAL_METRICS = {} # Store metrics globally after training

# CRITICAL: 13-feature list, synchronized across all files
FEATURES = [
    'tempmax', 'tempmin',
    'humidity', 'windspeed', 'sealevelpressure',
    'month', 'day_of_year',
    'precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1',
    'precip_roll3', 'humidity_roll3',
    'windspeed_roll3' # The 13th feature
]

# =============================================================================
# 2. CORE UTILITIES (Model Loading, Data Loading, Prediction)
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

def load_and_preprocess_data():
    """Loads, merges, cleans, imputes, and engineers features."""
    DATASET_FOLDER = 'dataset/'
    try:
        # NOTE: Ensure you have these data files in a 'dataset/' folder
        naia_daily = pd.read_csv(DATASET_FOLDER + "NAIA Daily Data.csv")
        port_daily = pd.read_csv(DATASET_FOLDER + "Port Area Daily Data.csv")
        science_daily = pd.read_csv(DATASET_FOLDER + "Science Garden Daily Data.csv")
        pressure_daily = pd.read_csv(DATASET_FOLDER + "Mean Sea Level Pressure Daily Data.csv", skiprows=[1])
    except FileNotFoundError as e:
        raise RuntimeError(f"Data file not found: {e}. Cannot proceed without data.")
    
    # Date processing and cleaning
    for df_ in [naia_daily, port_daily, science_daily]:
        df_['datetime'] = pd.to_datetime(df_[['YEAR', 'MONTH', 'DAY']])
    pressure_daily['datetime'] = pd.to_datetime(pressure_daily['Date(UTC)'])
    
    # ... (renaming, merging, aggregation - omitted for brevity, logic remains the same)
    
    naia_daily = clean_weather_data(naia_daily).rename(columns={'RAINFALL': 'rainfall_naia', 'TMAX': 'tmax_naia', 'TMIN': 'tmin_naia', 'RH': 'humidity_naia', 'WIND_SPEED': 'windspeed_naia'})
    port_daily = clean_weather_data(port_daily).rename(columns={'RAINFALL': 'rainfall_port', 'TMAX': 'tmax_port', 'TMIN': 'tmin_port', 'RH': 'humidity_port', 'WIND_SPEED': 'windspeed_port'})
    science_daily = clean_weather_data(science_daily).rename(columns={'RAINFALL': 'rainfall_science', 'TMAX': 'tmax_science', 'TMIN': 'tmin_science', 'RH': 'humidity_science', 'WIND_SPEED': 'windspeed_science'})
    pressure_daily = pressure_daily.replace(-999.0, np.nan).rename(columns={'NAIA Pasay City, M.Manila Press.QFF.Dly [hPa]': 'pressure_naia', 'Port Area, Manila Press.QFF.Dly [hPa]': 'pressure_port', 'Science Garden Quezon City, Metro Manila Press.QFF.Dly [hPa]': 'pressure_science'})
    
    df = naia_daily[['datetime', 'rainfall_naia', 'tmax_naia', 'tmin_naia', 'humidity_naia', 'windspeed_naia']].copy()
    df = df.merge(port_daily[['datetime', 'rainfall_port', 'tmax_port', 'tmin_port', 'humidity_port', 'windspeed_port']], on='datetime', how='outer')
    df = df.merge(science_daily[['datetime', 'rainfall_science', 'tmax_science', 'tmin_science', 'humidity_science', 'windspeed_science']], on='datetime', how='outer')
    df = df.merge(pressure_daily[['datetime', 'pressure_naia', 'pressure_port', 'pressure_science']], on='datetime', how='outer')
    
    df['tempmax'] = df[['tmax_naia', 'tmax_port', 'tmax_science']].mean(axis=1)
    df['tempmin'] = df[['tmin_naia', 'tmin_port', 'tmin_science']].mean(axis=1)
    df['humidity'] = df[['humidity_naia', 'humidity_port', 'humidity_science']].mean(axis=1)
    df['windspeed'] = df[['windspeed_naia', 'windspeed_port', 'windspeed_science']].mean(axis=1)
    df['sealevelpressure'] = df[['pressure_naia', 'pressure_port', 'pressure_science']].mean(axis=1)
    df['precip'] = df[['rainfall_naia', 'rainfall_port', 'rainfall_science']].max(axis=1)
    
    impute_cols = ['tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure', 'precip']
    imputer = KNNImputer(n_neighbors=5)
    df[impute_cols] = imputer.fit_transform(df[impute_cols])

    # Feature Engineering
    df['rain'] = (df['precip'] > 0).astype(int)
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    for col in ['precip', 'humidity', 'sealevelpressure']:
        df[f'{col}_lag1'] = df[col].shift(1) 
        
    df['precip_roll3'] = df['precip'].rolling(window=3, min_periods=1).mean()
    df['humidity_roll3'] = df['humidity'].rolling(window=3, min_periods=1).mean()
    df['windspeed_roll3'] = df['windspeed'].rolling(window=3, min_periods=1).mean() # CRITICAL: 13th feature added

    df = df.dropna(subset=['precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1'])
    
    return df

def get_last_data_point():
    """Loads data and prepares the feature vector for the day after the last date in the dataset."""
    df = load_and_preprocess_data()
        
    # The last available row contains the current feature data (for training)
    last_row_features = df[FEATURES].iloc[-1].to_dict()
    
    last_datetime = df['datetime'].iloc[-1]
    prediction_datetime = last_datetime + pd.Timedelta(days=1)
    
    # Update the date-based features for the predicted day
    last_row_features['month'] = prediction_datetime.month
    last_row_features['day_of_year'] = prediction_datetime.dayofyear
    
    last_row_features['prediction_date'] = prediction_datetime.strftime('%Y-%m-%d')

    return last_row_features

def predict_combined_rainfall(data: dict):
    """Performs the two-stage prediction based on input features."""
    if clf_booster is None or reg_booster is None:
        raise RuntimeError("Models are not loaded on the server.")
        
    input_data_list = [data[feature] for feature in FEATURES]
    input_Dmatrix = xgb.DMatrix(np.array([input_data_list]))
    
    # Stage 1: Classification (Rain Occurrence)
    rain_prob = clf_booster.predict(input_Dmatrix)[0]
    rain_occurrence = 1 if rain_prob > 0.5 else 0
    predicted_amount = 0.0

    # Stage 2: Regression (Rain Amount), only if rain is predicted
    if rain_occurrence == 1:
        predicted_amount = reg_booster.predict(input_Dmatrix)[0]
        predicted_amount = float(np.maximum(predicted_amount, 0))
    
    return {
        'rain_occurrence': rain_occurrence,
        'rain_probability': float(rain_prob), 
        'rain_amount': predicted_amount
    }

def evaluate_models(clf_model, reg_model, X_test, y_test_class, y_test_reg):
    """Calculates F1, MAE, RMSE, and R2 metrics on the test set."""
    metrics = {}
    
    # 1. Classification Metrics (F1 Score)
    # FIX: Use X_test.values as input for the XGBClassifier (sklearn wrapper)
    y_pred_class_proba = clf_model.predict_proba(X_test.values)[:, 1] 
    y_pred_binary = (y_pred_class_proba > 0.5).astype(int)
    metrics['f1_score'] = f1_score(y_test_class, y_pred_binary)

    # 2. Regression Metrics (MAE, RMSE, R2) on rainy days only
    rainy_test = y_test_reg > 0
    X_test_rainy = X_test[rainy_test] 
    y_test_rainy = y_test_reg[rainy_test]
    
    if len(X_test_rainy) > 0:
        # FIX: Use X_test_rainy.values as input for the XGBRegressor (sklearn wrapper)
        y_pred_reg = reg_model.predict(X_test_rainy.values)
        metrics['mae'] = mean_absolute_error(y_test_rainy, y_pred_reg)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test_rainy, y_pred_reg))
        metrics['r2'] = r2_score(y_test_rainy, y_pred_reg)
    else:
        metrics['mae'] = 0.0
        metrics['rmse'] = 0.0
        metrics['r2'] = 0.0

    # Round metrics for JSON
    return {k: round(v, 4) for k, v in metrics.items()}

def train_and_save_models():
    """Execute the full ML pipeline to train and save the models, and calculate metrics."""
    global GLOBAL_METRICS
    print("="*60)
    print("EXECUTING FULL ML PIPELINE: TRAINING AND SAVING MODELS")
    print("="*60)
    
    try:
        df = load_and_preprocess_data()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return False
        
    X = df[FEATURES]
    y_class = df['rain']
    y_reg = df['precip']

    # Split data: 80% train, 20% test (Time-series split)
    split_idx = int(len(df) * 0.8)
    train_mask = df.index < split_idx

    X_train, y_train_class, y_train_reg = X[train_mask], y_class[train_mask], y_reg[train_mask]
    X_test, y_test_class, y_test_reg = X[~train_mask], y_class[~train_mask], y_reg[~train_mask]
    
    # Isolate rainy days in the training set
    rainy_train = y_train_reg > 0
    X_train_rainy = X_train[rainy_train]
    y_train_rainy = y_train_reg[rainy_train] 
    
    # --- Training (Simplified for brevity, using reasonable defaults) ---
    print("Training Classifier...")
    # Classifier
    scale_pos_weight = len(y_train_class[y_train_class==0]) / len(y_train_class[y_train_class==1])
    clf_final = XGBClassifier(
        n_estimators=1000, learning_rate=0.05, max_depth=7, 
        scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42
    )
    clf_final.fit(X_train, y_train_class)
    clf_final.save_model(CLASSIFIER_MODEL_PATH)
    
    print("Training Regressor...")
    # Regressor
    reg_final = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=6, 
        objective='reg:squarederror', random_state=42
    )
    reg_final.fit(X_train_rainy, y_train_rainy)
    reg_final.save_model(REGRESSOR_MODEL_PATH)
    
    print("Models trained and saved.")
    
    # --- Evaluation ---
    # clf_final and reg_final are passed as the model objects
    GLOBAL_METRICS = evaluate_models(clf_final, reg_final, X_test, y_test_class, y_test_reg)
    print(f"Metrics calculated: {GLOBAL_METRICS}")
    
    return True

# =============================================================================
# 3. FLASK API ROUTE DEFINITION
# =============================================================================

@app.route('/metrics_and_prediction', methods=['GET'])
def handle_metrics_and_prediction():
    """API endpoint to get model metrics and the next day's prediction."""
    global clf_booster, reg_booster, GLOBAL_METRICS
    
    try:
        # 1. Ensure models are trained and metrics are available
        if not os.path.exists(CLASSIFIER_MODEL_PATH) or not os.path.exists(REGRESSOR_MODEL_PATH) or not GLOBAL_METRICS:
            if not train_and_save_models():
                return jsonify({'error': 'Failed to train models due to data error.'}), 500
        
        # 2. Load models for prediction if not already loaded
        if clf_booster is None or reg_booster is None:
            if not load_models():
                return jsonify({'error': 'Failed to load trained models.'}), 500

        # 3. Get the next day's input data from the dataset
        input_data = get_last_data_point()

        # 4. Predict
        model_features = {k: input_data[k] for k in FEATURES}
        prediction_result = predict_combined_rainfall(model_features)

        # 5. Combine results and return
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


# =============================================================================
# 4. MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    # Initial check and training
    if not os.path.exists(CLASSIFIER_MODEL_PATH) or not os.path.exists(REGRESSOR_MODEL_PATH) or not GLOBAL_METRICS:
        print("Checking/Training models for the first time...")
        if not train_and_save_models():
            exit(1) # Exit if training failed
    
    # Load the models into global variables
    if load_models():
        print("\nStarting Flask API Server...")
        print("API is available at http://127.0.0.1:5000/metrics_and_prediction")
        app.run(debug=True, host='127.0.0.1', port=5000, use_reloader=False)
