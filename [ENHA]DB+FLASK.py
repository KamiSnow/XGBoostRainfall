import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from flask import Flask, request, jsonify
from flask_cors import CORS # Used to allow your frontend to connect
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, f1_score

# =============================================================================
# 1. FLASK APPLICATION SETUP AND CONFIGURATION
# =============================================================================
app = Flask(__name__)
CORS(app) # Enable CORS for all routes (important for front-end integration)

# Define file paths for the saved models
CLASSIFIER_MODEL_PATH = 'xgb_classifier_rain_occurrence.json'
REGRESSOR_MODEL_PATH = 'xgb_regressor_rain_amount.json'

# Global model variables
clf_booster = None # Renamed to avoid confusion with clf_final in training
reg_booster = None # Renamed to avoid confusion with reg_final in training

# List of features your model was trained on - MUST MATCH THE INPUT DATA from frontend
FEATURES = [
    'tempmax', 'tempmin',
    'humidity', 'windspeed', 'sealevelpressure',
    'month', 'day_of_year',
    'precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1',
    'precip_roll3', 'humidity_roll3'
]

# =============================================================================
# 2. MODEL LOADING FUNCTION
# =============================================================================

def load_models():
    """Load the trained XGBoost models from disk."""
    global clf_booster, reg_booster
    try:
        # Load Classifier (Rain Occurrence)
        clf_booster = xgb.Booster()
        clf_booster.load_model(CLASSIFIER_MODEL_PATH)
        # Load Regressor (Rain Amount)
        reg_booster = xgb.Booster()
        reg_booster.load_model(REGRESSOR_MODEL_PATH)
        print("Models loaded successfully.")
        return True
    except xgb.core.XGBoostError as e:
        print(f"Error loading XGBoost model files: {e}")
        print(f"Models expected at: {CLASSIFIER_MODEL_PATH} and {REGRESSOR_MODEL_PATH}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        return False

# =============================================================================
# 3. CORE PREDICTION LOGIC
# =============================================================================

def predict_combined_rainfall(data: dict):
    """
    Performs the two-stage prediction based on input features.
    
    The key change here is explicitly extracting values in the correct order 
    required by the xgb.Booster objects.
    """
    if clf_booster is None or reg_booster is None:
        raise RuntimeError("Models are not loaded on the server.")
        
    # Prepare input features
    try:
        # CRITICAL FIX: Ensure the features are extracted and ordered correctly
        input_data_list = [data[feature] for feature in FEATURES]
        
        # Convert the single list of feature values into a NumPy array 
        # (with shape (1, 12)) and then to DMatrix for XGBoost prediction
        input_Dmatrix = xgb.DMatrix(np.array([input_data_list]))
        
    except KeyError as e:
        # This error handles cases where the frontend is missing a required field
        # The key names in the JSON must match the list in FEATURES
        raise ValueError(f"Missing required feature in request: {str(e).strip('()')}. Expected features: {FEATURES}")
    except Exception as e:
        raise RuntimeError(f"Data preparation failed: {e}")

    # Stage 1: Classification (Rain Occurrence)
    # Get the raw probability of rain (class 1). Prediction returns a float array, hence [0]
    rain_prob = clf_booster.predict(input_Dmatrix)[0]
    
    # Predict the binary class (1 for rain, 0 for no rain)
    rain_occurrence = 1 if rain_prob > 0.5 else 0
    
    predicted_amount = 0.0

    # Stage 2: Regression (Rain Amount), only if rain is predicted
    if rain_occurrence == 1:
        # Prediction returns a float array, hence [0]
        predicted_amount = reg_booster.predict(input_Dmatrix)[0]
        # Ensure non-negative and cast to standard Python float for JSON serialization
        predicted_amount = float(np.maximum(predicted_amount, 0))
    
    return {
        'rain_occurrence': rain_occurrence,
        'rain_probability': float(rain_prob), # Ensure float conversion
        'predicted_rainfall_mm': predicted_amount
    }

# =============================================================================
# 4. FLASK API ROUTE DEFINITION
# =============================================================================

@app.route('/predict_rain', methods=['POST'])
def handle_prediction_request():
    """API endpoint to receive data and return rainfall prediction."""
    if not request.is_json: # Use is_json for better request type checking
        return jsonify({'error': 'Missing JSON data in request body. Content-Type must be application/json.'}), 400

    try:
        data = request.get_json(force=True)
        # Call the core prediction logic
        prediction_result = predict_combined_rainfall(data)
        
        return jsonify(prediction_result)

    except ValueError as e:
        # Handles missing features/incorrect data structure
        return jsonify({'error': str(e), 'expected_features': FEATURES}), 400
    except RuntimeError as e:
        # Handles models not loaded
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}")
        return jsonify({'error': 'An unknown internal error occurred.'}), 500

# =============================================================================
# 5. ORIGINAL ML PIPELINE (TRAINING AND SAVING) - UNCHANGED
# =============================================================================

def train_and_save_models():
    """Execute the full ML pipeline to train the models."""
    print("="*60)
    print("EXECUTING FULL ML PIPELINE: TRAINING AND SAVING MODELS")
    print("="*60)
    
    # --- Step 1 & 2: Data Loading, Preprocessing, and Cleaning ---
    print("Step 1 & 2: Loading, Cleaning, and Preprocessing...")
    
    # Define the dataset folder location
    DATASET_FOLDER = 'dataset/'
    
    try:
        # NOTE: You must have these files in a 'dataset/' folder relative to this script
        naia_daily = pd.read_csv(DATASET_FOLDER + "NAIA Daily Data.csv")
        port_daily = pd.read_csv(DATASET_FOLDER + "Port Area Daily Data.csv")
        science_daily = pd.read_csv(DATASET_FOLDER + "Science Garden Daily Data.csv")
        pressure_daily = pd.read_csv(DATASET_FOLDER + "Mean Sea Level Pressure Daily Data.csv", skiprows=[1])
    except FileNotFoundError as e:
        print(f"ERROR: Data file not found: {e}")
        print(f"Please ensure the required CSV files are inside the '{DATASET_FOLDER}' directory.")
        return False
        
    for df_ in [naia_daily, port_daily, science_daily]:
        df_['datetime'] = pd.to_datetime(df_[['YEAR', 'MONTH', 'DAY']])
        
    pressure_daily['datetime'] = pd.to_datetime(pressure_daily['Date(UTC)'])
    pressure_daily = pressure_daily.drop('Date(UTC)', axis=1)

    def clean_weather_data(df_, station_name):
        df_ = df_.replace(-999.0, np.nan)
        if 'RAINFALL' in df_.columns:
            df_['RAINFALL'] = df_['RAINFALL'].replace(-1.0, 0.05)
            df_.loc[df_['RAINFALL'] < 0, 'RAINFALL'] = 0
        if 'RH' in df_.columns:
            df_['RH'] = np.clip(df_['RH'], 0, 100)
        if 'WIND_SPEED' in df_.columns:
            df_.loc[df_['WIND_SPEED'] < 0, 'WIND_SPEED'] = 0
        return df_

    naia_daily = clean_weather_data(naia_daily, "NAIA")
    port_daily = clean_weather_data(port_daily, "Port Area")
    science_daily = clean_weather_data(science_daily, "Science Garden")
    pressure_daily = pressure_daily.replace(-999.0, np.nan)

    # --- Step 3 & 4: Merging and Aggregation ---
    print("Step 3 & 4: Merging and Aggregation...")

    # Rename and merge
    naia_daily = naia_daily.rename(columns={'RAINFALL': 'rainfall_naia', 'TMAX': 'tmax_naia', 'TMIN': 'tmin_naia', 'RH': 'humidity_naia', 'WIND_SPEED': 'windspeed_naia', 'WIND_DIRECTION': 'winddir_naia'})
    port_daily = port_daily.rename(columns={'RAINFALL': 'rainfall_port', 'TMAX': 'tmax_port', 'TMIN': 'tmin_port', 'RH': 'humidity_port', 'WIND_SPEED': 'windspeed_port', 'WIND_DIRECTION': 'winddir_port'})
    science_daily = science_daily.rename(columns={'RAINFALL': 'rainfall_science', 'TMAX': 'tmax_science', 'TMIN': 'tmin_science', 'RH': 'humidity_science', 'WIND_SPEED': 'windspeed_science', 'WIND_DIRECTION': 'winddir_science'})
    
    df = naia_daily[['datetime', 'rainfall_naia', 'tmax_naia', 'tmin_naia', 'humidity_naia', 'windspeed_naia', 'winddir_naia']].copy()
    df = df.merge(port_daily[['datetime', 'rainfall_port', 'tmax_port', 'tmin_port', 'humidity_port', 'windspeed_port', 'winddir_port']], on='datetime', how='outer')
    df = df.merge(science_daily[['datetime', 'rainfall_science', 'tmax_science', 'tmin_science', 'humidity_science', 'windspeed_science', 'winddir_science']], on='datetime', how='outer')
    df = df.merge(pressure_daily, on='datetime', how='outer')
    
    df = df.rename(columns={'NAIA Pasay City, M.Manila Press.QFF.Dly [hPa]': 'pressure_naia', 'Port Area, Manila Press.QFF.Dly [hPa]': 'pressure_port', 'Science Garden Quezon City, Metro Manila Press.QFF.Dly [hPa]': 'pressure_science'})
    df = df.sort_values('datetime').reset_index(drop=True)

    df['tempmax'] = df[['tmax_naia', 'tmax_port', 'tmax_science']].mean(axis=1)
    df['tempmin'] = df[['tmin_naia', 'tmin_port', 'tmin_science']].mean(axis=1)
    df['humidity'] = df[['humidity_naia', 'humidity_port', 'humidity_science']].mean(axis=1)
    df['windspeed'] = df[['windspeed_naia', 'windspeed_port', 'windspeed_science']].mean(axis=1)
    df['sealevelpressure'] = df[['pressure_naia', 'pressure_port', 'pressure_science']].mean(axis=1)
    df['precip'] = df[['rainfall_naia', 'rainfall_port', 'rainfall_science']].max(axis=1)

    df = df.dropna(subset=['tempmax', 'tempmin', 'humidity', 'precip'], how='all')
    df = df.set_index('datetime').ffill(limit=2).reset_index()
    df = df.dropna(subset=['tempmax', 'tempmin', 'humidity', 'windspeed', 'sealevelpressure', 'precip'])
    
    # --- Step 5: Feature Engineering ---
    print("Step 5: Feature Engineering...")
    df['rain'] = (df['precip'] > 0).astype(int)
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Lagged features
    for col in ['precip', 'humidity', 'sealevelpressure']:
        df[f'{col}_lag1'] = df[col].shift(1)
        
    # Rolling features
    df['precip_roll3'] = df['precip'].rolling(window=3, min_periods=1).mean()
    df['humidity_roll3'] = df['humidity'].rolling(window=3, min_periods=1).mean()

    df = df.dropna(subset=['precip_lag1', 'humidity_lag1', 'sealevelpressure_lag1'])

    # --- Step 6: Prepare Features and Split Data ---
    print("Step 6: Preparing Features and Splitting Data...")
    
    X = df[FEATURES] # Uses the globally defined FEATURES list
    y_class = df['rain']
    y_reg = df['precip']

    split_idx = int(len(df) * 0.8)
    train_mask = df.index < split_idx
    # test_mask = df.index >= split_idx # Test mask is not used in training, removed for brevity

    X_train, y_train_class, y_train_reg = X[train_mask], y_class[train_mask], y_reg[train_mask]
    
    # Filter for rainy days for regression training
    rainy_train = y_train_reg > 0
    X_train_rainy = X_train[rainy_train]
    y_train_rainy = y_train_reg[rainy_train]

    # --- Step 7: Advanced Hyperparameter Tuning & Training (Classifier) ---
    print("\nStarting CLASSIFICATION Tuning...")
    
    param_grid_class_broad = {
        'learning_rate': [0.005, 0.01, 0.05, 0.1], 'max_depth': [5, 7, 9],
        'n_estimators': [500, 1000], 'colsample_bytree': [0.7, 0.9],
        'subsample': [0.7, 0.9], 'min_child_weight': [1, 3],
        'gamma': [0, 0.1], 'reg_alpha': [0, 0.1], 'reg_lambda': [1, 2]
    } 
    
    base_clf = XGBClassifier(
        scale_pos_weight=len(y_train_class[y_train_class==0]) / len(y_train_class[y_train_class==1]),
        eval_metric='logloss', random_state=42, tree_method='hist'
    )

    random_search_clf = RandomizedSearchCV(
        base_clf, param_distributions=param_grid_class_broad, n_iter=20, 
        scoring='f1', cv=3, verbose=0, random_state=42, n_jobs=-1
    )
    random_search_clf.fit(X_train, y_train_class)
    clf_final = random_search_clf.best_estimator_
    print(f"Classifier trained. Best F1 Score: {random_search_clf.best_score_:.3f}")
    
    # Save the final classification model (uses the trained model object)
    clf_final.save_model(CLASSIFIER_MODEL_PATH)

    # --- Step 8: Advanced Hyperparameter Tuning & Training (Regressor) ---
    print("Starting REGRESSION Tuning...")

    param_grid_reg_broad = {
        'learning_rate': [0.005, 0.01, 0.05, 0.1], 'max_depth': [4, 6, 8],
        'n_estimators': [500, 1000], 'colsample_bytree': [0.7, 0.9],
        'subsample': [0.7, 0.9], 'min_child_weight': [1, 3],
        'gamma': [0, 0.1], 'reg_alpha': [0, 0.1], 'reg_lambda': [1, 2]
    } 

    base_reg = XGBRegressor(random_state=42, tree_method='hist')

    random_search_reg = RandomizedSearchCV(
        base_reg, param_distributions=param_grid_reg_broad, n_iter=20, 
        scoring='neg_mean_absolute_error', cv=3, verbose=0, random_state=42, n_jobs=-1
    )
    random_search_reg.fit(X_train_rainy, y_train_rainy)
    reg_final = random_search_reg.best_estimator_
    print(f"Regressor trained. Best MAE: {-random_search_reg.best_score_:.3f} mm")

    # Save the final regression model (uses the trained model object)
    reg_final.save_model(REGRESSOR_MODEL_PATH)
    
    print("\nModels successfully trained and saved!")
    return True

# =============================================================================
# 6. MAIN EXECUTION BLOCK
# =============================================================================
if __name__ == '__main__':
    
    # Check if models exist. If not, train them.
    if not os.path.exists(CLASSIFIER_MODEL_PATH) or not os.path.exists(REGRESSOR_MODEL_PATH):
        if not train_and_save_models():
            # Exit if training failed (e.g., due to missing data files)
            exit(1)

    # Load the models before starting the application
    if load_models():
        print("\nStarting Flask API Server...")
        print("API is available at http://127.0.0.1:5000/predict_rain")
        
        # Start the server
        app.run(debug=True, host='127.0.0.1', port=5000)