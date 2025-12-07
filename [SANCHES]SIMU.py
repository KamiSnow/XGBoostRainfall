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

# ------------------------------------------------------------
# LOAD AND PREPARE DATA
# ------------------------------------------------------------
print("Loading dataset...")

df = pd.read_csv("Austin-2019-01-01-to-2023-07-22.csv")

cols = ["datetime", "tempmax", "tempmin", "temp", "humidity",
        "windspeed", "sealevelpressure", "solarradiation", "precip"]
df = df[cols].dropna()

df["datetime"] = pd.to_datetime(df["datetime"])
df["rain"] = (df["precip"] > 0).astype(int)
df["month"] = df["datetime"].dt.month
df["day_of_year"] = df["datetime"].dt.dayofyear
df = df.sort_values("datetime")

features = ["tempmax", "tempmin", "humidity", "windspeed",
            "sealevelpressure", "solarradiation", "month", "day_of_year"]

train_mask = df["datetime"].dt.year < 2023
test_mask = df["datetime"].dt.year == 2023

X_train = df.loc[train_mask, features]
X_test = df.loc[test_mask, features]
y_train_class = df.loc[train_mask, "rain"]
y_test_class = df.loc[test_mask, "rain"]
y_train_reg = df.loc[train_mask, "precip"]
y_test_reg = df.loc[test_mask, "precip"]

# ================================================================
# DEFINE SIMULATION FUNCTIONS
# ================================================================
def sop1_precision_recall_curve(X_train, y_train, X_test, y_test):
    """SOP 1 — Noisy Data Sensitivity → Precision–Recall Curve"""
    print("\nRunning SOP 1: Precision–Recall Curve Simulation...")

    # Inject noise (simulate satellite data distortion)
    X_train_noisy = X_train.copy()
    noise_idx = np.random.choice(X_train_noisy.index, size=int(0.1 * len(X_train_noisy)), replace=False)
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

    # Define models with eval_metric inside the constructor (XGBoost ≥2.0)
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

    # Fit models (no eval_metric here)
    model_low.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    model_high.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Extract validation RMSE
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

    plt.title("SOP 3: Feature Importance Table (XGBoost Gain)", fontsize=14, weight='bold')
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


# ================================================================
# MENU SYSTEM
# ================================================================
def main_menu(reg, X_train):
    while True:
        print("\nSelect an option:")
        print("1. SOP 1: Precision-Recall Curve")
        print("2. SOP 2: Learning Curve")
        print("3. SOP 3: Feature Importance Heatmap")
        print("4. Exit")

        choice = input("Select an option (1-4): ")

        if choice == "1":
            sop1_precision_recall_curve(X_train, y_train_class, X_test, y_test_class)
        elif choice == "2":
            sop2_learning_curve(X_train, y_train_reg, X_test, y_test_reg)
        elif choice == "3":
            print("\nRunning SOP 3: Feature Importance Simulation (Heatmap Style)...\n")
            sop3_feature_importance_heatmap(reg, X_train.columns)
        elif choice == "4":
            print("Exiting simulation.")
            break
        else:
            print("Invalid choice. Try again.")


# ------------------------------------------------------------
# TRAIN MODELS BEFORE MENU
# ------------------------------------------------------------
print("\nTraining baseline models...")

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

# ------------------------------------------------------------
# RUN PROGRAM
# ------------------------------------------------------------
if __name__ == "__main__":
    main_menu(reg, X_train)
