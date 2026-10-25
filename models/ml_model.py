"""
Random Forest model for AQI prediction.
Supports per-city model files via optional model_path parameter.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = MODEL_DIR / "rf_model.joblib"

FEATURE_COLS = ["pm25", "pm10", "no2", "o3", "co", "so2", "month", "day_of_week", "season_num"]
TARGET_COL = "aqi"


def _resolve_model_path(model_path=None, city: str = None) -> Path:
    """
    Resolve the path to a model file.
    Priority: explicit model_path > city-derived path > default path.
    """
    if model_path is not None:
        return Path(model_path)
    if city is not None:
        return MODEL_DIR / f"{city.lower()}_rf_model.joblib"
    return DEFAULT_MODEL_PATH


def prepare_features(df: pd.DataFrame):
    """
    Build feature matrix X and target vector y from the DataFrame.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    data = df[available_features + [TARGET_COL]].dropna()

    X = data[available_features]
    y = data[TARGET_COL]

    # Forward-fill any remaining NaN in X
    X = X.ffill().bfill().fillna(0)

    return X, y


def train_model(df: pd.DataFrame, model_path=None, city: str = None) -> tuple:
    """
    Train a Random Forest regressor and a Linear Regression for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data with feature columns and AQI target.
    model_path : str or Path, optional
        Explicit path to save the trained model. If omitted and city is
        provided, a city-specific path is used automatically.
    city : str, optional
        City name used to derive the model save path when model_path is None.

    Returns
    -------
    tuple : (rf_model, metrics_dict, feature_importance_df)
    """
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    X, y = prepare_features(df)

    if len(X) < 30:
        raise ValueError(f"Insufficient data for training: only {len(X)} rows available.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_mae  = mean_absolute_error(y_test, y_pred_rf)
    rf_rmse = root_mean_squared_error(y_test, y_pred_rf)
    rf_r2   = r2_score(y_test, y_pred_rf)

    # Linear Regression (for comparison)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)

    lr_mae  = mean_absolute_error(y_test, y_pred_lr)
    lr_rmse = root_mean_squared_error(y_test, y_pred_lr)
    lr_r2   = r2_score(y_test, y_pred_lr)

    metrics = {
        "rf": {"MAE": round(rf_mae, 2), "RMSE": round(rf_rmse, 2), "R2": round(rf_r2, 4)},
        "lr": {"MAE": round(lr_mae, 2), "RMSE": round(lr_rmse, 2), "R2": round(lr_r2, 4)},
        "y_test": y_test.values,
        "y_pred_rf": y_pred_rf,
        "y_pred_lr": y_pred_lr,
        "feature_names": list(X.columns),
    }

    # Feature importance
    fi_df = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save model
    save_path = _resolve_model_path(model_path, city)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, save_path)
    logger.info(f"Model saved to {save_path}")

    return rf, metrics, fi_df


def load_model(model_path=None, city: str = None):
    """
    Load the saved Random Forest model.

    Parameters
    ----------
    model_path : str or Path, optional
        Explicit path to the model file.
    city : str, optional
        City name — used to derive the model path when model_path is None.

    Returns
    -------
    Loaded model object, or None if not found / load fails.
    """
    try:
        import joblib
        path = _resolve_model_path(model_path, city)
        if path.exists():
            return joblib.load(path)
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
    return None


def predict_aqi(
    model,
    pm25: float,
    pm10: float,
    no2: float,
    o3: float,
    co: float,
    so2: float,
    month: int,
    day_of_week: int,
    season_num: int = None,
) -> tuple:
    """
    Predict AQI from input pollutant concentrations.

    Returns
    -------
    tuple : (predicted_aqi_float, category_str, color_hex)
    """
    from analysis.preprocess import get_aqi_category, SEASON_MAP, SEASON_NUM_MAP

    if season_num is None:
        season_str = SEASON_MAP.get(month, "Spring")
        season_num = SEASON_NUM_MAP.get(season_str, 1)

    # Build feature row using available feature columns
    feature_values = {
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "o3": o3,
        "co": co,
        "so2": so2,
        "month": month,
        "day_of_week": day_of_week,
        "season_num": season_num,
    }

    # Only use features the model was trained on
    if hasattr(model, "feature_names_in_"):
        feature_order = model.feature_names_in_
    else:
        feature_order = FEATURE_COLS

    row = pd.DataFrame([[feature_values.get(f, 0) for f in feature_order]], columns=feature_order)
    predicted = float(model.predict(row)[0])
    predicted = max(0, min(500, round(predicted, 1)))

    category, color = get_aqi_category(predicted)
    return predicted, category, color


def cross_validate_model(df: pd.DataFrame) -> float:
    """
    5-fold cross-validation of the Random Forest model.

    Returns
    -------
    float : mean R² across folds.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    X, y = prepare_features(df)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    mean_r2 = float(np.mean(scores))
    logger.info(f"5-fold CV R² scores: {scores.round(4)}, mean={mean_r2:.4f}")
    return mean_r2
