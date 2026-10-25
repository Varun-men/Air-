"""
Data cleaning and India National AQI (CPCB) calculation.
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# India CPCB AQI Breakpoints
# Format: list of (concentration_breakpoint, aqi_breakpoint) tuples
# ------------------------------------------------------------------ #
AQI_BREAKPOINTS = {
    "pm25": [
        (0, 0), (30, 50), (60, 100), (90, 200), (120, 300), (250, 400), (350, 500)
    ],
    "pm10": [
        (0, 0), (50, 50), (100, 100), (250, 200), (350, 300), (430, 400), (500, 500)
    ],
    "no2": [
        (0, 0), (40, 50), (80, 100), (180, 200), (280, 300), (400, 400), (500, 500)
    ],
    "o3": [
        (0, 0), (50, 50), (100, 100), (168, 200), (208, 300), (748, 400), (748, 500)
    ],
    "co": [
        (0, 0), (1, 50), (2, 100), (10, 200), (17, 300), (34, 400), (34, 500)
    ],
    "so2": [
        (0, 0), (40, 50), (80, 100), (380, 200), (800, 300), (1600, 400), (1600, 500)
    ],
}

AQI_CATEGORIES = [
    (50,  "Good",         "#00B050"),
    (100, "Satisfactory", "#92D050"),
    (200, "Moderate",     "#FFFF00"),
    (300, "Poor",         "#FF7C00"),
    (400, "Very Poor",    "#FF0000"),
    (500, "Severe",       "#C00000"),
]


def calculate_sub_index(value: float, breakpoints: list) -> float:
    """
    Calculate AQI sub-index for a pollutant using linear interpolation.

    Parameters
    ----------
    value : float
        Pollutant concentration.
    breakpoints : list of (concentration_bp, aqi_bp) tuples
        Official CPCB breakpoints for the pollutant.

    Returns
    -------
    float
        AQI sub-index value.
    """
    if pd.isna(value) or value < 0:
        return np.nan

    conc_bps = [bp[0] for bp in breakpoints]
    aqi_bps = [bp[1] for bp in breakpoints]

    # Clamp to max breakpoint
    if value >= conc_bps[-1]:
        return float(aqi_bps[-1])

    # Find the interval
    for i in range(len(conc_bps) - 1):
        c_lo, c_hi = conc_bps[i], conc_bps[i + 1]
        a_lo, a_hi = aqi_bps[i], aqi_bps[i + 1]
        if c_lo <= value <= c_hi:
            if c_hi == c_lo:
                return float(a_lo)
            sub_index = a_lo + (a_hi - a_lo) * (value - c_lo) / (c_hi - c_lo)
            return round(float(sub_index), 2)

    return np.nan


def calculate_aqi(row: pd.Series) -> float:
    """
    Compute overall AQI as the maximum sub-index across all available pollutants.

    Parameters
    ----------
    row : pd.Series
        A DataFrame row containing pollutant columns.

    Returns
    -------
    float
        Overall AQI value.
    """
    sub_indices = []
    for pollutant, bps in AQI_BREAKPOINTS.items():
        val = row.get(pollutant, np.nan)
        if not pd.isna(val):
            si = calculate_sub_index(float(val), bps)
            if not pd.isna(si):
                sub_indices.append(si)

    if not sub_indices:
        return np.nan

    return round(max(sub_indices), 1)


def dominant_pollutant(row: pd.Series) -> str:
    """Return the pollutant contributing the highest sub-index."""
    best_pollutant = None
    best_si = -1
    for pollutant, bps in AQI_BREAKPOINTS.items():
        val = row.get(pollutant, np.nan)
        if not pd.isna(val):
            si = calculate_sub_index(float(val), bps)
            if not pd.isna(si) and si > best_si:
                best_si = si
                best_pollutant = pollutant
    return best_pollutant or "Unknown"


def get_aqi_category(aqi: float) -> tuple:
    """
    Return (category_label, color_hex) for a given AQI value.

    Parameters
    ----------
    aqi : float

    Returns
    -------
    tuple (str, str)
    """
    if pd.isna(aqi):
        return ("Unknown", "#AAAAAA")
    for threshold, label, color in AQI_CATEGORIES:
        if aqi <= threshold:
            return (label, color)
    return ("Severe", "#C00000")


SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Monsoon/Autumn", 10: "Monsoon/Autumn", 11: "Monsoon/Autumn",
}

SEASON_NUM_MAP = {
    "Winter": 0, "Spring": 1, "Summer": 2, "Monsoon/Autumn": 3
}


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data, compute AQI and add temporal features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with at minimum a 'date' column and pollutant columns.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with AQI, category, and temporal columns added.
    """
    df = df.copy()

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Drop exact duplicate dates (keep first)
    df = df.drop_duplicates(subset=["date"], keep="first")

    # Fill short gaps with rolling mean (window=7 days)
    pollutant_cols = [c for c in ["pm25", "pm10", "no2", "o3", "co", "so2"] if c in df.columns]
    for col in pollutant_cols:
        df[col] = df[col].where(df[col] >= 0)  # remove negative values
        rolling_fill = df[col].rolling(window=7, min_periods=1, center=True).mean()
        df[col] = df[col].fillna(rolling_fill)

    # Compute AQI
    df["aqi"] = df.apply(calculate_aqi, axis=1)
    df["aqi"] = df["aqi"].fillna(df["aqi"].rolling(7, min_periods=1, center=True).mean())

    # AQI category and color
    df["aqi_category"] = df["aqi"].apply(lambda x: get_aqi_category(x)[0])
    df["aqi_color"] = df["aqi"].apply(lambda x: get_aqi_category(x)[1])

    # Dominant pollutant
    df["dominant_pollutant"] = df.apply(dominant_pollutant, axis=1)

    # Temporal features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%b")
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_name"] = df["date"].dt.strftime("%A")
    df["season"] = df["month"].map(SEASON_MAP)
    df["season_num"] = df["season"].map(SEASON_NUM_MAP)

    return df
