"""
Trend analysis functions for Delhi air quality data.
"""

import pandas as pd
import numpy as np


MONTH_ORDER = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

SEASON_ORDER = ["Winter", "Spring", "Summer", "Monsoon/Autumn"]

POLLUTANT_COLS = ["pm25", "pm10", "no2", "o3", "co", "so2"]


def monthly_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly averages of all pollutants and AQI.

    Returns a DataFrame indexed by month number (1–12) with a month_name column.
    """
    cols = [c for c in POLLUTANT_COLS + ["aqi"] if c in df.columns]
    monthly = (
        df.groupby("month")[cols]
        .mean()
        .reset_index()
        .round(2)
    )
    monthly["month_name"] = monthly["month"].apply(lambda m: MONTH_ORDER[m - 1])
    return monthly.sort_values("month")


def seasonal_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute seasonal averages.

    Seasons:
      Winter (Dec–Feb), Spring (Mar–May),
      Summer (Jun–Aug), Monsoon/Autumn (Sep–Nov)
    """
    cols = [c for c in POLLUTANT_COLS + ["aqi"] if c in df.columns]
    seasonal = df.groupby("season")[cols].mean().reset_index().round(2)

    # Sort by custom season order
    seasonal["season_order"] = seasonal["season"].apply(
        lambda s: SEASON_ORDER.index(s) if s in SEASON_ORDER else 99
    )
    seasonal = seasonal.sort_values("season_order").drop(columns="season_order")
    return seasonal


def worst_days(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return the top-N days with highest AQI.

    Returns columns: date, aqi, aqi_category, dominant_pollutant, pm25, pm10.
    """
    keep_cols = ["date", "aqi", "aqi_category", "dominant_pollutant"]
    for col in ["pm25", "pm10", "no2", "o3"]:
        if col in df.columns:
            keep_cols.append(col)

    keep_cols = [c for c in keep_cols if c in df.columns]
    worst = df.nlargest(n, "aqi")[keep_cols].copy()
    worst["date"] = worst["date"].dt.strftime("%Y-%m-%d")
    worst["aqi"] = worst["aqi"].round(1)
    return worst.reset_index(drop=True)


def pollutant_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of pollutants and AQI.
    """
    cols = [c for c in POLLUTANT_COLS + ["aqi"] if c in df.columns]
    return df[cols].corr().round(3)


def year_over_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute yearly averages for pollutants and AQI.
    """
    cols = [c for c in POLLUTANT_COLS + ["aqi"] if c in df.columns]
    yearly = df.groupby("year")[cols].mean().reset_index().round(2)
    return yearly


def dominant_pollutant_per_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which pollutant has the highest average sub-index per month.

    Returns a DataFrame with columns: month, month_name, dominant_pollutant, sub_index.
    """
    from analysis.preprocess import calculate_sub_index, AQI_BREAKPOINTS

    records = []
    for month_num, group in df.groupby("month"):
        best_pollutant = None
        best_si = -1
        for pollutant, bps in AQI_BREAKPOINTS.items():
            if pollutant not in group.columns:
                continue
            mean_val = group[pollutant].dropna().mean()
            if pd.isna(mean_val):
                continue
            si = calculate_sub_index(mean_val, bps)
            if not pd.isna(si) and si > best_si:
                best_si = si
                best_pollutant = pollutant
        records.append({
            "month": month_num,
            "month_name": MONTH_ORDER[month_num - 1],
            "dominant_pollutant": best_pollutant or "Unknown",
            "sub_index": round(best_si, 1),
        })

    return pd.DataFrame(records).sort_values("month")
