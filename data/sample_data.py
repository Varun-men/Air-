"""
Fallback synthetic air quality data generator.
Produces realistic seasonal patterns based on city-specific historical pollution profiles.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# City profiles: (base_offset, amplitude) pairs for seasonal sine-wave logic
# ---------------------------------------------------------------------------
CITY_PROFILES = {
    "Delhi": {
        "pm25_base": (80, 170), "pm10_ratio": 1.9, "no2_base": (40, 60),
        "o3_base": (50, 80), "co_base": (1.0, 3.5), "so2_base": (12, 22),
        "temp_base": (27, 18), "humidity_base": (55, 35), "wind_base": (4, 5),
    },
    "Mumbai": {
        "pm25_base": (40, 80), "pm10_ratio": 1.7, "no2_base": (35, 50),
        "o3_base": (45, 70), "co_base": (0.8, 2.0), "so2_base": (8, 15),
        "temp_base": (29, 6), "humidity_base": (75, 20), "wind_base": (5, 6),
    },
    "Bengaluru": {
        "pm25_base": (30, 60), "pm10_ratio": 1.6, "no2_base": (25, 40),
        "o3_base": (40, 65), "co_base": (0.6, 1.5), "so2_base": (6, 12),
        "temp_base": (25, 5), "humidity_base": (65, 20), "wind_base": (4, 4),
    },
    "Chennai": {
        "pm25_base": (35, 70), "pm10_ratio": 1.8, "no2_base": (30, 45),
        "o3_base": (50, 75), "co_base": (0.7, 1.8), "so2_base": (7, 14),
        "temp_base": (30, 7), "humidity_base": (72, 20), "wind_base": (5, 5),
    },
    "Kolkata": {
        "pm25_base": (60, 130), "pm10_ratio": 1.85, "no2_base": (38, 55),
        "o3_base": (48, 72), "co_base": (0.9, 2.8), "so2_base": (10, 20),
        "temp_base": (28, 12), "humidity_base": (68, 25), "wind_base": (4, 4),
    },
    "Hyderabad": {
        "pm25_base": (38, 75), "pm10_ratio": 1.75, "no2_base": (28, 42),
        "o3_base": (48, 78), "co_base": (0.7, 1.6), "so2_base": (7, 13),
        "temp_base": (28, 9), "humidity_base": (60, 22), "wind_base": (4, 5),
    },
    "Ahmedabad": {
        "pm25_base": (55, 110), "pm10_ratio": 2.0, "no2_base": (35, 55),
        "o3_base": (55, 80), "co_base": (0.8, 2.2), "so2_base": (10, 18),
        "temp_base": (30, 15), "humidity_base": (55, 25), "wind_base": (5, 6),
    },
    "Pune": {
        "pm25_base": (32, 65), "pm10_ratio": 1.7, "no2_base": (26, 40),
        "o3_base": (44, 68), "co_base": (0.6, 1.4), "so2_base": (6, 11),
        "temp_base": (26, 8), "humidity_base": (62, 22), "wind_base": (4, 5),
    },
    "Jaipur": {
        "pm25_base": (65, 120), "pm10_ratio": 2.1, "no2_base": (35, 52),
        "o3_base": (55, 82), "co_base": (0.9, 2.5), "so2_base": (11, 19),
        "temp_base": (29, 17), "humidity_base": (48, 22), "wind_base": (5, 7),
    },
    "Lucknow": {
        "pm25_base": (70, 140), "pm10_ratio": 1.95, "no2_base": (38, 58),
        "o3_base": (50, 76), "co_base": (0.95, 3.0), "so2_base": (12, 21),
        "temp_base": (27, 16), "humidity_base": (62, 28), "wind_base": (4, 5),
    },
}

INDIAN_CITIES = list(CITY_PROFILES.keys())


def generate_sample_data(city: str = "Delhi", days: int = 730) -> pd.DataFrame:
    """
    Generate realistic synthetic air quality data with seasonal variation
    for the given Indian city.

    Parameters
    ----------
    city : str
        City name — must be a key in CITY_PROFILES.
    days : int
        Number of days of synthetic data to produce.

    Returns
    -------
    pd.DataFrame with columns:
        date, pm25, pm10, no2, o3, co, so2,
        temperature, humidity, wind_speed, city
    """
    if city not in CITY_PROFILES:
        raise ValueError(f"Unknown city '{city}'. Available: {INDIAN_CITIES}")

    profile = CITY_PROFILES[city]

    np.random.seed(42)

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    date_index = pd.DatetimeIndex(dates)

    # Day-of-year fraction for seasonal cycles (0 → 2π over the year)
    doy = date_index.dayofyear
    season_angle = 2 * np.pi * doy / 365.25

    # ------------------------------------------------------------------ #
    # Seasonal modulation helpers
    # Winter peak (Dec–Feb): cos(angle) is highest near day 1 (Jan)
    # ------------------------------------------------------------------ #
    winter_factor = (np.cos(season_angle) + 1) / 2           # 0–1, peaks in Jan
    summer_factor = (np.sin(season_angle - np.pi / 2) + 1) / 2   # peaks in Jul

    # Monsoon dampening: reduces PM in Jul-Sep
    monsoon_mask = (date_index.month >= 7) & (date_index.month <= 9)
    monsoon_damp = np.where(monsoon_mask, 0.45, 1.0)

    def add_noise(arr, scale=0.12):
        """Add proportional Gaussian noise."""
        noise = np.random.normal(1.0, scale, size=len(arr))
        return np.clip(arr * noise, 0, None)

    # ------------------------------------------------------------------ #
    # PM2.5
    # ------------------------------------------------------------------ #
    pm25_off, pm25_amp = profile["pm25_base"]
    pm25_base = pm25_off + pm25_amp * winter_factor
    pm25 = add_noise(pm25_base * monsoon_damp, scale=0.18)
    pm25 = np.clip(pm25, 5, 450)

    # PM10: ratio * PM2.5 + winter dust
    pm10_base = pm25 * profile["pm10_ratio"] + 20 * winter_factor
    pm10 = add_noise(pm10_base, scale=0.15)
    pm10 = np.clip(pm10, 10, 700)

    # NO2
    no2_off, no2_amp = profile["no2_base"]
    no2_base = no2_off + no2_amp * winter_factor
    no2 = add_noise(no2_base * monsoon_damp, scale=0.20)
    no2 = np.clip(no2, 5, 180)

    # O3: photochemical, higher in summer
    o3_off, o3_amp = profile["o3_base"]
    o3_base = o3_off + o3_amp * summer_factor
    o3 = add_noise(o3_base, scale=0.20)
    o3 = np.clip(o3, 5, 200)

    # CO
    co_off, co_amp = profile["co_base"]
    co_base = co_off + co_amp * winter_factor
    co = add_noise(co_base * monsoon_damp, scale=0.20)
    co = np.clip(co, 0.1, 8.0)

    # SO2
    so2_off, so2_amp = profile["so2_base"]
    so2_base = so2_off + so2_amp * winter_factor
    so2 = add_noise(so2_base * monsoon_damp, scale=0.20)
    so2 = np.clip(so2, 1, 80)

    # ------------------------------------------------------------------ #
    # Meteorological variables
    # ------------------------------------------------------------------ #
    temp_off, temp_amp = profile["temp_base"]
    temp_base = temp_off + temp_amp * np.sin(season_angle - np.pi / 2)
    temperature = add_noise(temp_base, scale=0.06)
    temperature = np.clip(temperature, 2, 50)

    hum_off, hum_amp = profile["humidity_base"]
    humidity_base = hum_off + hum_amp * summer_factor
    humidity = add_noise(humidity_base, scale=0.10)
    humidity = np.clip(humidity, 15, 98)

    wind_off, wind_amp = profile["wind_base"]
    wind_base = wind_off + wind_amp * summer_factor
    wind_speed = add_noise(wind_base, scale=0.25)
    wind_speed = np.clip(wind_speed, 0.5, 25)

    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "pm25": np.round(pm25, 2),
        "pm10": np.round(pm10, 2),
        "no2": np.round(no2, 2),
        "o3": np.round(o3, 2),
        "co": np.round(co, 3),
        "so2": np.round(so2, 2),
        "temperature": np.round(temperature, 1),
        "humidity": np.round(humidity, 1),
        "wind_speed": np.round(wind_speed, 2),
    })

    df["city"] = city

    return df
