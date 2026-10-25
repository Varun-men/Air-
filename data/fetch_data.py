"""
OpenAQ v3 API data fetcher for Indian cities.
Fetches air quality measurements and saves to city-specific CSV files.
"""

import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://api.openaq.org/v3"
DATA_DIR = Path(__file__).parent

PARAMETERS = ["pm25", "pm10", "no2", "o3", "co", "so2"]

CITY_SEARCH_TERMS = {
    "Delhi": "Delhi",
    "Mumbai": "Mumbai",
    "Bengaluru": "Bengaluru",
    "Chennai": "Chennai",
    "Kolkata": "Kolkata",
    "Hyderabad": "Hyderabad",
    "Ahmedabad": "Ahmedabad",
    "Pune": "Pune",
    "Jaipur": "Jaipur",
    "Lucknow": "Lucknow",
}


def get_headers(api_key: str) -> dict:
    return {
        "X-API-Key": api_key,
        "Accept": "application/json",
    }


def fetch_city_locations(api_key: str, city: str) -> list:
    """Fetch all air quality monitoring locations for the given city."""
    search_term = CITY_SEARCH_TERMS.get(city, city)
    url = f"{BASE_URL}/locations"
    params = {
        "country_id": "IN",
        "city": search_term,
        "limit": 100,
    }
    try:
        resp = requests.get(url, headers=get_headers(api_key), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        locations = data.get("results", [])
        logger.info(f"Found {len(locations)} locations in {city}")
        return locations
    except Exception as e:
        logger.error(f"Failed to fetch {city} locations: {e}")
        return []


# Legacy alias kept for backward compatibility
def fetch_delhi_locations(api_key: str) -> list:
    return fetch_city_locations(api_key, "Delhi")


def fetch_measurements_for_location(
    api_key: str,
    location_id: int,
    date_from: str,
    date_to: str,
    limit: int = 1000,
) -> list:
    """Fetch measurements for a specific location and date range."""
    url = f"{BASE_URL}/locations/{location_id}/measurements"
    params = {
        "date_from": date_from,
        "date_to": date_to,
        "limit": limit,
    }
    try:
        resp = requests.get(url, headers=get_headers(api_key), params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        logger.warning(f"Failed to fetch measurements for location {location_id}: {e}")
        return []


def parse_measurements(measurements: list, location_name: str) -> list:
    """Parse raw measurement results into a flat list of records."""
    records = []
    for m in measurements:
        try:
            param = m.get("parameter", "").lower()
            if param not in PARAMETERS:
                continue
            value = m.get("value")
            if value is None or value < 0:
                continue
            date_obj = m.get("date", {})
            utc_str = date_obj.get("utc", "")
            if not utc_str:
                continue
            dt = pd.to_datetime(utc_str, utc=True)
            date_only = dt.date()
            records.append({
                "date": date_only,
                "location": location_name,
                "parameter": param,
                "value": float(value),
            })
        except Exception:
            continue
    return records


def fetch_city_data(api_key: str, city: str = "Delhi", days_back: int = 365) -> pd.DataFrame:
    """
    Fetch city AQI data from OpenAQ v3 API.
    Returns a DataFrame with columns: date, pm25, pm10, no2, o3, co, so2
    Falls back to sample data if the API call fails or returns no results.
    """
    from data.sample_data import generate_sample_data

    logger.info(f"Fetching {city} locations from OpenAQ v3...")
    locations = fetch_city_locations(api_key, city)

    if not locations:
        logger.warning(f"No locations found for {city} — falling back to sample data.")
        return generate_sample_data(city=city, days=days_back)

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    all_records = []

    for loc in locations[:10]:  # limit to first 10 locations to avoid rate limits
        loc_id = loc.get("id")
        loc_name = loc.get("name", f"Location_{loc_id}")

        # Chunk into 30-day windows to avoid hitting API limits
        chunk_start = start_date
        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=30), end_date)
            date_from = chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            date_to = chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ")

            measurements = fetch_measurements_for_location(
                api_key, loc_id, date_from, date_to
            )
            records = parse_measurements(measurements, loc_name)
            all_records.extend(records)

            chunk_start = chunk_end
            time.sleep(0.3)  # be polite to the API

    if not all_records:
        logger.warning(f"No measurements retrieved for {city} — falling back to sample data.")
        return generate_sample_data(city=city, days=days_back)

    raw_df = pd.DataFrame(all_records)

    # Pivot: average values per date & parameter across all stations
    pivot = (
        raw_df.groupby(["date", "parameter"])["value"]
        .mean()
        .unstack(level="parameter")
        .reset_index()
    )
    pivot.columns.name = None
    pivot["date"] = pd.to_datetime(pivot["date"])

    # Ensure all parameter columns exist
    for param in PARAMETERS:
        if param not in pivot.columns:
            pivot[param] = np.nan

    pivot = pivot[["date"] + PARAMETERS].sort_values("date").reset_index(drop=True)
    pivot["city"] = city

    # Save to city-specific CSV
    csv_path = DATA_DIR / f"{city.lower()}_aqi_data.csv"
    pivot.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(pivot)} rows to {csv_path}")

    return pivot


# Legacy alias kept for backward compatibility
def fetch_delhi_data(api_key: str, days_back: int = 365) -> pd.DataFrame:
    return fetch_city_data(api_key, city="Delhi", days_back=days_back)


def load_or_fetch_data(api_key: str, city: str = "Delhi") -> tuple:
    """
    Load from city-specific CSV if it exists and is recent (< 24h old),
    else fetch from OpenAQ API.
    Returns (DataFrame, source_label) where source_label is 'live', 'cached', or 'sample'.
    """
    from data.sample_data import generate_sample_data

    csv_path = DATA_DIR / f"{city.lower()}_aqi_data.csv"

    if csv_path.exists():
        age_hours = (time.time() - csv_path.stat().st_mtime) / 3600
        if age_hours < 24:
            logger.info(f"Loading cached data from {csv_path} (age: {age_hours:.1f}h)")
            try:
                df = pd.read_csv(csv_path, parse_dates=["date"])
                return df, "cached"
            except Exception as e:
                logger.warning(f"Failed to read cached CSV: {e}")

    logger.info(f"Fetching fresh data from OpenAQ API for {city}...")
    try:
        df = fetch_city_data(api_key, city=city, days_back=365)
        if df is not None and len(df) > 0:
            source = "live" if csv_path.exists() else "sample"
            return df, source
    except Exception as e:
        logger.error(f"API fetch failed for {city}: {e}")

    logger.warning(f"Falling back to sample data for {city}.")
    return generate_sample_data(city=city, days=365), "sample"
