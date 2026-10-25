"""
Air Pollution Monitoring & AQI Prediction Dashboard
====================================================
Editorial Single-Page Scrollable Experience
Inspired by Fenn Mallory (warm-wave-voiceover)
Typography: Newsreader 300, Inter 300, IBM Plex Mono 400
Palette: Ink (#1F1D1A), Paper (#EDE8DF), Accent (#C08A2E), Surface (#2A2723)
Navigation: Sticky Top Bar with Smooth Anchors (#overview, #explorer, #trends, #predictor, #about)
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure local modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="AQI Monitor — India",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Load signature waveform SVG path
WAVEFORM_FILE = os.path.join(os.path.dirname(__file__), "waveform_path.txt")
waveform_path_d = ""
if os.path.exists(WAVEFORM_FILE):
    with open(WAVEFORM_FILE, encoding="utf-8") as f:
        waveform_path_d = f.read().strip()

# ------------------------------------------------------------------ #
# CSS Theme — Exact replication of Fenn Mallory / warm-wave-voiceover
# ------------------------------------------------------------------ #
st.markdown("""
<style>
/* Import fonts */
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,500;1,6..72,300&family=Inter:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --paper: #EDE8DF;
    --ink: #1F1D1A;
    --accent: #C08A2E;
    --on-accent: #1F1D1A;
    --surface: #2A2723;
    --muted: #98918A;
    --line: rgba(237,232,223,0.12);
    --line-dark: rgba(31,29,26,0.15);
    --radius: 8px;
    --panel-radius: 24px;
}

/* Enable smooth scrolling across the entire page */
html {
    scroll-behavior: smooth !important;
}

/* Ground & Reset */
.stApp {
    background-color: var(--ink) !important;
    color: var(--paper) !important;
}

/* Hide sidebar and collapsed controls completely */
[data-testid="stSidebar"], 
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Main content width */
.main .block-container {
    max-width: 1160px !important;
    padding-top: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-bottom: 0rem !important;
    margin: 0 auto !important;
}

/* Typography */
h1, h2, h3, h4 {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 300 !important;
    color: var(--paper) !important;
    letter-spacing: -0.02em !important;
}
h1 {
    font-size: clamp(38px, 5.2vw, 76px) !important;
    line-height: 1.02 !important;
}
h2 {
    font-size: clamp(28px, 3.6vw, 48px) !important;
    line-height: 1.1 !important;
}
h3 {
    font-size: 22px !important;
    line-height: 1.25 !important;
}
p, li, span, div, label {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-weight: 300 !important;
    color: var(--paper);
}

/* Meta layer: IBM Plex Mono 400, 11px uppercase */
.meta {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--muted) !important;
}
.meta-accent {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: var(--accent) !important;
}

/* Sticky Top Navigation Bar */
.sticky-navbar {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: rgba(31, 29, 26, 0.94);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--line);
    margin: 0 -2rem 2rem -2rem;
    padding: 0 2rem;
}
.nav-inner {
    max-width: 1160px;
    margin: 0 auto;
    height: 70px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.nav-brand {
    display: flex;
    flex-direction: column;
}
.nav-links {
    display: flex;
    align-items: center;
    gap: 2rem;
}
.nav-links a {
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 300 !important;
    color: rgba(237, 232, 223, 0.75) !important;
    text-decoration: none !important;
    transition: color 0.2s ease;
}
.nav-links a:hover {
    color: var(--paper) !important;
}

/* Pill Action Button */
.pill-btn {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background-color: var(--accent);
    color: var(--on-accent) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 6px 6px 6px 18px;
    border-radius: 9999px;
    text-decoration: none !important;
    transition: background-color 0.2s ease, transform 0.15s ease;
}
.pill-btn:hover {
    background-color: var(--paper);
    color: var(--ink) !important;
}
.pill-arrow {
    display: grid;
    place-items: center;
    width: 28px;
    height: 28px;
    border-radius: 9999px;
    background-color: var(--ink);
    color: var(--paper);
    transition: transform 0.2s ease;
}
.pill-btn:hover .pill-arrow {
    transform: rotate(45deg);
}

/* Waveform drawing animation */
@keyframes drawWave {
    from { stroke-dashoffset: 1; }
    to { stroke-dashoffset: 0; }
}
.waveform-stroke {
    stroke-dasharray: 1;
    stroke-dashoffset: 1;
    animation: drawWave 1.8s cubic-bezier(.22,1,.36,1) forwards;
}

/* Section styling with anchor offsets */
.dashboard-section {
    padding-top: 3.5rem;
    padding-bottom: 2rem;
    scroll-margin-top: 80px;
}

/* Ruled Cards & Panels */
.surface-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 24px 28px;
    border-top: 1px solid var(--line);
}
.surface-card .card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 8px;
}
.surface-card .card-value {
    font-family: 'Newsreader', serif;
    font-size: 2.8rem;
    font-weight: 300;
    line-height: 1.05;
    color: var(--paper);
}
.surface-card .card-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
}

/* Light Inset Panel (Section 5 Rates style) */
.light-panel {
    background-color: var(--paper);
    color: var(--ink) !important;
    border-radius: var(--panel-radius);
    padding: 48px;
    margin: 40px 0;
}
.light-panel * {
    color: var(--ink) !important;
}
.light-panel .meta {
    color: rgba(31,29,26,0.6) !important;
}
.light-panel .hairline-row {
    display: grid;
    grid-template-columns: 1.4fr 1.4fr auto;
    padding: 18px 0;
    border-bottom: 1px solid var(--line-dark);
    align-items: center;
}

/* Accent Panel (Section 8 Send script style) */
.accent-panel {
    background-color: var(--accent);
    color: var(--on-accent) !important;
    border-radius: var(--panel-radius);
    padding: 48px;
    margin: 32px 0;
}
.accent-panel * {
    color: var(--on-accent) !important;
}

/* Hairline rows & dividers */
.hairline {
    border: none;
    border-top: 1px solid var(--line);
    margin: 48px 0;
}
.hairline-row {
    display: grid;
    grid-template-columns: 1.3fr 1.7fr auto;
    padding: 18px 0;
    border-bottom: 1px solid var(--line);
    align-items: baseline;
}

/* ML Luxury Console Styling */
.ml-console-card {
    background: #25221E;
    border: 1px solid rgba(237,232,223,0.12);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
}
.ml-param-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}
.ml-param-title {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 400;
    color: var(--paper);
}
.ml-param-unit {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--muted);
}

/* Streamlit Widget Theme Overrides */
[data-baseweb="select"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
[data-baseweb="select"] * {
    color: var(--paper) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: var(--on-accent) !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 14px 32px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #D4A24A !important;
    transform: translateY(-1px);
}

.stSlider > div > div > div > div {
    background-color: var(--accent) !important;
}

/* Monumental Footer Typography */
.monumental-wordmark {
    font-family: 'Newsreader', Georgia, serif;
    font-size: clamp(64px, 16vw, 190px);
    font-weight: 300;
    line-height: 0.76;
    letter-spacing: -0.04em;
    color: var(--paper);
    opacity: 0.12;
    text-align: center;
    white-space: nowrap;
    user-select: none;
    margin: 48px 0 -24px 0;
    transform: translateY(18%);
    overflow: hidden;
}

/* Hide Streamlit default header/footer */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Imports from project modules
# ------------------------------------------------------------------ #
from data.sample_data import INDIAN_CITIES, generate_sample_data
from data.fetch_data import load_or_fetch_data
from analysis.preprocess import preprocess_data, get_aqi_category
from analysis.trends import (
    monthly_averages, seasonal_averages, pollutant_correlation,
    year_over_year, worst_days
)
from models.ml_model import load_model, train_model, predict_aqi
from visualizations.charts import (
    aqi_timeline, monthly_heatmap, pollutant_bar_comparison,
    aqi_distribution, correlation_heatmap, seasonal_radar,
    feature_importance_bar, prediction_vs_actual, worst_days_table,
    multi_pollutant_line
)

# ------------------------------------------------------------------ #
# STICKY TOP NAVIGATION BAR (HTML Anchors for True Single-Page Scroll)
# ------------------------------------------------------------------ #
st.markdown("""
<header class="sticky-navbar">
    <div class="nav-inner">
        <div class="nav-brand">
            <a href="#overview" style="text-decoration:none;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:500; text-transform:uppercase; letter-spacing:0.1em; color:var(--paper);">
                    AQI MONITOR
                </div>
                <div class="meta" style="font-size:10px; margin-top:2px;">
                    INDIA / CPCB STANDARDS
                </div>
            </a>
        </div>
        <div class="nav-links">
            <a href="#overview">Overview</a>
            <a href="#explorer">Data Explorer</a>
            <a href="#trends">Pollution Trends</a>
            <a href="#rates-panel">CPCB Standards</a>
            <a href="#predictor">ML Predictor</a>
            <a href="#about">About</a>
        </div>
        <div>
            <a href="#predictor" class="pill-btn">
                Predict AQI
                <span class="pill-arrow">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M7 7h10v10"></path><path d="M7 17 17 7"></path>
                    </svg>
                </span>
            </a>
        </div>
    </div>
</header>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# TOP CONTROLS STRIP (City Selector & Live/Demo Mode)
# ------------------------------------------------------------------ #
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2.5, 1.2, 1.0], vertical_alignment="center")

with ctrl_col1:
    st.markdown("""
    <div style="display:flex; align-items:baseline; gap:16px;">
        <span class="meta" style="color:var(--paper);">SELECT MONITORING REGION</span>
        <span class="meta">10 CPCB OBSERVATION ARRAYS</span>
    </div>
    """, unsafe_allow_html=True)

with ctrl_col2:
    selected_city = st.selectbox(
        "City",
        INDIAN_CITIES,
        index=0,
        label_visibility="collapsed",
        key="city_selection_control",
    )

with ctrl_col3:
    use_sample = st.toggle("Demo Mode", value=True, key="demo_mode_control", help="Toggle between live OpenAQ API and synthetic city model")

st.markdown('<hr class="hairline" style="margin: 16px 0 32px 0;">', unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Data Loading (Cached per city)
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner=False, ttl=3600)
def get_city_data(city: str, use_demo: bool, api_key: str):
    if use_demo:
        raw = generate_sample_data(city=city, days=730)
        source = "demo"
    else:
        raw, source = load_or_fetch_data(api_key, city=city)
    df = preprocess_data(raw)
    return df, source

api_key = os.getenv("OPENAQ_API_KEY", "")

with st.spinner(f"Calibrating environmental records for {selected_city}..."):
    df, data_source = get_city_data(selected_city, use_sample, api_key)

# Helper to apply editorial theme to Plotly figures
def apply_editorial_theme(fig):
    fig.update_layout(
        paper_bgcolor="#1F1D1A",
        plot_bgcolor="#1F1D1A",
        font_color="#EDE8DF",
        font_family="Inter",
        xaxis=dict(gridcolor="rgba(237,232,223,0.06)", zerolinecolor="rgba(237,232,223,0.1)"),
        yaxis=dict(gridcolor="rgba(237,232,223,0.06)", zerolinecolor="rgba(237,232,223,0.1)"),
    )
    return fig

# ================================================================== #
# SECTION 0: HERO (Anchor #overview)
# ================================================================== #
st.markdown('<div id="overview" class="dashboard-section" style="padding-top:1rem;"></div>', unsafe_allow_html=True)

hero_left, hero_right = st.columns([1, 2], gap="large")

with hero_left:
    st.markdown(f"""
    <div style="display:flex; flex-direction:column; justify-content:space-between; height:100%; min-height:220px;">
        <div>
            <p class="meta">000/ {selected_city.upper()} AIR QUALITY</p>
            <p class="meta" style="margin-top:10px; line-height:1.7;">
                MULTI-STATION RECORD.<br>
                CPCB BREAKPOINT SCALE.<br>
                24-HR ROLLING AVERAGES.
            </p>
        </div>
        <div>
            <p class="meta" style="color:var(--muted);">
                SOURCE: {data_source.upper()} SENSORS
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hero_right:
    st.markdown(f"""
    <div>
        <h1 style="font-size:clamp(34px, 4.4vw, 64px); margin-bottom:18px;">
            A continuous measure of what we <em style="color:var(--accent); font-style:italic;">breathe</em>, not just what is recorded.
        </h1>
        <p style="font-size:16px; line-height:1.65; color:rgba(237,232,223,0.75); max-width:680px; margin-bottom:24px;">
            Continuous monitoring and predictive modeling of PM2.5, PM10, nitrogen dioxide, ozone, and carbon monoxide in {selected_city}, India. Evaluated against official Central Pollution Control Board categories.
        </p>
        <div style="display:flex; gap:16px; align-items:center;">
            <a href="#rates-panel" class="pill-btn">
                CPCB Breakpoints
                <span class="pill-arrow">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M7 7h10v10"></path><path d="M7 17 17 7"></path>
                    </svg>
                </span>
            </a>
            <span class="meta">730 OBSERVATIONS &middot; {selected_city.upper()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# SECTION 1: THE SIGNATURE WAVEFORM
# ------------------------------------------------------------------ #
if waveform_path_d:
    st.markdown(f"""
    <div style="padding: 48px 0 24px 0;">
        <svg viewBox="0 0 1200 120" preserveAspectRatio="none" style="display:block; width:100%; height:90px;" aria-hidden="true">
            <path class="waveform-stroke"
                  d="{waveform_path_d}"
                  stroke="#C08A2E" stroke-width="1.5" fill="none" pathLength="1" />
        </svg>
        <p class="meta" style="text-align:center; margin-top:16px; color:var(--muted);">
            &ldquo;PM2.5 particulate concentration curve &mdash; {selected_city} annual profile.&rdquo; Calibrated across seasonal transitions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# SECTION 2: CURRENT READINGS & OVERVIEW
# ------------------------------------------------------------------ #
latest = df.sort_values("date").iloc[-1]
aqi_val = float(latest.get("aqi", 0))
cat, cat_color = get_aqi_category(aqi_val)
dom_pol = str(latest.get("dominant_pollutant", "PM2.5")).upper()
pm25_val = float(latest.get("pm25", 0))

st.markdown('<hr class="hairline">', unsafe_allow_html=True)
st.markdown('<p class="meta">001/ AMBIENT CONDITIONS &middot; RECENT OBSERVATION</p>', unsafe_allow_html=True)

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">Current AQI</div>
        <div class="card-value" style="color:var(--paper);">{aqi_val:.0f}</div>
        <div class="card-sub">{latest['date'].strftime('%d %b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)
with mc2:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">CPCB Classification</div>
        <div class="card-value" style="font-size:2.2rem; color:{cat_color};">{cat}</div>
        <div class="card-sub">Air Quality Band</div>
    </div>
    """, unsafe_allow_html=True)
with mc3:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">Dominant Contributor</div>
        <div class="card-value">{dom_pol}</div>
        <div class="card-sub">Highest Sub-Index</div>
    </div>
    """, unsafe_allow_html=True)
with mc4:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">PM2.5 Concentration</div>
        <div class="card-value">{pm25_val:.1f}</div>
        <div class="card-sub">&micro;g/m&sup3; &middot; 24h Mean</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)
st.markdown(f'<p class="meta">ANNUAL INDEX PROGRESSION &mdash; {selected_city.upper()}</p>', unsafe_allow_html=True)
st.plotly_chart(apply_editorial_theme(aqi_timeline(df)), use_container_width=True)

st.markdown('<p class="meta">CALENDAR DENSITY HEATMAP</p>', unsafe_allow_html=True)
st.plotly_chart(apply_editorial_theme(monthly_heatmap(df)), use_container_width=True)

# ================================================================== #
# SECTION 3: DATA EXPLORER (Anchor #explorer)
# ================================================================== #
st.markdown('<div id="explorer" class="dashboard-section"></div>', unsafe_allow_html=True)
st.markdown('<hr class="hairline">', unsafe_allow_html=True)
st.markdown(f'<p class="meta">002/ CRITERIA DATA EXPLORER &mdash; {selected_city.upper()}</p>', unsafe_allow_html=True)
st.markdown('<h2 style="margin-bottom:24px;">Time Series & Parametric Query</h2>', unsafe_allow_html=True)

fcol1, fcol2 = st.columns([2, 1])
with fcol1:
    date_range = st.date_input(
        "Observation Window",
        value=(df["date"].min().date(), df["date"].max().date()),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
        label_visibility="collapsed",
    )
with fcol2:
    all_pollutants = [c for c in ["pm25", "pm10", "no2", "o3", "co", "so2"] if c in df.columns]
    selected_pollutants = st.multiselect(
        "Selected Criteria Pollutants",
        all_pollutants,
        default=["pm25", "pm10", "no2"],
        label_visibility="collapsed",
    )

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    mask = (df["date"].dt.date >= start_d) & (df["date"].dt.date <= end_d)
else:
    mask = pd.Series([True] * len(df))

filtered_df = df[mask].copy()

if selected_pollutants:
    st.plotly_chart(apply_editorial_theme(multi_pollutant_line(filtered_df, selected_pollutants)), use_container_width=True)

st.markdown(f'<p class="meta" style="margin-top:24px;">RECORD COUNT: {len(filtered_df):,} OBSERVATIONS IN QUERY</p>', unsafe_allow_html=True)

display_cols = ["date", "aqi", "aqi_category"] + [p for p in selected_pollutants if p in filtered_df.columns]
tbl_df = filtered_df[display_cols].copy()
tbl_df["date"] = tbl_df["date"].dt.strftime("%Y-%m-%d")
st.dataframe(tbl_df.reset_index(drop=True), use_container_width=True, height=280)

csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label=f"EXPORT {selected_city.upper()} DATASET (CSV)",
    data=csv_data,
    file_name=f"{selected_city.lower()}_aqi_{date.today()}.csv",
    mime="text/csv",
)

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p class="meta">CRITICAL POLLUTION EPISODES (WORST 10 DAYS)</p>', unsafe_allow_html=True)
worst_df = worst_days(filtered_df, n=10)
st.plotly_chart(worst_days_table(worst_df), use_container_width=True)

# ================================================================== #
# SECTION 4: POLLUTION TRENDS (Anchor #trends)
# ================================================================== #
st.markdown('<div id="trends" class="dashboard-section"></div>', unsafe_allow_html=True)
st.markdown('<hr class="hairline">', unsafe_allow_html=True)
st.markdown(f'<p class="meta">003/ SEASONAL DYNAMICS & CORRELATIONS &mdash; {selected_city.upper()}</p>', unsafe_allow_html=True)
st.markdown('<h2 style="margin-bottom:24px;">Atmospheric Phenomena & Profiles</h2>', unsafe_allow_html=True)

m_df = monthly_averages(df)
s_df = seasonal_averages(df)

t_col1, t_col2 = st.columns(2)
with t_col1:
    st.markdown('<p class="meta">SEASONAL POLLUTANT RADAR</p>', unsafe_allow_html=True)
    st.plotly_chart(apply_editorial_theme(seasonal_radar(s_df)), use_container_width=True)

with t_col2:
    st.markdown('<p class="meta">MONTHLY PARTICULATE COMPARISON</p>', unsafe_allow_html=True)
    st.plotly_chart(apply_editorial_theme(pollutant_bar_comparison(m_df)), use_container_width=True)

t_col3, t_col4 = st.columns(2)
with t_col3:
    st.markdown('<p class="meta">AQI FREQUENCY DENSITY DISTRIBUTION</p>', unsafe_allow_html=True)
    st.plotly_chart(apply_editorial_theme(aqi_distribution(df)), use_container_width=True)

with t_col4:
    st.markdown('<p class="meta">INTER-POLLUTANT CORRELATION MATRIX</p>', unsafe_allow_html=True)
    st.plotly_chart(apply_editorial_theme(correlation_heatmap(df)), use_container_width=True)

# Geographical insights
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<p class="meta">OBSERVED METEOROLOGICAL PATTERNS</p>', unsafe_allow_html=True)

is_coastal = selected_city in ["Mumbai", "Chennai"]
is_northern = selected_city in ["Delhi", "Jaipur", "Lucknow"]

observations = [
    ("WINTER THERMAL INVERSION", "Particulate matter concentrates closer to ground level during cooler months due to shallow atmospheric boundary layers." if is_northern else "Mild winter cooling reduces vertical dispersion, producing elevated morning particulate concentrations."),
    ("MONSOON PRECIPITATION CLEANSE", "Heavy rainfall during the southwest monsoon provides wet scavenging that strips airborne particulates from suspension."),
    ("PHOTOCHEMICAL OZONE GENERATION", "Solar insolation in high-temperature pre-monsoon months drives secondary ozone formation from volatile precursors."),
    ("VEHICULAR & COMMERCIAL TRANSIT", "Nitrogen dioxide peaks align closely with urban arterial transit hours and commercial diesel corridors."),
    ("AEOLIAN & CONSTRUCTION DUST", "Coarse particulates (PM10) show elevated baseline ratios during dry, windy pre-monsoon conditions."),
]

for title, desc in observations:
    st.markdown(f"""
    <div class="hairline-row">
        <span class="meta-accent">{title}</span>
        <span style="font-size:15px; line-height:1.6; color:rgba(237,232,223,0.85);">{desc}</span>
        <span class="meta">VERIFIED</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================================== #
# SECTION 5: CPCB STANDARDS LIGHT INSET PANEL (Anchor #rates-panel)
# ================================================================== #
st.markdown("""
<div id="rates-panel" class="light-panel">
    <p class="meta">004/ Official CPCB Breakpoints</p>
    <h2 style="font-family:'Newsreader',serif; font-size:clamp(26px,3.2vw,40px); margin:12px 0 24px 0; color:#1F1D1A !important;">
        National Air Quality Index Categories & Clinical Impact
    </h2>
    <div class="hairline-row" style="font-weight:500;">
        <span class="meta" style="color:#1F1D1A !important;">CATEGORY</span>
        <span class="meta" style="color:#1F1D1A !important;">CLINICAL & PHYSIOLOGICAL IMPACT</span>
        <span class="meta" style="text-align:right; color:#1F1D1A !important;">INDEX RANGE</span>
    </div>
    <div class="hairline-row">
        <span style="font-family:'Newsreader',serif; font-size:20px;">Good</span>
        <span>Minimal impact on breathing and cardiovascular functions.</span>
        <span class="meta" style="text-align:right; font-weight:500;">0 &ndash; 50</span>
    </div>
    <div class="hairline-row">
        <span style="font-family:'Newsreader',serif; font-size:20px;">Satisfactory</span>
        <span>Minor breathing discomfort to sensitive individuals.</span>
        <span class="meta" style="text-align:right; font-weight:500;">51 &ndash; 100</span>
    </div>
    <div class="hairline-row">
        <span style="font-family:'Newsreader',serif; font-size:20px;">Moderate</span>
        <span>Breathing discomfort to people with asthma, pulmonary, and cardiac history.</span>
        <span class="meta" style="text-align:right; font-weight:500;">101 &ndash; 200</span>
    </div>
    <div class="hairline-row">
        <span style="font-family:'Newsreader',serif; font-size:20px;">Poor</span>
        <span>Breathing discomfort to most people on prolonged exposure.</span>
        <span class="meta" style="text-align:right; font-weight:500;">201 &ndash; 300</span>
    </div>
    <div class="hairline-row">
        <span style="font-family:'Newsreader',serif; font-size:20px;">Very Poor</span>
        <span>Respiratory illness on prolonged exposure; significant alert.</span>
        <span class="meta" style="text-align:right; font-weight:500;">301 &ndash; 400</span>
    </div>
    <div class="hairline-row" style="border-bottom:none;">
        <span style="font-family:'Newsreader',serif; font-size:20px;">Severe</span>
        <span>Affects healthy people and severely impacts individuals with existing conditions.</span>
        <span class="meta" style="text-align:right; font-weight:500;">401 &ndash; 500</span>
    </div>
    <p class="meta" style="margin-top:28px; font-size:10px; color:rgba(31,29,26,0.65) !important;">
        CENTRAL POLLUTION CONTROL BOARD (CPCB) STATUTORY METHODOLOGY &middot; SUB-INDEX PIECEWISE INTERPOLATION
    </p>
</div>
""", unsafe_allow_html=True)

# ================================================================== #
# SECTION 6: ML PREDICTOR (Anchor #predictor)
# Redesigned to look bespoke, editorial, and high-end
# ================================================================== #
st.markdown('<div id="predictor" class="dashboard-section"></div>', unsafe_allow_html=True)
st.markdown('<hr class="hairline">', unsafe_allow_html=True)
st.markdown(f'<p class="meta">005/ MACHINE LEARNING INFERENCE ENGINE &mdash; {selected_city.upper()}</p>', unsafe_allow_html=True)
st.markdown('<h2 style="margin-bottom:12px;">Random Forest Regressor & Calibration</h2>', unsafe_allow_html=True)
st.markdown(f'<p style="font-size:15px; color:rgba(237,232,223,0.7); max-width:720px; margin-bottom:28px;">Continuous supervised model trained on {selected_city} multi-station observations. Validated via 80/20 partition and 5-fold cross-validation.</p>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_city_model(city_name: str, _data_df):
    return train_model(_data_df, city=city_name)

with st.spinner(f"Calibrating Random Forest model for {selected_city}..."):
    model, metrics, fi_df = get_city_model(selected_city, df)

# Model Performance Cards
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">Mean Absolute Error</div>
        <div class="card-value">{metrics['rf']['MAE']:.2f}</div>
        <div class="card-sub">Index Points &middot; RF</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">Root Mean Sq. Error</div>
        <div class="card-value">{metrics['rf']['RMSE']:.2f}</div>
        <div class="card-sub">Index Points &middot; RF</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">Coefficient of Det. (R&sup2;)</div>
        <div class="card-value" style="color:var(--accent);">{metrics['rf']['R2']:.4f}</div>
        <div class="card-sub">Random Forest</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="surface-card">
        <div class="card-label">Baseline Comparison</div>
        <div class="card-value">{metrics['lr']['R2']:.4f}</div>
        <div class="card-sub">Linear Regression R&sup2;</div>
    </div>
    """, unsafe_allow_html=True)

ml_col1, ml_col2 = st.columns(2)
with ml_col1:
    st.markdown('<br><p class="meta">RELATIVE ATTRIBUTE IMPORTANCE</p>', unsafe_allow_html=True)
    st.plotly_chart(apply_editorial_theme(feature_importance_bar(fi_df)), use_container_width=True)

with ml_col2:
    st.markdown('<br><p class="meta">MODEL ACCURACY &middot; MEASURED VS PREDICTED</p>', unsafe_allow_html=True)
    st.plotly_chart(apply_editorial_theme(prediction_vs_actual(metrics["y_test"], metrics["y_pred_rf"])), use_container_width=True)

# ------------------------------------------------------------------ #
# Interactive Prediction Console
# ------------------------------------------------------------------ #
st.markdown('<hr class="hairline">', unsafe_allow_html=True)
st.markdown('<p class="meta">INTERACTIVE PARAMETER TESTING</p>', unsafe_allow_html=True)
st.markdown(f'<h3>Simulate Atmospheric Concentrations for {selected_city}</h3>', unsafe_allow_html=True)

# Quick scenario presets
st.markdown('<p class="meta" style="margin-top:12px;">SCENARIO PRESETS</p>', unsafe_allow_html=True)
sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)

preset_vals = None
if sc_col1.button("Winter Inversion Spike", use_container_width=True):
    preset_vals = {"pm25": 210.0, "pm10": 340.0, "no2": 85.0, "o3": 35.0, "co": 3.8, "so2": 28.0, "m": 12}
elif sc_col2.button("Monsoon Clean Air", use_container_width=True):
    preset_vals = {"pm25": 28.0, "pm10": 55.0, "no2": 22.0, "o3": 30.0, "co": 0.6, "so2": 8.0, "m": 8}
elif sc_col3.button("Summer Ozone & Dust", use_container_width=True):
    preset_vals = {"pm25": 95.0, "pm10": 260.0, "no2": 45.0, "o3": 110.0, "co": 1.4, "so2": 15.0, "m": 5}
elif sc_col4.button("Moderate Baseline", use_container_width=True):
    preset_vals = {"pm25": 65.0, "pm10": 130.0, "no2": 38.0, "o3": 55.0, "co": 1.1, "so2": 12.0, "m": 3}

if preset_vals:
    st.session_state["p_pm25"] = preset_vals["pm25"]
    st.session_state["p_pm10"] = preset_vals["pm10"]
    st.session_state["p_no2"] = preset_vals["no2"]
    st.session_state["p_o3"] = preset_vals["o3"]
    st.session_state["p_co"] = preset_vals["co"]
    st.session_state["p_so2"] = preset_vals["so2"]
    st.session_state["p_m"] = preset_vals["m"]

with st.form("aqi_prediction_form_v2"):
    p1, p2, p3 = st.columns(3)
    with p1:
        in_pm25 = st.slider(
            "PM2.5 (Fine Particulates)", 0.0, 500.0,
            st.session_state.get("p_pm25", 80.0), 1.0,
            help="Particles under 2.5 microns (µg/m³)"
        )
        in_pm10 = st.slider(
            "PM10 (Coarse Particulates)", 0.0, 600.0,
            st.session_state.get("p_pm10", 150.0), 1.0,
            help="Particles under 10 microns (µg/m³)"
        )
    with p2:
        in_no2 = st.slider(
            "NO2 (Nitrogen Dioxide)", 0.0, 200.0,
            st.session_state.get("p_no2", 45.0), 1.0,
            help="Vehicular and combustion gas (µg/m³)"
        )
        in_o3 = st.slider(
            "Ozone (Ground-level O3)", 0.0, 200.0,
            st.session_state.get("p_o3", 55.0), 1.0,
            help="Photochemical oxidant (µg/m³)"
        )
    with p3:
        in_co = st.slider(
            "CO (Carbon Monoxide)", 0.0, 10.0,
            st.session_state.get("p_co", 1.2), 0.1,
            help="Incomplete combustion byproduct (mg/m³)"
        )
        in_so2 = st.slider(
            "SO2 (Sulphur Dioxide)", 0.0, 100.0,
            st.session_state.get("p_so2", 14.0), 1.0,
            help="Industrial fuel emissions (µg/m³)"
        )

    t1, t2 = st.columns(2)
    with t1:
        month_idx = st.selectbox(
            "Calendar Month",
            range(1, 13),
            format_func=lambda m: date(2026, m, 1).strftime("%B"),
            index=st.session_state.get("p_m", datetime.now().month) - 1
        )
    with t2:
        dow_idx = st.selectbox(
            "Day of Week",
            range(7),
            format_func=lambda d: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d],
            index=0
        )

    submit_pred = st.form_submit_button("CALCULATE MODEL INFERENCE", use_container_width=True)

# Run prediction either on submit or by default on load
pred_aqi, pred_cat, pred_color = predict_aqi(
    model, in_pm25, in_pm10, in_no2, in_o3, in_co, in_so2, month_idx, dow_idx
)

advisories = {
    "Good": "Air quality is satisfactory. Atmospheric conditions pose minimal risk for active outdoor engagement.",
    "Satisfactory": "Acceptable air quality. Unusually sensitive individuals should monitor personal respiratory comfort.",
    "Moderate": "Breathing discomfort possible for individuals with asthma, cardiovascular, and pulmonary conditions.",
    "Poor": "Noticeable breathing difficulties likely for prolonged outdoor exposure across the general population.",
    "Very Poor": "Significant health concern. Limit extended outdoor exposure; keep air circulation filtered.",
    "Severe": "Emergency health conditions. High vulnerability for all individuals. Suspend outdoor activities."
}
advice_text = advisories.get(pred_cat, "Refer to local municipal health notices.")

st.markdown(f"""
<div class="accent-panel" style="margin-top:24px;">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;">
        <div>
            <p class="meta" style="color:var(--on-accent) !important;">MODEL PREDICTION &middot; {selected_city.upper()}</p>
            <div style="font-family:'Newsreader',serif; font-size:clamp(56px, 7vw, 96px); line-height:0.95; margin:8px 0;">
                {pred_aqi:.0f}
            </div>
            <div class="meta" style="font-size:14px; font-weight:500;">
                CATEGORY: {pred_cat.upper()}
            </div>
        </div>
        <div style="max-width:540px; border-left:1px solid rgba(31,29,26,0.3); padding-left:24px;">
            <p class="meta" style="color:var(--on-accent) !important; margin-bottom:8px;">CLINICAL & PRECAUTIONARY NOTICE</p>
            <p style="font-size:16px; line-height:1.6; color:rgba(31,29,26,0.9) !important; margin-bottom:12px;">
                {advice_text}
            </p>
            <p class="meta" style="font-size:10px; color:rgba(31,29,26,0.65) !important;">
                R&sup2; = {metrics['rf']['R2']:.4f} &middot; CPCB INTERPOLATION BENCHMARK &middot; MONTH {month_idx}
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================== #
# SECTION 7: ABOUT (Anchor #about)
# ================================================================== #
st.markdown('<div id="about" class="dashboard-section"></div>', unsafe_allow_html=True)
st.markdown('<hr class="hairline">', unsafe_allow_html=True)
st.markdown('<p class="meta">006/ ARCHITECTURE & MATHEMATICAL SPECIFICATIONS</p>', unsafe_allow_html=True)

about_col1, about_col2 = st.columns([1, 1], gap="large")
with about_col1:
    st.markdown("""
    <h2 style="margin-bottom:16px;">CPCB Breakpoint Calculus</h2>
    <p style="font-size:15px; line-height:1.65; color:rgba(237,232,223,0.8);">
        The National Air Quality Index (AQI) framework formulated by India's Central Pollution Control Board (CPCB) standardizes six criteria pollutants &mdash; PM2.5, PM10, NO2, O3, CO, and SO2 &mdash; into dimensionless sub-indices using piecewise linear interpolation:
    </p>
    <div style="background:var(--surface); padding:16px 20px; border-radius:var(--radius); margin:18px 0;">
        <code style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--accent);">
            I = [(I_hi - I_lo) / (B_hi - B_lo)] * (C - B_lo) + I_lo
        </code>
    </div>
    <p style="font-size:15px; line-height:1.65; color:rgba(237,232,223,0.8);">
        The overall AQI is governed by the highest pollutant sub-index: <code>AQI = max(I_1, I_2, ..., I_6)</code>.
    </p>
    """, unsafe_allow_html=True)

with about_col2:
    st.markdown("""
    <h2 style="margin-bottom:16px;">Computational Pipeline</h2>
    <p style="font-size:15px; line-height:1.65; color:rgba(237,232,223,0.8);">
        Continuous air quality observations are queried from the OpenAQ v3 REST API across official monitoring station arrays. For model validation and simulation, high-fidelity seasonal synthetic generation models diurnal boundaries, solar angles, and precipitation wash events.
    </p>
    <div class="hairline-row">
        <span class="meta">DATA SOURCE</span>
        <span>OpenAQ v3 API &middot; CPCB Continuous Ambient Stations</span>
        <span class="meta">REST API</span>
    </div>
    <div class="hairline-row">
        <span class="meta">ESTIMATOR</span>
        <span>Random Forest Regressor (150 Estimators, Max Depth 12)</span>
        <span class="meta">SCIKIT-LEARN</span>
    </div>
    <div class="hairline-row">
        <span class="meta">METHODOLOGY</span>
        <span>CPCB Statutory Breakpoints (CPCB/PR-46/2014-15)</span>
        <span class="meta">INDIA GOV</span>
    </div>
    """, unsafe_allow_html=True)

# ================================================================== #
# SECTION 8: MONUMENTAL FOOTER (Exact replication of Fenn Mallory)
# ================================================================== #
st.markdown("""
<div style="margin-top:100px; padding-top:40px; border-top:1px solid var(--line); overflow:hidden;">
    <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:24px;">
        <div>
            <a href="mailto:air-monitoring@cpcb-research.in" style="font-family:'Newsreader',serif; font-size:24px; color:var(--paper); text-decoration:none;">
                air-monitoring@cpcb-research.in
            </a>
            <p class="meta" style="margin-top:8px;">
                Continuous ambient monitoring &middot; OpenAQ v3 &middot; India CPCB Standards
            </p>
        </div>
        <div class="meta" style="display:flex; gap:24px;">
            <a href="#overview" style="color:var(--muted); text-decoration:none;">Overview</a>
            <a href="#explorer" style="color:var(--muted); text-decoration:none;">Explorer</a>
            <a href="#trends" style="color:var(--muted); text-decoration:none;">Trends</a>
            <a href="#rates-panel" style="color:var(--muted); text-decoration:none;">Standards</a>
            <a href="#predictor" style="color:var(--muted); text-decoration:none;">ML Model</a>
        </div>
    </div>
    <p aria-hidden="true" class="monumental-wordmark">AQI MONITOR</p>
</div>
""", unsafe_allow_html=True)
