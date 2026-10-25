"""
All Plotly chart functions for the Air Pollution Monitoring dashboard.
All charts use the dark theme (template="plotly_dark").
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

DARK_TEMPLATE = "plotly_dark"

AQI_COLOR_MAP = {
    "Good":            "#00B050",
    "Satisfactory":    "#92D050",
    "Moderate":        "#FFFF00",
    "Poor":            "#FF7C00",
    "Very Poor":       "#FF0000",
    "Severe":          "#C00000",
}

POLLUTANT_COLORS = {
    "pm25": "#EF553B",
    "pm10": "#FF7F0E",
    "no2":  "#636EFA",
    "o3":   "#00CC96",
    "co":   "#AB63FA",
    "so2":  "#FFA15A",
}

POLLUTANT_LABELS = {
    "pm25": "PM2.5 (µg/m³)",
    "pm10": "PM10 (µg/m³)",
    "no2":  "NO₂ (µg/m³)",
    "o3":   "O₃ (µg/m³)",
    "co":   "CO (mg/m³)",
    "so2":  "SO₂ (µg/m³)",
}


def aqi_timeline(df: pd.DataFrame) -> go.Figure:
    """Line chart of AQI over time, colored segments by AQI category."""
    fig = go.Figure()

    # Background bands
    band_defs = [
        (0,   50,  "#00B050", "Good"),
        (50,  100, "#92D050", "Satisfactory"),
        (100, 200, "#FFFF00", "Moderate"),
        (200, 300, "#FF7C00", "Poor"),
        (300, 400, "#FF0000", "Very Poor"),
        (400, 500, "#C00000", "Severe"),
    ]
    for y0, y1, color, label in band_defs:
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=color,
            opacity=0.08,
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=10,
            annotation_font_color=color,
        )

    # Main AQI line
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["aqi"],
        mode="lines",
        name="AQI",
        line=dict(color="#00B4D8", width=1.5),
        hovertemplate=(
            "<b>%{x|%d %b %Y}</b><br>"
            "AQI: %{y:.0f}<br>"
            "<extra></extra>"
        ),
    ))

    # 7-day rolling average
    df_plot = df.copy()
    df_plot["aqi_7d"] = df_plot["aqi"].rolling(7, center=True).mean()
    fig.add_trace(go.Scatter(
        x=df_plot["date"],
        y=df_plot["aqi_7d"],
        mode="lines",
        name="7-day avg",
        line=dict(color="#FF9500", width=2.5, dash="dash"),
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="AQI Timeline — Delhi",
        xaxis_title="Date",
        yaxis_title="AQI",
        yaxis=dict(range=[0, 520]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=80, b=60),
        height=420,
    )
    return fig


def monthly_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap: x=month, y=year, color=mean AQI."""
    pivot = df.pivot_table(values="aqi", index="year", columns="month", aggfunc="mean")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=[str(y) for y in pivot.index.tolist()],
        colorscale=[
            [0.0,  "#00B050"],
            [0.1,  "#92D050"],
            [0.2,  "#FFFF00"],
            [0.5,  "#FF7C00"],
            [0.7,  "#FF0000"],
            [1.0,  "#C00000"],
        ],
        zmin=0, zmax=500,
        colorbar=dict(title="AQI"),
        hovertemplate="<b>%{y} %{x}</b><br>Avg AQI: %{z:.0f}<extra></extra>",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Monthly Average AQI Heatmap",
        xaxis_title="Month",
        yaxis_title="Year",
        margin=dict(l=60, r=60, t=80, b=60),
        height=320,
    )
    return fig


def pollutant_bar_comparison(monthly_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of monthly avg PM2.5, PM10, NO2."""
    fig = go.Figure()
    for pollutant in ["pm25", "pm10", "no2"]:
        if pollutant not in monthly_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=POLLUTANT_LABELS.get(pollutant, pollutant),
            x=monthly_df["month_name"],
            y=monthly_df[pollutant],
            marker_color=POLLUTANT_COLORS.get(pollutant, "#888"),
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Monthly Average Pollutant Levels",
        xaxis_title="Month",
        yaxis_title="Concentration (µg/m³)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=60),
        height=380,
    )
    return fig


def aqi_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of AQI distribution, colored by AQI category."""
    fig = go.Figure()

    bins = [0, 50, 100, 200, 300, 400, 500]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    colors = [AQI_COLOR_MAP[l] for l in labels]

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        subset = df[(df["aqi"] > lo) & (df["aqi"] <= hi)]
        if len(subset) == 0:
            continue
        fig.add_trace(go.Histogram(
            x=subset["aqi"],
            name=labels[i],
            marker_color=colors[i],
            xbins=dict(start=lo, end=hi, size=10),
            hovertemplate=f"<b>{labels[i]}</b><br>AQI: %{{x}}<br>Days: %{{y}}<extra></extra>",
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="AQI Distribution",
        xaxis_title="AQI Value",
        yaxis_title="Number of Days",
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=60),
        height=360,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap of pollutant correlations."""
    cols = [c for c in ["pm25", "pm10", "no2", "o3", "co", "so2", "aqi"] if c in df.columns]
    corr = df[cols].corr().round(3)
    labels = [POLLUTANT_LABELS.get(c, c).split(" ")[0] for c in cols]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        colorbar=dict(title="Correlation"),
        hovertemplate="<b>%{y} × %{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Pollutant Correlation Matrix",
        margin=dict(l=80, r=20, t=80, b=80),
        height=420,
    )
    return fig


def seasonal_radar(seasonal_df: pd.DataFrame) -> go.Figure:
    """Radar/polar chart comparing seasons across normalized pollutant values."""
    pollutants = [c for c in ["pm25", "pm10", "no2", "o3", "co", "so2"] if c in seasonal_df.columns]
    labels = [POLLUTANT_LABELS.get(p, p).split(" ")[0] for p in pollutants]

    # Normalize each pollutant to 0-100 scale for radar comparability
    norm_df = seasonal_df[pollutants].copy()
    for col in pollutants:
        col_max = norm_df[col].max()
        if col_max > 0:
            norm_df[col] = norm_df[col] / col_max * 100
        else:
            norm_df[col] = 0

    season_colors = {
        "Winter":         "#4CC9F0",
        "Spring":         "#4ADE80",
        "Summer":         "#FACC15",
        "Monsoon/Autumn": "#F97316",
    }

    fig = go.Figure()
    for _, row in seasonal_df.iterrows():
        season = row["season"]
        values = [norm_df.loc[_, p] for p in pollutants]
        values += [values[0]]  # close loop
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=season,
            opacity=0.75,
            line=dict(color=season_colors.get(season, "#AAAAAA")),
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Seasonal Pollutant Profile (Normalized)",
        polar=dict(radialaxis=dict(visible=True, range=[0, 110])),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=60, r=60, t=80, b=80),
        height=480,
    )
    return fig


def feature_importance_bar(feature_importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of Random Forest feature importances."""
    fi = feature_importance_df.sort_values("importance")
    labels = [POLLUTANT_LABELS.get(f, f.replace("_", " ").title()) for f in fi["feature"]]

    fig = go.Figure(go.Bar(
        x=fi["importance"],
        y=labels,
        orientation="h",
        marker=dict(
            color=fi["importance"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Importance"),
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Feature Importance — Random Forest",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        margin=dict(l=140, r=60, t=80, b=60),
        height=380,
    )
    return fig


def prediction_vs_actual(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """Scatter plot of actual vs predicted AQI with R² annotation."""
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, y_pred)
    max_val = max(y_true.max(), y_pred.max()) + 20

    fig = go.Figure()

    # Perfect prediction reference line
    fig.add_trace(go.Scatter(
        x=[0, max_val],
        y=[0, max_val],
        mode="lines",
        name="Perfect prediction",
        line=dict(color="#AAAAAA", dash="dash", width=1.5),
    ))

    # Scatter
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode="markers",
        name="Predictions",
        marker=dict(
            color=y_true,
            colorscale="RdYlGn_r",
            size=5,
            opacity=0.7,
            colorbar=dict(title="Actual AQI"),
        ),
        hovertemplate="Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>",
    ))

    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"<b>R² = {r2:.4f}</b>",
        showarrow=False,
        font=dict(size=14, color="#FFFFFF"),
        bgcolor="rgba(0,0,0,0.5)",
        borderpad=6,
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Actual vs Predicted AQI",
        xaxis_title="Actual AQI",
        yaxis_title="Predicted AQI",
        margin=dict(l=60, r=60, t=80, b=60),
        height=420,
    )
    return fig


def worst_days_table(worst_df: pd.DataFrame) -> go.Figure:
    """Styled Plotly table of worst AQI days."""
    col_labels = {
        "date":               "Date",
        "aqi":                "AQI",
        "aqi_category":       "Category",
        "dominant_pollutant": "Dominant Pollutant",
        "pm25":               "PM2.5",
        "pm10":               "PM10",
        "no2":                "NO₂",
        "o3":                 "O₃",
    }
    display_cols = [c for c in col_labels if c in worst_df.columns]
    headers = [col_labels[c] for c in display_cols]
    values = [worst_df[c].tolist() for c in display_cols]

    # Row colors based on AQI category
    from analysis.preprocess import get_aqi_category
    def _hex_to_rgba(h: str, alpha: float = 0.35) -> str:
        h = h.lstrip('#')
        if len(h) == 6:
            r, g, b = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"
        return "rgba(42,39,35,0.7)"

    fill_colors = []
    for cat in worst_df.get("aqi_category", ["Unknown"] * len(worst_df)):
        _, color = get_aqi_category(
            {"Good": 25, "Satisfactory": 75, "Moderate": 150,
             "Poor": 250, "Very Poor": 350, "Severe": 450}.get(cat, 300)
        )
        fill_colors.append(_hex_to_rgba(color, 0.3))

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#2A2723",
            font=dict(color="#EDE8DF", size=12),
            align="left",
            height=36,
        ),
        cells=dict(
            values=values,
            fill_color=[fill_colors] * len(display_cols),
            font=dict(color="#EDE8DF", size=11),
            align="left",
            height=30,
        ),
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title=f"Top {len(worst_df)} Worst Air Quality Days",
        margin=dict(l=20, r=20, t=80, b=20),
        height=420,
    )
    return fig


def multi_pollutant_line(df: pd.DataFrame, pollutants: list) -> go.Figure:
    """Multi-line chart of selected pollutants over time."""
    fig = go.Figure()
    for p in pollutants:
        if p not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["date"],
            y=df[p],
            mode="lines",
            name=POLLUTANT_LABELS.get(p, p),
            line=dict(color=POLLUTANT_COLORS.get(p), width=1.5),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{POLLUTANT_LABELS.get(p, p)}: %{{y:.2f}}<extra></extra>",
        ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Pollutant Trends Over Time",
        xaxis_title="Date",
        yaxis_title="Concentration",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=60),
        height=400,
    )
    return fig


def year_over_year_chart(yoy_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing yearly average AQI."""
    fig = go.Figure(go.Bar(
        x=yoy_df["year"].astype(str),
        y=yoy_df["aqi"],
        marker_color=yoy_df["aqi"],
        marker=dict(
            color=yoy_df["aqi"],
            colorscale=[[0, "#00B050"], [0.4, "#FFFF00"], [0.7, "#FF7C00"], [1.0, "#C00000"]],
            cmin=0, cmax=350,
            showscale=True,
            colorbar=dict(title="AQI"),
        ),
        text=yoy_df["aqi"].round(0).astype(int),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg AQI: %{y:.1f}<extra></extra>",
    ))

    fig.update_layout(
        template=DARK_TEMPLATE,
        title="Year-over-Year Average AQI",
        xaxis_title="Year",
        yaxis_title="Average AQI",
        margin=dict(l=60, r=20, t=80, b=60),
        height=360,
    )
    return fig
