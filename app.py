"""
NewX Services - Smart Spare Parts Demand Forecasting & JIT Inventory System
Streamlit dashboard (PR-0027)

Loads precomputed artifacts exported from the analysis notebook.
Run locally:  streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NewX Spare Parts Forecasting",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ART = Path("artifacts")


# -----------------------------------------------------------------------------
# Data loading (cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    data = {}
    data["weekly"] = pd.read_csv(ART / "weekly_demand.csv", index_col=0, parse_dates=True)
    data["pattern"] = pd.read_csv(ART / "pattern_classification.csv")
    data["forecast"] = pd.read_csv(ART / "forecast_8wk.csv", index_col=0, parse_dates=True)
    data["inventory"] = pd.read_csv(ART / "inventory_plan.csv")
    data["abc_xyz"] = pd.read_csv(ART / "abc_xyz.csv", index_col=0)
    data["final_best"] = pd.read_csv(ART / "final_best.csv")
    data["ranking"] = pd.read_csv(ART / "model_ranking.csv")
    with open(ART / "best_models.json") as f:
        data["best_models"] = json.load(f)
    data["top_items"] = pd.read_csv(ART / "top_line_items.csv", index_col=0)
    data["top_models"] = pd.read_csv(ART / "top_vehicle_models.csv", index_col=0)
    data["monthly"] = pd.read_csv(ART / "monthly_trend.csv", index_col=0, parse_dates=True)
    return data


def artifacts_exist():
    required = ["weekly_demand.csv", "pattern_classification.csv", "forecast_8wk.csv",
                "inventory_plan.csv", "abc_xyz.csv", "final_best.csv"]
    return ART.exists() and all((ART / r).exists() for r in required)


# -----------------------------------------------------------------------------
# Reorder-point math (interactive - recomputed live)
# -----------------------------------------------------------------------------
def service_level_to_z(service_level_pct):
    # Inverse normal CDF for common service levels (no scipy dependency)
    table = {80: 0.8416, 85: 1.0364, 90: 1.2816, 91: 1.3408, 92: 1.4051,
             93: 1.4758, 94: 1.5548, 95: 1.6449, 96: 1.7507, 97: 1.8808,
             98: 2.0537, 99: 2.3263}
    keys = sorted(table)
    sl = int(round(service_level_pct))
    if sl in table:
        return table[sl]
    # linear interpolate
    lo = max([k for k in keys if k <= sl], default=keys[0])
    hi = min([k for k in keys if k >= sl], default=keys[-1])
    if lo == hi:
        return table[lo]
    frac = (sl - lo) / (hi - lo)
    return table[lo] + frac * (table[hi] - table[lo])


def compute_reorder(weekly, lead_time_weeks, service_level_pct, recent_weeks=12):
    z = service_level_to_z(service_level_pct)
    rows = []
    for part in weekly.columns:
        recent = weekly[part].iloc[-recent_weeks:]
        mu, sigma = recent.mean(), recent.std()
        lt_demand = mu * lead_time_weeks
        safety = z * sigma * np.sqrt(lead_time_weeks)
        rop = lt_demand + safety
        rows.append({
            "Part": part,
            "Avg Weekly Demand": round(mu, 1),
            "Demand Std": round(sigma, 1),
            "Lead-time Demand": round(lt_demand, 1),
            "Safety Stock": round(safety, 1),
            "Reorder Point": int(np.ceil(rop)),
        })
    return pd.DataFrame(rows).sort_values("Avg Weekly Demand", ascending=False)


# -----------------------------------------------------------------------------
# Guard: artifacts missing
# -----------------------------------------------------------------------------
if not artifacts_exist():
    st.title("🔧 NewX Spare Parts Forecasting")
    st.error("Artifacts not found.")
    st.markdown(
        "This app expects an `artifacts/` folder with the exported CSV/JSON files.\n\n"
        "Run the **export cell** at the end of the analysis notebook to generate them, "
        "then place the `artifacts/` folder in the app repo root."
    )
    st.stop()

data = load_artifacts()
weekly = data["weekly"]
PARTS = list(weekly.columns)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
st.sidebar.title("🔧 NewX Services")
st.sidebar.caption("Smart Spare Parts Demand Forecasting & JIT Inventory System")
st.sidebar.markdown("---")
st.sidebar.metric("Spare parts tracked", len(PARTS))
st.sidebar.metric("Weeks of history", len(weekly))
st.sidebar.metric(
    "Total units (history)", f"{int(weekly.values.sum()):,}"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Project:** PM-PR-0027  \n"
    "**Category:** Inventory Forecasting  \n"
    "**Goal:** JIT spare-parts stocking"
)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Smart Spare Parts Demand Forecasting & JIT Inventory System")
st.markdown(
    "Forecasting weekly spare-parts demand and converting it into actionable "
    "**reorder points** so service centres hold the minimum stock needed to meet demand."
)

tab_eda, tab_forecast, tab_reorder, tab_abc = st.tabs(
    ["📊 Data Insights", "📈 Forecasts", "📦 Reorder Calculator", "🗂️ ABC-XYZ Strategy"]
)

# =============================================================================
# TAB 1 - EDA
# =============================================================================
with tab_eda:
    st.subheader("Exploratory Data Insights")

    c1, c2 = st.columns(2)
    with c1:
        ti = data["top_items"].reset_index()
        ti.columns = ["Line Item", "Count"]
        fig = px.bar(ti, x="Count", y="Line Item", orientation="h",
                     title="Top 20 Invoice Line Items", color="Count",
                     color_continuous_scale="Viridis")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=520,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Line items split into stockable spare parts (engine oil, air filter, etc.) "
            "and non-stock service/labour items, which we exclude from forecasting."
        )

    with c2:
        tm = data["top_models"].reset_index()
        tm.columns = ["Vehicle Model", "Count"]
        fig = px.bar(tm, x="Count", y="Vehicle Model", orientation="h",
                     title="Top 10 Vehicle Models by Service Volume", color="Count",
                     color_continuous_scale="Magma")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=520,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Service volume is concentrated in a few models, so parts specific to "
            "those models deserve stocking priority."
        )

    mt = data["monthly"].reset_index()
    mt.columns = ["Month", "Units"]
    fig = px.area(mt, x="Month", y="Units", title="Total Monthly Spare-Part Consumption",
                  markers=True)
    fig.update_traces(line_color="#2c7fb8")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Broadly stable consumption with mild fluctuation - good news for forecasting "
        "the high-volume consumables."
    )

    st.markdown("#### Demand Pattern Classification (Syntetos-Boylan)")
    pat = data["pattern"]
    colors = {"Smooth": "#1a9850", "Erratic": "#fd8d3c",
              "Intermittent": "#4575b4", "Lumpy": "#d73027"}
    fig = px.scatter(pat, x="ADI", y="CV2", color="Pattern", text="Part",
                     color_discrete_map=colors, height=520,
                     title="ADI vs CV-squared - which forecasting method each part needs")
    fig.add_vline(x=1.32, line_dash="dash", line_color="grey")
    fig.add_hline(y=0.49, line_dash="dash", line_color="grey")
    fig.update_traces(marker=dict(size=16, line=dict(width=1, color="black")),
                      textposition="top center")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bottom-left (Smooth) parts suit tight JIT replenishment; top-right "
        "(Lumpy/Intermittent) parts carry more forecast risk and need larger safety buffers."
    )

# =============================================================================
# TAB 2 - FORECASTS
# =============================================================================
with tab_forecast:
    st.subheader("8-Week Demand Forecast")

    best_models = data["best_models"]
    part = st.selectbox("Select a spare part", PARTS, key="fc_part")

    actual = weekly[part]
    fc = data["forecast"][part]

    model_used = best_models.get(part, "n/a")
    pattern = pat.set_index("Part").loc[part, "Pattern"] if part in pat["Part"].values else "n/a"

    m1, m2, m3 = st.columns(3)
    m1.metric("Best model", model_used)
    m2.metric("Demand pattern", pattern)
    m3.metric("Forecast next 8 wks", f"{int(round(fc.sum()))} units")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name="Actual",
                             line=dict(color="#253494")))
    fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name="Forecast",
                             line=dict(color="#d7301f", dash="dash"), mode="lines+markers"))
    fig.update_layout(title=f"{part} - weekly demand and 8-week forecast",
                      height=460, xaxis_title="Week", yaxis_title="Units")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Model Leaderboard (mean RMSE across all parts)")
    rank = data["ranking"]
    fig = px.bar(rank, x="Mean_RMSE_across_parts", y="Model", orientation="h",
                 color="Mean_RMSE_across_parts", color_continuous_scale="Blues_r",
                 title="Lower is better")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Nine algorithms were tested - classical (Holt-Winters, SARIMA), intermittent "
        "(Croston's, SBA, TSB), modern ML (global LightGBM), Prophet and Theta. The best "
        "model is chosen per part; no single algorithm dominates."
    )

# =============================================================================
# TAB 3 - REORDER CALCULATOR
# =============================================================================
with tab_reorder:
    st.subheader("Interactive Reorder Point & Safety Stock Calculator")
    st.markdown(
        "Adjust the supplier lead time and target service level to see how reorder "
        "points and safety stock change. This is the operational JIT deliverable."
    )

    c1, c2, c3 = st.columns(3)
    lead_time = c1.slider("Lead time (weeks)", 1, 6, 2)
    service_level = c2.slider("Service level (%)", 80, 99, 95)
    recent_weeks = c3.slider("Recent weeks for estimate", 4, 24, 12)

    z = service_level_to_z(service_level)
    c3.caption(f"Z-score = {z:.3f}")

    rop_df = compute_reorder(weekly, lead_time, service_level, recent_weeks)

    st.dataframe(rop_df, use_container_width=True, hide_index=True)

    fig = px.bar(rop_df, x="Part", y="Reorder Point",
                 title=f"Reorder Points  (lead time {lead_time}w, {service_level}% service level)",
                 color="Reorder Point", color_continuous_scale="Tealgrn")
    fig.update_layout(height=420, coloraxis_showscale=False, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    total_stock = int(rop_df["Reorder Point"].sum())
    total_safety = int(round(rop_df["Safety Stock"].sum()))
    m1, m2 = st.columns(2)
    m1.metric("Total reorder-point stock", f"{total_stock} units")
    m2.metric("Total safety stock", f"{total_safety} units")
    st.caption(
        "When on-hand stock for a part drops to its reorder point, place a replenishment "
        "order. Higher service levels and longer lead times both raise safety stock."
    )

# =============================================================================
# TAB 4 - ABC-XYZ
# =============================================================================
with tab_abc:
    st.subheader("ABC-XYZ Inventory Strategy")
    st.markdown(
        "Combining **volume importance (ABC)** with **demand predictability (XYZ)** "
        "to assign the right control policy to each part."
    )

    abc = data["abc_xyz"]
    abc_show = abc.reset_index().rename(columns={"index": "Part"})
    st.dataframe(abc_show, use_container_width=True, hide_index=True)

    # Build count matrix
    count_matrix = (abc.groupby(["ABC", "XYZ"]).size()
                    .unstack(fill_value=0)
                    .reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0))

    # Annotate with part names
    annot = pd.DataFrame("", index=["A", "B", "C"], columns=["X", "Y", "Z"])
    for part_name, r in abc.iterrows():
        cell = annot.loc[r["ABC"], r["XYZ"]]
        annot.loc[r["ABC"], r["XYZ"]] = (cell + "<br>" + str(part_name)).strip("<br>")

    fig = go.Figure(data=go.Heatmap(
        z=count_matrix.values,
        x=["X (steady)", "Y (variable)", "Z (erratic)"],
        y=["A (high vol)", "B (mid vol)", "C (low vol)"],
        text=annot.values, texttemplate="%{text}",
        colorscale="YlGnBu", showscale=True,
        colorbar=dict(title="parts"),
    ))
    fig.update_layout(title="ABC-XYZ Matrix", height=480)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Recommended policy by quadrant")
    st.markdown(
        "- **AX / AY** - high volume, predictable: tight JIT, frequent small orders, low safety stock.\n"
        "- **AZ / BZ / CZ** - erratic demand: larger safety stock or periodic review; forecasts less reliable.\n"
        "- **C-row** - low volume: simple min-max rules; do not over-invest in modelling.\n"
    )

st.markdown("---")
st.caption(
    "PR-0027 - NewX Services Spare Parts Demand Forecasting. "
    "Forecasts use a per-part selection across nine algorithms; reorder points assume a "
    "normal demand approximation at the chosen service level."
)
