"""
6G-Enabled Smart Manufacturing Analytics Dashboard
====================================================
Diagnostic analytics platform for Thales smart-manufacturing telemetry:
machine health, production performance, quality, network performance,
KPI engine, and an executive efficiency-classification report built on
the Logistic Regression model trained in Thales_Group_Analysis.ipynb.

Run:  streamlit run app.py
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import streamlit as st

# ════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="6G Smart Manufacturing Analytics | Thales",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "Manufacturing_Cleaned_Dataset.csv"
MODEL_PATH = Path(__file__).parent / "manufacturing_logistic_model.pkl"

# ════════════════════════════════════════════════════════════════════
#  GLOBAL THEME
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg:        #070B14;
    --surface:   #0E1420;
    --surface2:  #141C2C;
    --border:    rgba(255,255,255,0.08);
    --cyan:      #2DD4F0;
    --cyan2:     #0E9DB5;
    --indigo:    #6C7BFF;
    --amber:     #F5B22A;
    --red:       #FF5C72;
    --green:     #34D399;
    --text1:     #EAF0FB;
    --text2:     #8593AD;
}

html, body, .stApp { background-color: var(--bg) !important; font-family:'Inter',sans-serif; color:var(--text1); }
#MainMenu, footer, header { visibility:hidden; }

[data-testid="stSidebar"] { background: var(--surface) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] * { color: var(--text1) !important; }

.hero {
    background: linear-gradient(120deg, rgba(45,212,240,.10), rgba(108,123,255,.06));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 26px 30px;
    margin-bottom: 22px;
}
.hero h1 { margin:0; font-size:1.85rem; font-weight:800; letter-spacing:-.01em; }
.hero p { margin:6px 0 0 0; color:var(--text2); font-size:.92rem; }

.card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}

.kpi {
    background: var(--surface2);
    border-radius: 12px;
    padding: 16px 16px;
    text-align: center;
    border-left: 4px solid var(--cyan);
}
.kpi .label { font-size:.68rem; letter-spacing:.07em; text-transform:uppercase; color:var(--text2); margin:0; }
.kpi .value { font-size:1.7rem; font-weight:800; margin:6px 0 0 0; color:var(--text1); }
.kpi .delta { font-size:.72rem; margin-top:2px; }
.kpi.amber { border-left-color:var(--amber); }
.kpi.red   { border-left-color:var(--red); }
.kpi.green { border-left-color:var(--green); }
.kpi.indigo{ border-left-color:var(--indigo); }

.section-title {
    font-size: 1.35rem; font-weight: 800; color: var(--text1);
    margin: 6px 0 2px 0; padding-bottom: 8px;
    border-bottom: 2px solid var(--cyan); display:inline-block;
}
.section-sub { font-size:.82rem; color:var(--text2); margin-bottom:18px; }

.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:.7rem; font-weight:700; letter-spacing:.03em; }
.badge.high { background:rgba(52,211,153,.15); color:var(--green); }
.badge.medium { background:rgba(245,178,42,.15); color:var(--amber); }
.badge.low { background:rgba(255,92,114,.15); color:var(--red); }

.js-plotly-plot .plotly { background: transparent !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background:#23304a; border-radius:3px; }

div[data-testid="stExpander"] { border:1px solid var(--border) !important; border-radius:10px !important; }
.streamlit-expanderHeader { background: var(--surface2) !important; color: var(--cyan) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
</style>
""", unsafe_allow_html=True)

PALETTE = {"High": "#34D399", "Medium": "#F5B22A", "Low": "#FF5C72"}
MODE_PALETTE = {"Active": "#2DD4F0", "Idle": "#6C7BFF", "Maintenance": "#FF5C72"}

# ════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading manufacturing telemetry…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Derive Efficiency_Status / Operation_Mode only if genuinely absent
    if "Efficiency_Status" not in df.columns:
        def _eff(row):
            score = (
                (row["Production_Speed"] / 500) * 0.35
                + (1 - row["Error_Rate"] / 15) * 0.25
                + (1 - row["Defect_Rate"] / 10) * 0.20
                + (row["Maintenance_Score"]) * 0.20
            )
            if score >= 0.66:
                return "High"
            elif score >= 0.40:
                return "Medium"
            return "Low"
        df["Efficiency_Status"] = df.apply(_eff, axis=1)

    if "Operation_Mode" not in df.columns:
        def _mode(row):
            if row["Production_Speed"] < 60:
                return "Idle"
            if row["Error_Rate"] > 12 or row["Maintenance_Score"] > 0.85:
                return "Maintenance"
            return "Active"
        df["Operation_Mode"] = df.apply(_mode, axis=1)

    df["Efficiency_Status"] = pd.Categorical(
        df["Efficiency_Status"], categories=["Low", "Medium", "High"], ordered=True
    )
    return df


@st.cache_resource(show_spinner=False)
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(show_spinner=False)
def compute_kpis(df: pd.DataFrame) -> dict:
    norm_temp = (df["Temperature_C"] - df["Temperature_C"].min()) / (
        df["Temperature_C"].max() - df["Temperature_C"].min() + 1e-9
    )
    norm_vib = (df["Vibration_Hz"] - df["Vibration_Hz"].min()) / (
        df["Vibration_Hz"].max() - df["Vibration_Hz"].min() + 1e-9
    )
    risk = (norm_temp * 0.30 + norm_vib * 0.30 + df["Maintenance_Score"] * 0.40)
    machine_health_index = float(np.clip(100 * (1 - risk.mean()), 0, 100))

    return {
        "machine_health_index": machine_health_index,
        "avg_production_speed": df["Production_Speed"].mean(),
        "defect_density_score": df["Defect_Rate"].mean(),
        "error_frequency_index": df["Error_Rate"].mean(),
        "avg_network_latency": df["Network_Latency"].mean(),
        "avg_packet_loss": df["Packet_Loss"].mean(),
        "avg_maintenance_score": df["Maintenance_Score"].mean(),
        "avg_temperature": df["Temperature_C"].mean(),
        "avg_vibration": df["Vibration_Hz"].mean(),
        "avg_power": df["Power_Consumption"].mean(),
        "high_eff_pct": (df["Efficiency_Status"] == "High").mean() * 100,
        "machines_online": df["Machine_ID"].nunique(),
        "records": len(df),
    }


def kpi_card(label: str, value: str, css_class: str = "") -> str:
    return f'<div class="kpi {css_class}"><p class="label">{label}</p><p class="value">{value}</p></div>'


def gauge(value: float, title: str, max_val: float = 100, suffix: str = "", color: str = "#2DD4F0") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix, "font": {"color": "#EAF0FB", "size": 30}},
        title={"text": title, "font": {"color": "#8593AD", "size": 13}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#8593AD"},
            "bar": {"color": color},
            "bgcolor": "#141C2C",
            "borderwidth": 0,
            "steps": [
                {"range": [0, max_val * 0.4], "color": "rgba(255,92,114,.15)"},
                {"range": [max_val * 0.4, max_val * 0.7], "color": "rgba(245,178,42,.15)"},
                {"range": [max_val * 0.7, max_val], "color": "rgba(52,211,153,.15)"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(t=50, b=10, l=20, r=20),
                       paper_bgcolor="rgba(0,0,0,0)", font_color="#EAF0FB")
    return fig


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#EAF0FB",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=40, l=40, r=20),
    )
    fig.update_xaxes(gridcolor="#1d2840")
    fig.update_yaxes(gridcolor="#1d2840")
    return fig


# ════════════════════════════════════════════════════════════════════
#  LOAD DATA
# ════════════════════════════════════════════════════════════════════
raw_df = load_data()
model_artifacts = load_model()

NUMERIC_COLS = [
    "Temperature_C", "Vibration_Hz", "Power_Consumption", "Network_Latency",
    "Packet_Loss", "Defect_Rate", "Production_Speed", "Maintenance_Score", "Error_Rate",
]

# ════════════════════════════════════════════════════════════════════
#  SIDEBAR — NAVIGATION + FILTERS
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏭 6G Smart Mfg")
    st.caption("Thales Manufacturing Intelligence")
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔧 Machine Health", "⚙️ Production", "🛡️ Quality",
         "📡 Network", "🔗 Correlation Analysis", "📑 Executive Report"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### Filters")

    if st.button("↺ Reset Filters", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("flt_"):
                del st.session_state[k]
        st.rerun()

    machine_search = st.text_input("Search Machine ID", key="flt_search", placeholder="e.g. 12")

    all_machines = sorted(raw_df["Machine_ID"].unique())
    if machine_search:
        try:
            num = int(machine_search)
            all_machines_filtered = [m for m in all_machines if str(num) in str(m)]
        except ValueError:
            all_machines_filtered = all_machines
    else:
        all_machines_filtered = all_machines

    sel_machines = st.multiselect(
        "Machine ID", options=all_machines_filtered, default=[], key="flt_machines",
        help="Leave empty to include all machines",
    )
    sel_modes = st.multiselect(
        "Operation Mode", options=sorted(raw_df["Operation_Mode"].unique()),
        default=[], key="flt_modes",
    )
    sel_months = st.multiselect(
        "Month", options=sorted(raw_df["Month"].unique()), default=[], key="flt_months",
    )
    sel_days = st.multiselect(
        "Day", options=sorted(raw_df["Day"].unique()), default=[], key="flt_days",
    )
    hour_range = st.slider("Hour Range", 0, 23, (0, 23), key="flt_hours")

    metric_options = NUMERIC_COLS
    sel_metric = st.selectbox("Primary Metric", metric_options, index=6, key="flt_metric")
    sel_features = st.multiselect(
        "Multi-Feature Selector", metric_options,
        default=["Temperature_C", "Vibration_Hz", "Production_Speed"], key="flt_features",
    )

    st.markdown("---")
    st.caption(f"Dataset: {len(raw_df):,} records · {raw_df['Machine_ID'].nunique()} machines")

# Apply filters
df = raw_df.copy()
if sel_machines:
    df = df[df["Machine_ID"].isin(sel_machines)]
if sel_modes:
    df = df[df["Operation_Mode"].isin(sel_modes)]
if sel_months:
    df = df[df["Month"].isin(sel_months)]
if sel_days:
    df = df[df["Day"].isin(sel_days)]
df = df[(df["Hour"] >= hour_range[0]) & (df["Hour"] <= hour_range[1])]

if df.empty:
    st.warning("No records match the current filters. Showing full dataset instead.")
    df = raw_df.copy()

kpis = compute_kpis(df)

# ════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown(
        '<div class="hero"><h1>6G-Enabled Smart Manufacturing Analytics</h1>'
        '<p>Executive overview of factory-wide machine health, production performance, '
        'quality and network telemetry across the connected production line.</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi_card("Machine Health Index", f"{kpis['machine_health_index']:.1f}", "green"), unsafe_allow_html=True)
    c2.markdown(kpi_card("Avg Production Speed", f"{kpis['avg_production_speed']:.0f} u/hr"), unsafe_allow_html=True)
    c3.markdown(kpi_card("Defect Density", f"{kpis['defect_density_score']:.2f}%", "amber"), unsafe_allow_html=True)
    c4.markdown(kpi_card("Error Frequency", f"{kpis['error_frequency_index']:.2f}%", "red"), unsafe_allow_html=True)
    c5.markdown(kpi_card("High-Efficiency Share", f"{kpis['high_eff_pct']:.1f}%", "indigo"), unsafe_allow_html=True)

    st.write("")
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(gauge(kpis["machine_health_index"], "Factory Health Index", color="#34D399"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge(kpis["avg_maintenance_score"] * 100, "Maintenance Risk Score", color="#F5B22A"), use_container_width=True)
    with g3:
        st.plotly_chart(gauge(kpis["avg_network_latency"], "Avg Network Latency", max_val=50, suffix=" ms", color="#2DD4F0"), use_container_width=True)

    st.markdown('<p class="section-title">Quick Insights</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])
    with col1:
        eff_trend = df.groupby(["Month", "Efficiency_Status"], observed=True).size().reset_index(name="Count")
        fig = px.area(eff_trend, x="Month", y="Count", color="Efficiency_Status",
                       color_discrete_map=PALETTE, title="Efficiency Status Trend by Month")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col2:
        eff_counts = df["Efficiency_Status"].value_counts().reindex(["High", "Medium", "Low"])
        fig = px.pie(values=eff_counts.values, names=eff_counts.index, hole=0.55,
                      color=eff_counts.index, color_discrete_map=PALETTE, title="Efficiency Mix")
        st.plotly_chart(style_fig(fig, height=380), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        mode_counts = df["Operation_Mode"].value_counts()
        fig = px.bar(x=mode_counts.index, y=mode_counts.values, color=mode_counts.index,
                      color_discrete_map=MODE_PALETTE, title="Operation Mode Distribution",
                      labels={"x": "Operation Mode", "y": "Records"})
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with col4:
        worst = df.groupby("Machine_ID")["Error_Rate"].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=worst.index.astype(str), y=worst.values, title="Top 10 Machines by Error Rate",
                      labels={"x": "Machine ID", "y": "Avg Error Rate (%)"}, color=worst.values,
                      color_continuous_scale="Reds")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.expander("📥 Export Filtered Dataset"):
        st.download_button("Download CSV", df.to_csv(index=False).encode(), "filtered_manufacturing_data.csv", "text/csv")
        st.dataframe(df.head(200), use_container_width=True, height=300)

# ════════════════════════════════════════════════════════════════════
#  PAGE: DATA VALIDATION (embedded inside Dashboard expander too, plus its own block)
# ════════════════════════════════════════════════════════════════════
def data_validation_block(d: pd.DataFrame):
    st.markdown('<p class="section-title">Data Validation & Preparation</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Quality control checks against the active filtered dataset.</p>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    v1.markdown(kpi_card("Missing Values", f"{int(d.isna().sum().sum()):,}"), unsafe_allow_html=True)
    v2.markdown(kpi_card("Duplicate Records", f"{int(d.duplicated().sum()):,}", "amber"), unsafe_allow_html=True)
    v3.markdown(kpi_card("Records", f"{len(d):,}", "indigo"), unsafe_allow_html=True)
    v4.markdown(kpi_card("Machines", f"{d['Machine_ID'].nunique()}", "green"), unsafe_allow_html=True)

    ranges = {
        "Temperature_C": (0, 120), "Vibration_Hz": (0, 10), "Power_Consumption": (0, 20),
        "Network_Latency": (0, 100), "Packet_Loss": (0, 10), "Defect_Rate": (0, 20),
        "Production_Speed": (0, 600), "Maintenance_Score": (0, 1), "Error_Rate": (0, 25),
    }
    rows = []
    for col, (lo, hi) in ranges.items():
        invalid = ((d[col] < lo) | (d[col] > hi)).sum()
        rows.append({"Sensor": col, "Expected Range": f"{lo}–{hi}", "Invalid Records": int(invalid),
                      "Status": "✅ OK" if invalid == 0 else "⚠️ Review"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
#  PAGE: MACHINE HEALTH
# ════════════════════════════════════════════════════════════════════
if page == "🔧 Machine Health":
    st.markdown('<p class="section-title">Machine-Level Sensor Health</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Temperature, vibration, power, and predictive-maintenance posture per machine.</p>', unsafe_allow_html=True)

    data_validation_block(df)
    st.write("")

    machine_summary = df.groupby("Machine_ID").agg(
        Avg_Temperature=("Temperature_C", "mean"),
        Avg_Vibration=("Vibration_Hz", "mean"),
        Avg_Power=("Power_Consumption", "mean"),
        Avg_Maintenance_Score=("Maintenance_Score", "mean"),
        Avg_Error_Rate=("Error_Rate", "mean"),
    ).round(2).reset_index()

    h1, h2 = st.columns(2)
    with h1:
        fig = px.bar(machine_summary.sort_values("Avg_Temperature", ascending=False).head(15),
                      x="Machine_ID", y="Avg_Temperature", title="Average Temperature by Machine (Top 15)",
                      color="Avg_Temperature", color_continuous_scale="OrRd")
        fig.update_xaxes(type="category")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with h2:
        fig = px.line(machine_summary.sort_values("Machine_ID"), x="Machine_ID", y="Avg_Vibration",
                       markers=True, title="Average Vibration across Machines")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    h3, h4 = st.columns(2)
    with h3:
        radar_machines = sel_machines if sel_machines else machine_summary["Machine_ID"].head(5).tolist()
        fig = go.Figure()
        for m in radar_machines[:6]:
            row = machine_summary[machine_summary["Machine_ID"] == m]
            if row.empty:
                continue
            r = row.iloc[0]
            vals = [r["Avg_Temperature"] / 90, r["Avg_Vibration"] / 5, r["Avg_Power"] / 10,
                    r["Avg_Maintenance_Score"], r["Avg_Error_Rate"] / 15]
            fig.add_trace(go.Scatterpolar(r=vals + [vals[0]],
                                            theta=["Temp", "Vibration", "Power", "Maint. Risk", "Error Rate", "Temp"],
                                            fill="toself", name=f"Machine {m}"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), title="Machine Health Radar (Normalized)")
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    with h4:
        heat = df.pivot_table(index="Machine_ID", columns="Hour", values=sel_metric, aggfunc="mean")
        fig = px.imshow(heat, aspect="auto", color_continuous_scale="Viridis",
                          title=f"Machine × Hour Heatmap — {sel_metric}", labels=dict(color=sel_metric))
        st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    st.markdown('<p class="section-title">Machine Health Scorecards</p>', unsafe_allow_html=True)
    machine_summary["Health_Index"] = (
        100 * (1 - (
            (machine_summary["Avg_Temperature"] / machine_summary["Avg_Temperature"].max()) * 0.30
            + (machine_summary["Avg_Vibration"] / machine_summary["Avg_Vibration"].max()) * 0.30
            + machine_summary["Avg_Maintenance_Score"] * 0.40
        ))
    ).round(1)
    cards = st.columns(5)
    near_threshold = machine_summary.sort_values("Health_Index").head(5)
    for i, (_, row) in enumerate(near_threshold.iterrows()):
        cls = "red" if row["Health_Index"] < 40 else "amber" if row["Health_Index"] < 65 else "green"
        cards[i % 5].markdown(
            kpi_card(f"Machine {int(row['Machine_ID'])}", f"{row['Health_Index']:.0f}", cls), unsafe_allow_html=True
        )
    st.caption("⚠️ Machines shown above have the lowest computed Health Index — closest to operating thresholds.")
    st.dataframe(machine_summary.sort_values("Health_Index"), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE: PRODUCTION
# ════════════════════════════════════════════════════════════════════
if page == "⚙️ Production":
    st.markdown('<p class="section-title">Production Performance Diagnostics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Throughput, consistency, and machine-comparison analysis.</p>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    p1.markdown(kpi_card("Avg Production Speed", f"{df['Production_Speed'].mean():.0f} u/hr"), unsafe_allow_html=True)
    p2.markdown(kpi_card("Peak Speed", f"{df['Production_Speed'].max():.0f} u/hr", "green"), unsafe_allow_html=True)
    p3.markdown(kpi_card("Std. Deviation (Consistency)", f"{df['Production_Speed'].std():.1f}", "amber"), unsafe_allow_html=True)

    speed_by_machine = df.groupby("Machine_ID")["Production_Speed"].mean().sort_values(ascending=False)
    t1, t2 = st.columns(2)
    with t1:
        top10 = speed_by_machine.head(10)
        fig = px.bar(x=top10.index.astype(str), y=top10.values, title="Top 10 Performing Machines",
                      color=top10.values, color_continuous_scale="Greens", labels={"x": "Machine", "y": "Avg Speed"})
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with t2:
        bottom10 = speed_by_machine.tail(10)
        fig = px.bar(x=bottom10.index.astype(str), y=bottom10.values, title="Bottom 10 Performing Machines",
                      color=bottom10.values, color_continuous_scale="Reds", labels={"x": "Machine", "y": "Avg Speed"})
        st.plotly_chart(style_fig(fig), use_container_width=True)

    t3, t4 = st.columns(2)
    with t3:
        trend = df.groupby(["Month", "Day"])["Production_Speed"].mean().reset_index()
        trend["Date"] = trend["Month"].astype(str) + "-" + trend["Day"].astype(str)
        fig = px.line(trend, x="Date", y="Production_Speed", title="Production Trend Over Time")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with t4:
        fig = px.box(df, x="Operation_Mode", y="Production_Speed", color="Operation_Mode",
                      color_discrete_map=MODE_PALETTE, title="Production Speed Consistency by Operation Mode")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    fig = px.histogram(df, x="Production_Speed", nbins=40, title="Production Speed Distribution",
                         color_discrete_sequence=["#2DD4F0"])
    st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.expander("📋 Machine Comparison Table"):
        st.dataframe(speed_by_machine.reset_index().rename(columns={"Production_Speed": "Avg_Production_Speed"}),
                      use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE: QUALITY
# ════════════════════════════════════════════════════════════════════
if page == "🛡️ Quality":
    st.markdown('<p class="section-title">Quality & Error Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Defect rate, error rate, and machine quality bottlenecks.</p>', unsafe_allow_html=True)

    q1, q2, q3 = st.columns(3)
    q1.markdown(kpi_card("Avg Error Rate", f"{df['Error_Rate'].mean():.2f}%", "red"), unsafe_allow_html=True)
    q2.markdown(kpi_card("Avg Defect Rate", f"{df['Defect_Rate'].mean():.2f}%", "amber"), unsafe_allow_html=True)
    q3.markdown(kpi_card("Quality Score", f"{100 - df[['Error_Rate','Defect_Rate']].mean().mean():.1f}", "green"), unsafe_allow_html=True)

    q4, q5 = st.columns(2)
    with q4:
        fig = px.scatter(df.sample(min(4000, len(df)), random_state=1), x="Temperature_C", y="Defect_Rate",
                           color="Efficiency_Status", color_discrete_map=PALETTE, opacity=0.6,
                           title="Temperature vs Defect Rate")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with q5:
        fig = px.box(df, x="Machine_ID" if df["Machine_ID"].nunique() <= 15 else "Operation_Mode",
                      y="Error_Rate", color="Operation_Mode", color_discrete_map=MODE_PALETTE,
                      title="Error Rate Distribution")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    q6, q7 = st.columns(2)
    with q6:
        fig = px.histogram(df, x="Defect_Rate", nbins=40, color="Efficiency_Status",
                             color_discrete_map=PALETTE, title="Defect Rate Histogram", barmode="overlay")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with q7:
        rank = df.groupby("Machine_ID")[["Error_Rate", "Defect_Rate"]].mean().assign(
            Quality_Score=lambda x: 100 - (x["Error_Rate"] + x["Defect_Rate"]) / 2
        ).sort_values("Quality_Score").head(10)
        fig = px.bar(rank.reset_index(), x="Machine_ID", y="Quality_Score", title="Lowest Quality-Score Machines",
                      color="Quality_Score", color_continuous_scale="Reds")
        fig.update_xaxes(type="category")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    corr = df[["Error_Rate", "Defect_Rate", "Temperature_C", "Vibration_Hz", "Maintenance_Score"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Quality Correlation Matrix")
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE: NETWORK
# ════════════════════════════════════════════════════════════════════
if page == "📡 Network":
    st.markdown('<p class="section-title">6G Network Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Latency and packet-loss telemetry across the connected machine fleet.</p>', unsafe_allow_html=True)

    n1, n2, n3 = st.columns(3)
    n1.markdown(kpi_card("Avg Network Latency", f"{df['Network_Latency'].mean():.2f} ms", "indigo"), unsafe_allow_html=True)
    n2.markdown(kpi_card("Avg Packet Loss", f"{df['Packet_Loss'].mean():.2f}%", "amber"), unsafe_allow_html=True)
    n3.markdown(kpi_card("Network SLA Risk Records", f"{(df['Packet_Loss'] > 4).sum():,}", "red"), unsafe_allow_html=True)

    n4, n5 = st.columns(2)
    with n4:
        fig = px.scatter(df.sample(min(4000, len(df)), random_state=2), x="Network_Latency", y="Packet_Loss",
                           color="Operation_Mode", color_discrete_map=MODE_PALETTE, opacity=0.6,
                           title="Network Latency vs Packet Loss")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with n5:
        latency_machine = df.groupby("Machine_ID")["Network_Latency"].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=latency_machine.index.astype(str), y=latency_machine.values, title="Top 10 Machines by Avg Latency",
                      color=latency_machine.values, color_continuous_scale="Blues")
        st.plotly_chart(style_fig(fig), use_container_width=True)

    n6, n7 = st.columns(2)
    with n6:
        fig = px.violin(df, y="Network_Latency", x="Operation_Mode", color="Operation_Mode",
                          color_discrete_map=MODE_PALETTE, box=True, title="Latency Distribution by Operation Mode")
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with n7:
        hourly = df.groupby("Hour")[["Network_Latency", "Packet_Loss"]].mean().reset_index()
        fig = px.area(hourly, x="Hour", y=["Network_Latency", "Packet_Loss"], title="Hourly Network Quality Trend")
        st.plotly_chart(style_fig(fig), use_container_width=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE: CORRELATION ANALYSIS
# ════════════════════════════════════════════════════════════════════
if page == "🔗 Correlation Analysis":
    st.markdown('<p class="section-title">Cross-Metric Diagnostics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Relationships between sensor, production, quality and network metrics.</p>', unsafe_allow_html=True)

    pairs = [
        ("Temperature_C", "Defect_Rate", "Temperature vs Defect Rate"),
        ("Vibration_Hz", "Error_Rate", "Vibration vs Error Rate"),
        ("Power_Consumption", "Production_Speed", "Power Consumption vs Production Speed"),
        ("Maintenance_Score", "Error_Rate", "Maintenance Score vs Error Rate"),
        ("Network_Latency", "Packet_Loss", "Network Latency vs Packet Loss"),
    ]
    sample = df.sample(min(4000, len(df)), random_state=3)
    cols = st.columns(2)
    for i, (x, y, title) in enumerate(pairs):
        fig = px.scatter(sample, x=x, y=y, color="Efficiency_Status", color_discrete_map=PALETTE,
                           opacity=0.55, trendline="ols", title=title)
        cols[i % 2].plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<p class="section-title">Correlation Heatmap</p>', unsafe_allow_html=True)
    corr = df[NUMERIC_COLS].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Full Metric Correlation Matrix")
    st.plotly_chart(style_fig(fig, height=500), use_container_width=True)

    st.markdown('<p class="section-title">Interactive Pair Plot</p>', unsafe_allow_html=True)
    pp_features = sel_features if len(sel_features) >= 2 else ["Temperature_C", "Vibration_Hz", "Production_Speed"]
    fig = px.scatter_matrix(sample, dimensions=pp_features, color="Efficiency_Status",
                              color_discrete_map=PALETTE, title="Pair Plot — Selected Features")
    st.plotly_chart(style_fig(fig, height=650), use_container_width=True)

    st.markdown('<p class="section-title">Sunburst — Mode → Efficiency</p>', unsafe_allow_html=True)
    sb = df.groupby(["Operation_Mode", "Efficiency_Status"], observed=True).size().reset_index(name="Count")
    fig = px.sunburst(sb, path=["Operation_Mode", "Efficiency_Status"], values="Count",
                        color="Efficiency_Status", color_discrete_map=PALETTE)
    st.plotly_chart(style_fig(fig, height=480), use_container_width=True)

# ════════════════════════════════════════════════════════════════════
#  PAGE: EXECUTIVE REPORT  (KPI engine + ML model)
# ════════════════════════════════════════════════════════════════════
if page == "📑 Executive Report":
    st.markdown('<p class="section-title">Executive KPI Report</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Factory-wide KPI engine and Logistic Regression efficiency classifier.</p>', unsafe_allow_html=True)

    kpi_table = pd.DataFrame({
        "KPI": ["Machine Health Index", "Avg Production Speed", "Defect Density Score",
                "Error Frequency Index", "Avg Network Latency", "Avg Packet Loss",
                "Avg Maintenance Score", "Avg Temperature", "Avg Vibration", "Avg Power Consumption"],
        "Value": [
            f"{kpis['machine_health_index']:.1f}", f"{kpis['avg_production_speed']:.1f} u/hr",
            f"{kpis['defect_density_score']:.2f}%", f"{kpis['error_frequency_index']:.2f}%",
            f"{kpis['avg_network_latency']:.2f} ms", f"{kpis['avg_packet_loss']:.2f}%",
            f"{kpis['avg_maintenance_score']:.2f}", f"{kpis['avg_temperature']:.1f} °C",
            f"{kpis['avg_vibration']:.2f} Hz", f"{kpis['avg_power']:.2f} kW",
        ],
    })
    st.dataframe(kpi_table, use_container_width=True, hide_index=True)

    e1, e2, e3, e4 = st.columns(4)
    e1.markdown(kpi_card("Machine Health Index", f"{kpis['machine_health_index']:.1f}", "green"), unsafe_allow_html=True)
    e2.markdown(kpi_card("Machines Monitored", f"{kpis['machines_online']}", "indigo"), unsafe_allow_html=True)
    e3.markdown(kpi_card("Records Analyzed", f"{kpis['records']:,}"), unsafe_allow_html=True)
    e4.markdown(kpi_card("High-Efficiency Share", f"{kpis['high_eff_pct']:.1f}%", "amber"), unsafe_allow_html=True)

    st.markdown('<p class="section-title">Efficiency Status Classifier (Logistic Regression)</p>', unsafe_allow_html=True)

    if model_artifacts is None:
        st.warning("Model artifact `manufacturing_logistic_model.pkl` not found alongside app.py.")
    else:
        model = model_artifacts["model"]
        scaler = model_artifacts["scaler"]
        le_mode = model_artifacts["operation_mode_encoder"]
        le_target = model_artifacts["target_encoder"]
        feature_names = model_artifacts["feature_names"]

        # Map cleaned-dataset column names -> notebook's pre-rename feature names
        rename_back = {
            "Power_Consumption": "Power_Consumption_kW",
            "Network_Latency": "Network_Latency_ms",
            "Packet_Loss": "Packet_Loss_%",
            "Defect_Rate": "Quality_Control_Defect_Rate_%",
            "Production_Speed": "Production_Speed_units_per_hr",
            "Maintenance_Score": "Predictive_Maintenance_Score",
            "Error_Rate": "Error_Rate_%",
        }

        with st.expander("ℹ️ About this model"):
            st.write(
                "A Logistic Regression classifier trained in `Thales_Group_Analysis.ipynb` predicts "
                "`Efficiency_Status` (High / Medium / Low) from sensor, production, quality and network "
                "features. Inputs are standardized with the notebook's fitted `StandardScaler` before inference."
            )

        tab1, tab2 = st.tabs(["🔮 Live Prediction", "📈 Model Performance on Current Filter"])

        with tab1:
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                in_machine = st.number_input("Machine ID", 1, 50, 1)
                in_mode = st.selectbox("Operation Mode", le_mode.classes_)
                in_temp = st.slider("Temperature (°C)", 30.0, 90.0, 60.0)
                in_vib = st.slider("Vibration (Hz)", 0.1, 5.0, 2.5)
                in_power = st.slider("Power Consumption (kW)", 1.5, 10.0, 5.7)
            with fc2:
                in_latency = st.slider("Network Latency (ms)", 1.0, 50.0, 25.0)
                in_loss = st.slider("Packet Loss (%)", 0.0, 5.0, 2.5)
                in_defect = st.slider("Defect Rate (%)", 0.0, 10.0, 5.0)
                in_speed = st.slider("Production Speed (u/hr)", 50.0, 500.0, 275.0)
                in_maint = st.slider("Maintenance Score", 0.0, 1.0, 0.5)
            with fc3:
                in_error = st.slider("Error Rate (%)", 0.0, 15.0, 7.5)
                in_hour = st.slider("Hour", 0, 23, 12)
                in_minute = st.slider("Minute", 0, 59, 0)
                in_day = st.slider("Day", 1, 31, 15)
                in_month = st.slider("Month", 1, 12, 1)

            if st.button("Predict Efficiency Status", type="primary"):
                row = {
                    "Machine_ID": in_machine,
                    "Operation_Mode": le_mode.transform([in_mode])[0],
                    "Temperature_C": in_temp,
                    "Vibration_Hz": in_vib,
                    "Power_Consumption_kW": in_power,
                    "Network_Latency_ms": in_latency,
                    "Packet_Loss_%": in_loss,
                    "Quality_Control_Defect_Rate_%": in_defect,
                    "Production_Speed_units_per_hr": in_speed,
                    "Predictive_Maintenance_Score": in_maint,
                    "Error_Rate_%": in_error,
                    "Hour": in_hour, "Minute": in_minute, "Day": in_day, "Month": in_month,
                }
                X_input = pd.DataFrame([row])[feature_names]
                X_scaled = scaler.transform(X_input)
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                label = le_target.inverse_transform([pred])[0]
                badge_cls = label.lower()
                st.markdown(f'<span class="badge {badge_cls}">Predicted Efficiency: {label}</span>', unsafe_allow_html=True)
                proba_df = pd.DataFrame({"Class": le_target.classes_, "Probability": proba})
                fig = px.bar(proba_df, x="Class", y="Probability", color="Class",
                              color_discrete_map=PALETTE, title="Class Probabilities")
                st.plotly_chart(style_fig(fig, height=320), use_container_width=True)

        with tab2:
            sample_n = min(5000, len(df))
            sample_df = df.sample(sample_n, random_state=42) if len(df) > sample_n else df.copy()
            X_eval = sample_df.rename(columns=rename_back)
            X_eval["Operation_Mode"] = le_mode.transform(X_eval["Operation_Mode"])
            X_eval = X_eval[feature_names]
            X_eval_scaled = scaler.transform(X_eval)
            preds = model.predict(X_eval_scaled)
            pred_labels = le_target.inverse_transform(preds)
            actual_labels = sample_df["Efficiency_Status"].astype(str).values
            accuracy = (pred_labels == actual_labels).mean()

            mc1, mc2 = st.columns(2)
            mc1.markdown(kpi_card("Accuracy on Filtered Sample", f"{accuracy*100:.1f}%", "green"), unsafe_allow_html=True)
            mc2.markdown(kpi_card("Sample Size", f"{sample_n:,}", "indigo"), unsafe_allow_html=True)

            cm = pd.crosstab(pd.Series(actual_labels, name="Actual"),
                               pd.Series(pred_labels, name="Predicted"))
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix")
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    with st.expander("📥 Download Executive Summary"):
        st.download_button("Download KPI Table (CSV)", kpi_table.to_csv(index=False).encode(),
                             "executive_kpi_summary.csv", "text/csv")
        st.download_button("Download Filtered Dataset (CSV)", df.to_csv(index=False).encode(),
                             "filtered_manufacturing_data.csv", "text/csv")
