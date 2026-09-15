import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import os

st.set_page_config(page_title="Drive Failure Risk Dashboard", layout="wide", page_icon="💾")
st.markdown(
    """
    <style>

    /* Main dashboard background */
    .stApp {
        background-color: #EAF7F1;
    }

    /* Left sidebar - same green */
    section[data-testid="stSidebar"] {
        background-color: #EAF7F1;
    }

    /* Top Streamlit header - same green */
    header[data-testid="stHeader"] {
        background-color: #EAF7F1;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
data = pd.read_csv("dashboard_data.csv")

metrics = {}
if os.path.exists("dashboard_metrics.json"):
    with open("dashboard_metrics.json") as f:
        metrics = json.load(f)

cost_curve = None
if os.path.exists("cost_curve_data.csv"):
    cost_curve = pd.read_csv("cost_curve_data.csv")

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("💾 Hard Drive Failure Prediction Dashboard")
st.markdown(
    """
    This tool predicts which hard drives are **likely to fail in the next 7 days**,
    using real sensor data (called SMART data) that every hard drive reports about its own health —
    things like bad sectors, temperature, and error counts.

    A machine learning model (XGBoost) was trained on historical Backblaze data center drives
    to learn the warning signs that show up *before* a drive actually fails.
    """
)

st.divider()

# ---------------------------------------------------------
# Plain-English performance summary
# ---------------------------------------------------------
st.subheader("📊 How good is this model, in plain terms?")

col1, col2, col3, col4 = st.columns(4)
recall_pct = metrics.get("model_recall", 0.62) * 100
roc_auc = metrics.get("model_roc_auc", 0.78)
threshold_pct = metrics.get("chosen_threshold", 0.36) * 100
total_failures = metrics.get("total_test_failures", "—")

col1.metric("Failures Caught", f"{recall_pct:.0f}%", help="Out of all drives that actually failed, this is the percentage the model correctly flagged in advance.")
col2.metric("Model Quality Score", f"{roc_auc:.2f}", help="ROC-AUC score, from 0.5 (random guessing) to 1.0 (perfect). 0.81 means the model is genuinely learning useful patterns.")
col3.metric("Alert Threshold", f"{threshold_pct:.0f}%", help="A drive is flagged 'at risk' once its predicted failure probability crosses this percentage.")
col4.metric("Real Failures (test period)", total_failures, help="Number of drives that actually failed during the time period used to test this model.")

st.info(
    f"**In plain words:** this model correctly identifies about **{recall_pct:.0f} out of every 100** drives "
    "that are actually going to fail, giving operations teams a heads-up before it happens. "
    "It isn't perfect — no failure-prediction system is — but it turns a completely blind guess into "
    "a meaningfully informed one.",
    icon="💡"
)

st.divider()

# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("🔍 Filter the Drive List")

selected_models = st.sidebar.multiselect(
    "Drive model",
    options=sorted(data["model"].unique()),
    default=None,
    help="Show only specific hard drive models"
)

risk_threshold = st.sidebar.slider(
    "Minimum risk to display",
    min_value=0, max_value=100, value=10,
    format="%d%%",
    help="Only show drives with at least this much predicted failure risk"
) / 100

filtered = data.copy()
if selected_models:
    filtered = filtered[filtered["model"].isin(selected_models)]
filtered = filtered[filtered["failure_probability"] >= risk_threshold]
filtered = filtered.sort_values("failure_probability", ascending=False)

st.sidebar.markdown("---")
st.sidebar.metric("Drives shown", len(filtered))
st.sidebar.metric("Total drives in system", len(data))

# ---------------------------------------------------------
# Main at-risk table
# ---------------------------------------------------------
st.subheader(f"🚨 At-Risk Drives — {len(filtered)} drives need attention")

if len(filtered) == 0:
    st.success("No drives currently meet this risk threshold. Try lowering the slider on the left.")
else:
    display_table = filtered[["serial_number", "model", "date", "failure_probability"]].copy()
    display_table["Risk Level"] = pd.cut(
        display_table["failure_probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["🟡 Moderate", "🟠 High", "🔴 Critical"]
    )
    display_table["failure_probability"] = (display_table["failure_probability"] * 100).round(1).astype(str) + "%"
    display_table.columns = ["Serial Number", "Drive Model", "Last Reading Date", "Failure Risk", "Risk Level"]

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    csv = display_table.to_csv(index=False)
    st.download_button("⬇️ Download this list as CSV", csv, "at_risk_drives.csv", "text/csv")

st.divider()

# ---------------------------------------------------------
# Risk distribution chart (Plotly, with click-to-highlight)
# ---------------------------------------------------------
st.subheader("📈 How risk is spread across all drives")
st.caption("Most drives should be low-risk. A healthy fleet looks like a tall bar near zero, with a small tail of risky drives. Click a bar to highlight it.")

# Build readable percentage bins instead of raw interval labels
bin_edges = np.linspace(0, 1, 11)
bin_labels = [f"{int(bin_edges[i]*100)}–{int(bin_edges[i+1]*100)}%" for i in range(len(bin_edges)-1)]
counts, _ = np.histogram(data["failure_probability"], bins=bin_edges)

# Warm color palette, from light gold to deep red-orange
warm_palette = ["#FFE0B2", "#FFCC80", "#FFB74D", "#FFA726", "#FF9800",
                 "#FB8C00", "#F57C00", "#EF6C00", "#E65100", "#D84315"]

selected_point = st.session_state.get("risk_chart", {}).get("selection", {}).get("point_indices", [])

colors = []
for i in range(len(counts)):
    if selected_point and i not in selected_point:
        colors.append("#F5F5F5")  # dimmed / light gray for non-selected
    else:
        colors.append(warm_palette[i])

fig_hist = go.Figure(
    data=[go.Bar(x=bin_labels, y=counts, marker_color=colors)]
)
fig_hist.update_layout(
    xaxis_title="Predicted Failure Risk Range",
    yaxis_title="Number of Drives",
    margin=dict(t=10, b=10),
    height=420,
)

st.plotly_chart(fig_hist, use_container_width=True, on_select="rerun", key="risk_chart")

st.divider()

# ---------------------------------------------------------
# Cost curve explanation (Phase 5 insight)
# ---------------------------------------------------------
if cost_curve is not None:
    st.subheader("💰 Why this alert threshold, and not another one?")
    st.markdown(
        """
        Every prediction system has a trade-off: catch more real failures, or avoid false alarms.
        The chart below shows the total estimated cost (missed failures + wasted early replacements)
        at every possible alert threshold.

        We didn't just pick the threshold that looked "cheapest" on paper — a purely cost-minimizing
        threshold turned out to catch almost no real failures, since failures are so rare that false
        alarms rack up fast even at low rates. Instead, we required the model to **catch at least 60%
        of real failures**, and picked the cheapest threshold that still met that bar.
        """
    )
    chosen_t = metrics.get("chosen_threshold", 0.45)

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=cost_curve["threshold"], y=cost_curve["cost"],
        mode="lines", line=dict(color="#E65100", width=3),
        name="Total Cost"
    ))
    fig_cost.add_vline(
        x=chosen_t, line_dash="dash", line_color="#D84315",
        annotation_text=f"Chosen threshold ({chosen_t:.2f})",
        annotation_position="top"
    )
    fig_cost.update_layout(
        xaxis_title="Alert Threshold",
        yaxis_title="Total Estimated Cost ($)",
        margin=dict(t=40, b=10),
        height=420,
    )
    st.plotly_chart(fig_cost, use_container_width=True)
    st.caption("Total estimated cost ($) at each possible alert threshold. The chosen threshold balances catching real failures against false-alarm cost.")

st.divider()
st.markdown(
    "<div style= 'text-align:center; color:#8a8a8a; font-size:1em;'>"
    "Developed by <b>Nuha Mushtaq</b> · <b>Tabassum Fathima</b> · <b>Yerra Shailaja</b>"
    "</div>",
    unsafe_allow_html=True
)
st.caption("Built on real Backblaze data center hard drive data (2020) · Model: XGBoost · Prediction window: 7 days")