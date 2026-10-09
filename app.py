from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.dashboard_core import (
    ANGLE_COL,
    EXPERIMENT_COL,
    GROUP_COL,
    MAIN_AXIS,
    SECONDARY_AXIS,
    attach_groups,
    axis_column,
    axis_label,
    geometry_summary,
    group_experiments,
    mean_pairwise_curve_distance,
    prediction_summary,
    variability_by_group,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
QRF_ROOT = ROOT / "src" / "pipeline" / "ml" / "qrf" / "results" / "models"
HGP_ROOT = ROOT / "src" / "pipeline" / "ml" / "hgp" / "results" / "models"
GEOMETRY_PATH = DATA / "processed" / "geometry.csv"
SETUP_PATH = DATA / "processed" / "processed_bending_setup.csv"
MACHINE_PATH = DATA / "processed" / "machine.csv"
MOVEMENT_PATH = DATA / "processed" / "movement.csv"
SENSOR_PATH = DATA / "processed" / "sensor.csv"
RF_RESULTS_PATH = DATA / "rf_augmented" / "final_model_results_raw.parquet"
SK_PREDICTIONS_PATH = ROOT / "src" / "pipeline" / "ml" / "sk" / "result" / "sk_prediction_details.csv"
SK_METRICS_PATH = ROOT / "src" / "pipeline" / "ml" / "sk" / "result" / "sk_global_metrics.csv"
BENDING_RANKING_PATH = (
    ROOT / "src" / "pipeline" / "ml" / "qrf" / "data" / "bending_setup_group_spread_feature_ranking.parquet"
)
MAX_ANGLE = 44

SOURCE_LABELS = {
    "real": "Real experimental data",
    "sensor_augmented_noise__time_wrapping__scaling__jittering": "Sensor-based augmentation",
    "within_group_interpolation_raw": "Within-group interpolation",
}
SOURCE_PATHS = {
    "sensor_augmented_noise__time_wrapping__scaling__jittering": DATA
    / "rf_augmented"
    / "final_geometry_sensor_augmented_noise__time_wrapping__scaling__jittering.parquet",
    "within_group_interpolation_raw": DATA
    / "rf_augmented"
    / "final_geometry_within_group_interpolation_raw.parquet",
}
RANK_ONE_SPLIT = {"main": 1, "secondary": 8}
AXIS_OPTIONS = {"Main": "main", "Secondary": "secondary"}
PROCESS_PARAMETERS = [
    "Collet boost",
    "Pressure-die distance",
    "Mandrel retraction timing",
    "Pressure-die boost",
    "Clamp-die lateral position",
    "Mandrel position",
    "Pressure-die lateral position",
]
METRIC_LABELS = {
    "coverage_percent": "Coverage [%]",
    "coverage_error_percent": "Calibration error [%]",
    "mean_interval_width": "Mean interval width [mm]",
    "mae": "MAE [mm]",
    "rmse": "RMSE [mm]",
    "r2": "R2",
    "interval_score": "Interval score",
    "negative_log_predictive_density": "NLPD",
}


st.set_page_config(
    page_title="TubeBEND Thesis Dashboard",
    page_icon="",
    layout="wide",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --tb-blue: #2563eb;
          --tb-navy: #102033;
          --tb-muted: #64748b;
          --tb-border: #dbe5f0;
          --tb-bg: #f6f8fb;
        }
        .stApp { background: var(--tb-bg); color: var(--tb-navy); }
        section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--tb-border); }
        .main-title {
          font-size: 2.25rem; line-height: 1.12; font-weight: 760; color: var(--tb-navy);
          margin: 0.15rem 0 0.25rem;
        }
        .subtitle { color: var(--tb-muted); font-size: 1.02rem; margin-bottom: 1rem; }
        .card {
          background: #fff; border: 1px solid var(--tb-border); border-radius: 10px;
          padding: 1rem 1.05rem; box-shadow: 0 8px 28px rgba(15, 23, 42, 0.045);
          min-height: 100%;
        }
        .metric-card {
          background: #fff; border: 1px solid var(--tb-border); border-radius: 8px;
          padding: 0.8rem 0.9rem; min-height: 86px;
        }
        .metric-card .value { font-weight: 760; font-size: 1.45rem; color: var(--tb-blue); }
        .metric-card .label { color: var(--tb-muted); font-size: 0.86rem; }
        .accordion-head {
          background: #fff; border: 1px solid var(--tb-border); border-radius: 10px;
          padding: 0.8rem 0.9rem; margin: 0.42rem 0 0.1rem;
        }
        .step-dot {
          display: inline-flex; align-items: center; justify-content: center;
          width: 30px; height: 30px; border-radius: 999px; background: var(--tb-blue);
          color: #fff; font-weight: 720; margin-right: 0.65rem;
        }
        .step-title { font-weight: 720; color: var(--tb-navy); font-size: 1.03rem; }
        .step-desc { color: var(--tb-muted); font-size: 0.88rem; margin-left: 2.55rem; }
        div[data-testid="stButton"] > button {
          border-radius: 8px; border: 1px solid var(--tb-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
        <script>
        const doc = window.parent.document;
        const allowedKeys = new Set([
          "Tab", "Escape", "Enter", "ArrowUp", "ArrowDown",
          "ArrowLeft", "ArrowRight", "Home", "End"
        ]);

        function lockSelectboxInputs() {
          doc.querySelectorAll('div[data-baseweb="select"] input').forEach((input) => {
            input.readOnly = true;
            input.setAttribute("inputmode", "none");
            input.setAttribute("autocomplete", "off");

            if (input.dataset.selectionOnly === "true") {
              return;
            }
            input.dataset.selectionOnly = "true";
            input.addEventListener("keydown", (event) => {
              if (!allowedKeys.has(event.key)) {
                event.preventDefault();
              }
            }, true);
            input.addEventListener("beforeinput", (event) => event.preventDefault(), true);
            input.addEventListener("paste", (event) => event.preventDefault(), true);
          });
        }

        lockSelectboxInputs();
        new MutationObserver(lockSelectboxInputs).observe(doc.body, {
          childList: true,
          subtree: true
        });
        </script>
        """,
        height=0,
    )


@st.cache_data(show_spinner=False)
def read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


@st.cache_data(show_spinner=False)
def read_parquet(path: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_parquet(path, **kwargs)


@st.cache_data(show_spinner="Loading thesis datasets...")
def load_base_data() -> dict[str, pd.DataFrame]:
    geometry = read_csv(str(GEOMETRY_PATH))
    setup = read_csv(str(SETUP_PATH))
    observed = attach_groups(geometry, setup)
    return {"geometry": geometry, "setup": setup, "observed": observed}


@st.cache_data(show_spinner=False)
def load_signal(path: str, experiment_id: int, max_rows: int = 5000) -> pd.DataFrame:
    df = read_csv(path)
    df = df[df[EXPERIMENT_COL].astype(int).eq(int(experiment_id))].copy()
    if len(df) > max_rows:
        step = int(np.ceil(len(df) / max_rows))
        df = df.iloc[::step].copy()
    return df


@st.cache_data(show_spinner=False)
def load_augmented(source: str) -> pd.DataFrame:
    if source == "real":
        base = load_base_data()["observed"].copy()
        base["Synthetic"] = False
        base["group_id"] = base[GROUP_COL]
        return base
    return read_parquet(str(SOURCE_PATHS[source]))


@st.cache_data(show_spinner=False)
def load_rf_results() -> pd.DataFrame:
    return read_parquet(str(RF_RESULTS_PATH)) if RF_RESULTS_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sk_predictions() -> pd.DataFrame:
    return read_csv(str(SK_PREDICTIONS_PATH)) if SK_PREDICTIONS_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_sk_metrics() -> pd.DataFrame:
    return read_csv(str(SK_METRICS_PATH)) if SK_METRICS_PATH.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_ranking() -> pd.DataFrame:
    return read_parquet(str(BENDING_RANKING_PATH)) if BENDING_RANKING_PATH.exists() else pd.DataFrame()


def qrf_run_dir(source: str, axis_key: str) -> Path | None:
    split_index = RANK_ONE_SPLIT[axis_key]
    candidates = sorted(
        (QRF_ROOT / source / axis_key).glob(
            f"qrf_geometry_{source}_{axis_key}_split_{split_index:03d}_rank_*"
        )
    )
    return candidates[0] if candidates else None


def hgp_run_dir(source: str, axis_key: str) -> Path:
    return HGP_ROOT / source / axis_key


@st.cache_data(show_spinner=False)
def load_prediction_file(path: str) -> pd.DataFrame:
    pred = read_csv(path)
    if "y_median" not in pred.columns and "y_mean" in pred.columns:
        pred = pred.rename(columns={"y_mean": "y_median"})
    return pred


@st.cache_data(show_spinner=False)
def load_metric_file(path: str) -> pd.DataFrame:
    return read_csv(path)


def prediction_and_metrics(model: str, source: str, axis_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = qrf_run_dir(source, axis_key) if model == "QRF" else hgp_run_dir(source, axis_key)
    if run_dir is None or not run_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    pred_path = run_dir / "test_predictions.csv"
    metric_path = run_dir / "metrics.csv"
    pred = load_prediction_file(str(pred_path)) if pred_path.exists() else pd.DataFrame()
    metrics = load_metric_file(str(metric_path)) if metric_path.exists() else pd.DataFrame()
    return pred, metrics


@st.cache_data(show_spinner=False)
def collect_uncertainty_metrics() -> pd.DataFrame:
    rows = []
    for source in SOURCE_LABELS:
        for axis_key in ("main", "secondary"):
            for model in ("QRF", "HGP"):
                _, metrics = prediction_and_metrics(model, source, axis_key)
                if metrics.empty:
                    continue
                row = metrics.iloc[0].to_dict()
                row["Model"] = model
                row["source_key"] = source
                row["Training data"] = SOURCE_LABELS[source]
                row["Axis"] = axis_label(axis_key)
                rows.append(row)
    return pd.DataFrame(rows)


def sidebar() -> str:
    with st.sidebar:
        st.markdown("### TubeBEND")
        st.caption("Thesis research dashboard")
        return st.radio(
            "Navigation",
            ["Workflow", "Dataset", "Results"],
            label_visibility="collapsed",
        )


def header() -> None:
    st.markdown(
        '<div class="main-title">Uncertainty-Aware Machine Learning for Rotary Tube Bending</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Interactive visualization of the complete methodological workflow.</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(
        f'<div class="card"><b>{title}</b><br><span style="color:#64748b">{body}</span></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="value">{value}</div>
          <div class="label">{label}</div>
          <div style="color:#94a3b8;font-size:0.78rem">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def geometry_plot(df: pd.DataFrame, axis_col: str, title: str, synthetic_col: str | None = None) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    df = df[pd.to_numeric(df[ANGLE_COL], errors="coerce").between(0, MAX_ANGLE)].copy()
    for experiment_id, part in df.groupby(EXPERIMENT_COL):
        synthetic = bool(part[synthetic_col].iloc[0]) if synthetic_col and synthetic_col in part else False
        fig.add_trace(
            go.Scatter(
                x=part[ANGLE_COL],
                y=part[axis_col],
                mode="lines",
                name=("Synthetic" if synthetic else "Real") + f" {int(experiment_id)}",
                line=dict(
                    color="rgba(37,99,235,0.38)" if synthetic else "rgba(30,41,59,0.68)",
                    dash="dash" if synthetic else "solid",
                    width=1.5,
                ),
                legendgroup="synthetic" if synthetic else "real",
                showlegend=False,
            )
        )
    summary = geometry_summary(df, axis_col)
    if not summary.empty:
        fig.add_trace(
            go.Scatter(
                x=summary[ANGLE_COL],
                y=summary["mean"],
                mode="lines+markers",
                name="Mean profile",
                line=dict(color="#2563eb", width=3),
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Angle or distance",
        yaxis_title=axis_col,
        hovermode="x unified",
    )
    fig.update_xaxes(range=[0, MAX_ANGLE])
    return fig


def signal_plot(signal_df: pd.DataFrame, channels: list[str], normalize: bool) -> go.Figure:
    fig = go.Figure()
    if signal_df.empty:
        return fig
    x = signal_df["Time_[s]"] if "Time_[s]" in signal_df else signal_df.index
    for channel in channels:
        y = pd.to_numeric(signal_df[channel], errors="coerce")
        if normalize:
            std = y.std()
            y = (y - y.mean()) / std if pd.notna(std) and std else y - y.mean()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=channel))
    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=35, b=10),
        xaxis_title="Time [s]",
        yaxis_title="Normalized value" if normalize else "Native channel value",
        hovermode="x unified",
    )
    return fig


def prediction_interval_plot(pred: pd.DataFrame, model: str, title: str) -> go.Figure:
    if not pred.empty and ANGLE_COL in pred.columns:
        pred = pred[pd.to_numeric(pred[ANGLE_COL], errors="coerce").between(0, MAX_ANGLE)].copy()
    summary = prediction_summary(pred)
    fig = go.Figure()
    if summary.empty:
        return fig
    x_poly = pd.concat([summary[ANGLE_COL], summary[ANGLE_COL].iloc[::-1]])
    y_poly = pd.concat([summary["y_upper"], summary["y_lower"].iloc[::-1]])
    fig.add_trace(
        go.Scatter(
            x=x_poly,
            y=y_poly,
            fill="toself",
            fillcolor="rgba(37,99,235,0.16)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Prediction interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary[ANGLE_COL],
            y=summary["y_true"],
            mode="lines+markers",
            name="Measured geometry",
            line=dict(color="#1f2937", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=summary[ANGLE_COL],
            y=summary["y_center"],
            mode="lines",
            name=f"{model} central prediction",
            line=dict(color="#2563eb", width=3),
        )
    )
    outside = summary[
        (summary["y_true"] < summary["y_lower"]) | (summary["y_true"] > summary["y_upper"])
    ]
    if not outside.empty:
        fig.add_trace(
            go.Scatter(
                x=outside[ANGLE_COL],
                y=outside["y_true"],
                mode="markers",
                name="Outside interval",
                marker=dict(color="#dc2626", size=7, symbol="x"),
            )
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=470,
        margin=dict(l=10, r=10, t=52, b=10),
        xaxis_title="Angle or distance",
        yaxis_title=str(pred["target_column"].iloc[0]) if "target_column" in pred else "Geometry [mm]",
        hovermode="x unified",
    )
    fig.update_xaxes(range=[0, MAX_ANGLE])
    return fig


def render_data_preparation(data: dict[str, pd.DataFrame]) -> None:
    observed, setup = data["observed"], data["setup"]
    tabs = st.tabs(["Overview", "Process & Sensors", "Geometry Targets"])
    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            card(
                "Tube Bending Process",
                "Rotary draw bending setup represented by process parameters such as bend die, pressure die, mandrel, collet, and clamp-die positions. PDF figure extraction is prepared as a deployment script, but no extracted schematic is committed yet.",
            )
        with c2:
            card(
                "Process Measurements",
                "Machine torque/load channels, movement signals, and additional sensor load channels are loaded from the processed exports.",
            )
        with c3:
            card(
                "Measurement Orientations",
                "Geometry targets are represented as Main-axis and Secondary-axis profiles sampled over 45 measurement locations.",
            )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            metric_card("Prepared experiments", f"{observed[EXPERIMENT_COL].nunique():,}", "valid physical experiments")
        with m2:
            metric_card("Bending setup groups", f"{setup[GROUP_COL].nunique():,}")
        with m3:
            metric_card("Geometry points", f"{observed[ANGLE_COL].nunique():,}", "per target profile")
        with m4:
            metric_card("Machine load channels", "10")
        with m5:
            metric_card("Sensor load channels", "3")
        with m6:
            metric_card("Movement channels", "10")
    with tabs[1]:
        group_id = st.selectbox("Bending setup group", sorted(setup[GROUP_COL].astype(int)), key="prep_group")
        experiments = group_experiments(setup, int(group_id))
        experiment_id = st.selectbox("Experiment", experiments, index=None, placeholder="Select an experiment...")
        if experiment_id is None:
            st.info("Select a physical experiment to load process and sensor traces.")
        else:
            source_kind = st.radio("Signal source", ["Machine load", "Sensor load", "Movement"], horizontal=True)
            path = {"Machine load": MACHINE_PATH, "Sensor load": SENSOR_PATH, "Movement": MOVEMENT_PATH}[source_kind]
            signal_df = load_signal(str(path), int(experiment_id))
            channels = [c for c in signal_df.columns if c not in {EXPERIMENT_COL, "Time_[s]"}]
            selected = st.multiselect("Channels", channels, default=channels[: min(3, len(channels))])
            normalize = st.checkbox("Normalized comparison view", value=False)
            st.plotly_chart(signal_plot(signal_df, selected, normalize), width="stretch")
    with tabs[2]:
        axis_key = AXIS_OPTIONS[st.radio("Geometry axis", list(AXIS_OPTIONS), horizontal=True, key="prep_axis")]
        axis_col = axis_column(axis_key)
        group_id = st.selectbox("Group ID", sorted(setup[GROUP_COL].astype(int)), key="geometry_group")
        group_df = observed[observed[GROUP_COL].astype(int).eq(int(group_id))]
        st.plotly_chart(
            geometry_plot(group_df, axis_col, f"{axis_label(axis_key)} profiles for group {group_id}"),
            width="stretch",
        )


def render_augmentation(data: dict[str, pd.DataFrame]) -> None:
    setup = data["setup"]
    axis_key = AXIS_OPTIONS[st.radio("Geometry axis", list(AXIS_OPTIONS), horizontal=True, key="aug_axis")]
    axis_col = axis_column(axis_key)
    group_id = st.selectbox("Bending setup group", sorted(setup[GROUP_COL].astype(int)), key="aug_group")
    method = st.selectbox("Augmentation method", list(SOURCE_LABELS), format_func=lambda x: SOURCE_LABELS[x])
    aug = load_augmented(method)
    group_col = "group_id" if "group_id" in aug.columns else GROUP_COL
    group_df = aug[aug[group_col].astype(int).eq(int(group_id))].copy()
    if "Synthetic" in group_df:
        real = group_df[~group_df["Synthetic"].astype(bool)]
        synthetic = group_df[group_df["Synthetic"].astype(bool)]
        keep_synth_ids = sorted(synthetic[EXPERIMENT_COL].dropna().unique())[:6]
        plot_df = pd.concat([real, synthetic[synthetic[EXPERIMENT_COL].isin(keep_synth_ids)]])
    else:
        plot_df = group_df
    st.plotly_chart(
        geometry_plot(plot_df, axis_col, f"{SOURCE_LABELS[method]}: group {group_id}", synthetic_col="Synthetic"),
        width="stretch",
    )
    st.markdown(
        """
        <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0.65rem;margin-top:0.25rem">
          <div class="metric-card" style="min-height:58px;padding:0.55rem 0.65rem">
            <div class="value" style="font-size:1.05rem">315</div>
            <div class="label" style="font-size:0.74rem">Real prepared profiles</div>
          </div>
          <div class="metric-card" style="min-height:58px;padding:0.55rem 0.65rem">
            <div class="value" style="font-size:1.05rem">1,500</div>
            <div class="label" style="font-size:0.74rem">Profiles per augmented dataset</div>
          </div>
          <div class="metric-card" style="min-height:58px;padding:0.55rem 0.65rem">
            <div class="value" style="font-size:1.05rem">1,185</div>
            <div class="label" style="font-size:0.74rem">Additional generated profiles</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Each augmentation strategy independently expands the dataset to 1,500 profiles; the two synthetic strategies are not one combined dataset.")


def render_geometry_prediction(data: dict[str, pd.DataFrame]) -> None:
    pred = load_sk_predictions()
    metrics = load_sk_metrics()
    rf_results = load_rf_results()
    if pred.empty:
        st.warning("No saved geometry-prediction detail artifact was found.")
        return
    axis_target = st.radio("Target", [MAIN_AXIS, SECONDARY_AXIS], horizontal=True, key="rf_axis")
    groups = sorted(pred[pred["target"].eq(axis_target)][GROUP_COL].astype(int).unique())
    group_id = st.selectbox("Group ID", groups, key="rf_group")
    group_pred = pred[(pred["target"].eq(axis_target)) & (pred[GROUP_COL].astype(int).eq(int(group_id)))]
    experiment_id = st.selectbox("Experiment", sorted(group_pred[EXPERIMENT_COL].astype(int).unique()), key="rf_exp")
    exp = group_pred[group_pred[EXPERIMENT_COL].astype(int).eq(int(experiment_id))]
    exp = exp[pd.to_numeric(exp["Angle[degree]"], errors="coerce").between(0, MAX_ANGLE)].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=exp["Angle[degree]"], y=exp["y_true"], mode="lines+markers", name="Measured", line=dict(color="#1f2937", width=3)))
    fig.add_trace(go.Scatter(x=exp["Angle[degree]"], y=exp["y_pred_mean"], mode="lines", name="Predicted", line=dict(color="#2563eb", width=3, dash="dash")))
    fig.update_layout(template="plotly_white", height=450, title="Measured vs predicted geometry", xaxis_title="Angle [degree]", yaxis_title=axis_target)
    fig.update_xaxes(range=[0, MAX_ANGLE])
    st.plotly_chart(fig, width="stretch")
    if not metrics.empty:
        st.dataframe(metrics[metrics["target"].eq(axis_target)], width="stretch", hide_index=True)
    if not rf_results.empty:
        row = rf_results.iloc[0]
        st.caption(
            f"RF thesis pipeline reference: Main-axis R2={row.get('r2_main'):.3f}, MAE={row.get('mae_main'):.4f} mm; "
            f"Secondary-axis R2={row.get('r2_secondary'):.3f}, MAE={row.get('mae_secondary'):.4f} mm."
        )


def render_uncertainty() -> None:
    st.markdown("#### QRF vs HGP prediction intervals")
    c1, c2, c3 = st.columns([1, 1.4, 1.2])
    axis_key = AXIS_OPTIONS[c1.selectbox("Geometry axis", list(AXIS_OPTIONS), key="uq_axis")]
    source = c2.selectbox("Training data source", list(SOURCE_LABELS), format_func=lambda x: SOURCE_LABELS[x], key="uq_source")
    qrf_pred, qrf_metrics = prediction_and_metrics("QRF", source, axis_key)
    hgp_pred, hgp_metrics = prediction_and_metrics("HGP", source, axis_key)
    group_options = sorted(set(qrf_pred.get(GROUP_COL, pd.Series(dtype=int)).dropna().astype(int)).union(set(hgp_pred.get(GROUP_COL, pd.Series(dtype=int)).dropna().astype(int))))
    group_id = c3.selectbox("Evaluation group", group_options, index=0 if group_options else None, placeholder="No exported group")
    if group_id is not None:
        qrf_group = qrf_pred[qrf_pred[GROUP_COL].astype(int).eq(int(group_id))] if not qrf_pred.empty else pd.DataFrame()
        hgp_group = hgp_pred[hgp_pred[GROUP_COL].astype(int).eq(int(group_id))] if not hgp_pred.empty else pd.DataFrame()
        left, right = st.columns(2)
        left.plotly_chart(prediction_interval_plot(qrf_group, "QRF", "Quantile Regression Forest"), width="stretch")
        right.plotly_chart(prediction_interval_plot(hgp_group, "HGP", "Heteroscedastic Gaussian Process"), width="stretch")
    metric_rows = []
    for model, metrics in (("QRF", qrf_metrics), ("HGP", hgp_metrics)):
        if metrics.empty:
            continue
        row = metrics.iloc[0].to_dict()
        row["Model"] = model
        metric_rows.append(row)
    if metric_rows:
        compare = pd.DataFrame(metric_rows)
        cols = ["Model", "coverage_percent", "coverage_error_percent", "mean_interval_width", "mae", "rmse", "r2", "interval_score", "negative_log_predictive_density"]
        visible = [c for c in cols if c in compare.columns]
        st.dataframe(compare[visible].rename(columns=METRIC_LABELS), width="stretch", hide_index=True)


def render_variability(data: dict[str, pd.DataFrame]) -> None:
    observed, setup = data["observed"], data["setup"]
    axis_key = AXIS_OPTIONS[st.radio("Geometry axis", list(AXIS_OPTIONS), horizontal=True, key="var_axis")]
    axis_col = axis_column(axis_key)
    parameter = st.selectbox("Bending parameter", PROCESS_PARAMETERS, key="var_param")
    group_id = st.selectbox("Group ID", sorted(setup[GROUP_COL].astype(int)), key="var_group")
    group_df = observed[observed[GROUP_COL].astype(int).eq(int(group_id))]
    st.plotly_chart(geometry_plot(group_df, axis_col, f"Physical repetitions in group {group_id}"), width="stretch")
    distance = mean_pairwise_curve_distance(group_df, axis_col)
    if distance is None:
        st.info("This group has fewer than two complete physical repetitions, so pairwise variability is not estimable.")
    else:
        st.metric("Mean pairwise Euclidean distance", f"{distance:.3f}", "45-point geometry profiles")
    assoc = variability_by_group(observed, setup, axis_col, parameter).dropna(subset=["mean_pairwise_distance", parameter])
    fig = px.scatter(assoc, x=parameter, y="mean_pairwise_distance", size="n_experiments", hover_data=[GROUP_COL], template="plotly_white", title=f"Within-group variability vs {parameter}")
    st.plotly_chart(fig, width="stretch")
    ranking = load_ranking()
    if not ranking.empty:
        st.dataframe(ranking[ranking["geometry_type"].eq(axis_key)][["rank", "feature", "permutation_importance_mean", "rf_impurity_importance"]], width="stretch", hide_index=True)


WORKFLOW_STEPS = [
    ("Data Preparation", "Explore the original TubeBEND dataset, sensor/process measurements, geometry representation, and preprocessing operations."),
    ("Data Augmentation", "Explore synthetic geometry generation through sensor-based augmentation and within-group interpolation."),
    ("Geometry Prediction", "Explore measured and predicted tube geometry using saved prediction artifacts and RF reference metrics."),
    ("Uncertainty Estimation", "Compare Quantile Regression Forests and Heteroscedastic Gaussian Processes for prediction interval estimation."),
]


def render_workflow(data: dict[str, pd.DataFrame]) -> None:
    if "open_step" not in st.session_state:
        st.session_state.open_step = 0
    for i, (title, desc) in enumerate(WORKFLOW_STEPS):
        st.markdown(
            f'<div class="accordion-head"><span class="step-dot">{i + 1}</span><span class="step-title">{title}</span><div class="step-desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button(("Collapse" if st.session_state.open_step == i else "Open") + f" Step {i + 1}", key=f"step_button_{i}"):
            st.session_state.open_step = i
        if st.session_state.open_step == i:
            if i == 0:
                render_data_preparation(data)
            elif i == 1:
                render_augmentation(data)
            elif i == 2:
                render_geometry_prediction(data)
            else:
                render_uncertainty()


def render_dataset_page(data: dict[str, pd.DataFrame]) -> None:
    observed, setup = data["observed"], data["setup"]
    cols = st.columns(4)
    cols[0].metric("Experiments", observed[EXPERIMENT_COL].nunique())
    cols[1].metric("Groups", setup[GROUP_COL].nunique())
    cols[2].metric("Geometry rows", f"{len(observed):,}")
    cols[3].metric("Measurement points", observed[ANGLE_COL].nunique())
    st.dataframe(setup, width="stretch", hide_index=True)


def render_results_page() -> None:
    metric_df = collect_uncertainty_metrics()
    if metric_df.empty:
        st.warning("No uncertainty metrics were found.")
        return
    common_metrics = []
    for metric in METRIC_LABELS:
        if metric not in metric_df.columns:
            continue
        complete_for_both_models = (
            metric_df
            .dropna(subset=[metric])
            .groupby(["Training data", "Axis"])["Model"]
            .agg(lambda values: {"QRF", "HGP"}.issubset(set(values)))
        )
        if not complete_for_both_models.empty and bool(complete_for_both_models.all()):
            common_metrics.append(metric)

    if not common_metrics:
        st.warning("No metric is available for both QRF and HGP across the exported result groups.")
        return

    metric = st.selectbox(
        "Result metric",
        common_metrics,
        format_func=lambda x: METRIC_LABELS[x],
    )
    plot_df = metric_df.dropna(subset=[metric]).copy()
    st.caption(
        "Each bar is evaluated against the held-out measured geometry. "
        "The x-axis shows the training data source; colors distinguish geometry axes."
    )
    colors = {
        "Main-axis": "#2563eb",
        "Secondary-axis": "#14b8a6",
    }
    left, right = st.columns(2)
    for model, column in [("QRF", left), ("HGP", right)]:
        model_df = plot_df[plot_df["Model"].eq(model)].copy()
        fig = px.bar(
            model_df,
            x="Training data",
            y=metric,
            color="Axis",
            barmode="group",
            template="plotly_white",
            title=f"{model} - {METRIC_LABELS[metric]}",
            color_discrete_map=colors,
        )
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=58, b=10),
            legend_title_text="Geometry axis",
            yaxis_title=METRIC_LABELS[metric],
            xaxis_title="Training data source",
        )
        fig.update_xaxes(tickangle=-30)
        column.plotly_chart(fig, width="stretch")


def main() -> None:
    inject_css()
    nav = sidebar()
    header()
    data = load_base_data()
    if nav == "Workflow":
        render_workflow(data)
    elif nav == "Dataset":
        render_dataset_page(data)
    else:
        render_results_page()


if __name__ == "__main__":
    main()
