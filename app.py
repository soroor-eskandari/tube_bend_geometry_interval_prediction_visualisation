"""TubeBEND thesis dashboard.

Uncertainty-aware prediction intervals for rotary draw bending geometry.
Single-file Streamlit app: run with `streamlit run app.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ML = ROOT / "src" / "pipeline" / "ml"

ANGLE = "Angle[degree]ORDistance[mm]"
GROUP = "Group_ID"
EXP = "Experiment_ID"
MAIN = "Main-axis [mm]"
SECONDARY = "Secondary-axis [mm]"
MAX_ANGLE = 44
SYNTHETIC_ID_START = 1_000_000  # generated experiments get IDs >= this
NOMINAL = 90.0

AXES = {
    "Main axis": {"key": "main", "col": MAIN, "split": 1, "held_out": "Collet boost = 0.85"},
    "Secondary axis": {"key": "secondary", "col": SECONDARY, "split": 8, "held_out": "Mandrel retraction timing = 2"},
}
SENSOR_KEY = "sensor_augmented_noise__time_wrapping__scaling__jittering"
INTERP_KEY = "within_group_interpolation_raw"
SOURCES = {
    "real": {"label": "Real data", "color": "#334155"},
    INTERP_KEY: {"label": "Within-group interpolation", "color": "#0d9488"},
    SENSOR_KEY: {"label": "Sensor augmentation", "color": "#7c3aed"},
}
MODELS = {
    "QRF": {"label": "Quantile Regression Forest", "color": "#2563eb", "symbol": "circle"},
    "HGP": {"label": "Heteroscedastic Gaussian Process", "color": "#ea580c", "symbol": "diamond"},
}
MEASURED = "#0f172a"

PARAMETERS = [
    "Collet boost",
    "Pressure-die distance",
    "Mandrel retraction timing",
    "Pressure-die boost",
    "Clamp-die lateral position",
    "Mandrel position",
]
# Thesis Table 4.20 (held-out permutation importance) and Table 4.21 (adjusted direction).
VARIABILITY_RANKING = {
    "main": {
        "Collet boost": 0.028675,
        "Pressure-die distance": 0.023374,
        "Mandrel retraction timing": 0.004738,
        "Pressure-die boost": 0.002199,
        "Clamp-die lateral position": 0.000592,
        "Mandrel position": -0.000102,
    },
    "secondary": {
        "Pressure-die distance": 0.029940,
        "Collet boost": 0.004572,
        "Mandrel retraction timing": 0.001579,
        "Mandrel position": 0.000982,
        "Clamp-die lateral position": -0.000318,
        "Pressure-die boost": -0.000342,
    },
}
VARIABILITY_DIRECTION = {
    "main": {
        "Collet boost": "positive",
        "Pressure-die distance": "approximately neutral",
        "Mandrel retraction timing": "negative",
        "Pressure-die boost": "weak positive",
        "Clamp-die lateral position": "positive",
        "Mandrel position": "negative",
    },
    "secondary": {
        "Collet boost": "negative",
        "Pressure-die distance": "positive",
        "Mandrel retraction timing": "positive",
        "Pressure-die boost": "approximately neutral",
        "Clamp-die lateral position": "negative",
        "Mandrel position": "negative",
    },
}
# The twelve process signals used as model inputs (thesis Fig. 3.8).
SIGNALS = {
    "machine": {
        "MACHINE_BEND-DIE_LATERAL_Max_Torque_[%]": "Bend-die lateral torque [%]",
        "MACHINE_BEND-DIE_ROTATING_Max_Torque_[%]": "Bend-die rotating torque [%]",
        "MACHINE_CLAMP-DIE_LATERAL_Max_Torque_[%]": "Clamp-die lateral torque [%]",
        "MACHINE_COLLET_AXIAL_Max_Torque_[%]": "Collet axial torque [%]",
        "MACHINE_MANDREL_AXIAL_Max_Torque_[%]": "Mandrel axial torque [%]",
        "MACHINE_PRESSURE-DIE_LATERAL_Max_Torque_[%]": "Pressure-die lateral torque [%]",
    },
    "movement": {
        "BEND-DIE_ROTATING_Angle_[°]": "Bend-die angle [°]",
        "BEND-DIE_LATERAL_Movement_[mm]": "Bend-die lateral [mm]",
        "CLAMP-DIE_LATERAL_Movement_[mm]": "Clamp-die lateral [mm]",
        "COLLET_AXIAL_Movement_[mm]": "Collet axial [mm]",
        "MANDREL_AXIAL_Movement_[mm]": "Mandrel axial [mm]",
        "PRESSURE-DIE_AXIAL_Movement_[mm]": "Pressure-die axial [mm]",
    },
}

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TubeBEND · Prediction Intervals",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

pio.templates["thesis"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e2e8f0", zeroline=False, linecolor="#cbd5e1", ticks=""),
        yaxis=dict(gridcolor="#e2e8f0", zeroline=False, linecolor="#cbd5e1", ticks=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font_size=12),
        margin=dict(l=8, r=8, t=48, b=8),
    )
)
pio.templates.default = "plotly_white+thesis"
PLOT_CONFIG = {"displayModeBar": False}

st.markdown(
    """
    <style>
    .block-container { padding-top: 3.5rem; padding-bottom: 3rem; max-width: 1400px; }
    .tb-title { font-size: 1.9rem; font-weight: 700; line-height: 1.15; color: #0f172a; margin: 0; }
    .tb-sub { color: #64748b; font-size: 0.98rem; margin-top: 0.35rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    .stTabs [aria-selected="true"] { color: #2563eb !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: #2563eb !important; }
    button[kind="segmented_controlActive"], button[kind="pillsActive"] {
        background: #eff6ff !important; border-color: #2563eb !important; color: #1d4ed8 !important; }
    .tb-kpi { border: 1px solid #e2e8f0; border-radius: 10px; padding: .8rem 1rem; background: #fff; height: 100%; }
    .tb-kpi .v { font-size: 1.6rem; font-weight: 700; color: #0f172a; line-height: 1.2; }
    .tb-kpi .l { font-size: .82rem; color: #64748b; margin-bottom: .2rem; }
    .tb-kpi .n { font-size: .8rem; color: #94a3b8; margin-top: .15rem; }
    .tb-kpi .n.good { color: #b91c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (all cached)
# ---------------------------------------------------------------------------


def parse_ids(value: object) -> list[int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [int(v) for v in value] if value is not None else []


@st.cache_data(show_spinner=False)
def load_setup() -> pd.DataFrame:
    setup = pd.read_parquet(DATA / "processed" / "processed_bending_setup.parquet")
    setup[GROUP] = setup[GROUP].astype(int)
    setup["experiments"] = setup["Experiment_Number"].map(parse_ids)
    return setup


@st.cache_data(show_spinner=False)
def load_geometry(source: str) -> pd.DataFrame:
    """Geometry profiles with columns EXP, GROUP, ANGLE, MAIN, SECONDARY, synthetic."""
    if source == "real":
        geometry = pd.read_parquet(DATA / "processed" / "geometry.parquet")
        mapping = load_setup()[[GROUP, "experiments"]].explode("experiments")
        mapping = mapping.rename(columns={"experiments": EXP}).astype({EXP: int})
        df = geometry.merge(mapping, on=EXP, how="inner")
        df["synthetic"] = False
    else:
        df = pd.read_parquet(DATA / "rf_augmented" / f"final_geometry_{source}.parquet")
        df = df.rename(columns={"group_id": GROUP, "Synthetic": "synthetic"})
    df = df[[EXP, GROUP, ANGLE, MAIN, SECONDARY, "synthetic"]].copy()
    df[GROUP] = df[GROUP].astype(int)
    df["synthetic"] = df["synthetic"].astype(bool)
    return df[df[ANGLE].between(0, MAX_ANGLE)]


def prediction_dir(model: str, source: str, axis_key: str, split: int) -> Path:
    if model == "HGP":
        return ML / "hgp" / "results" / "models" / source / axis_key
    root = ML / "qrf" / "results" / "models" / source / axis_key
    matches = sorted(root.glob(f"qrf_geometry_{source}_{axis_key}_split_{split:03d}_rank_*"))
    return matches[0] if matches else root


@st.cache_data(show_spinner=False)
def load_predictions(model: str, source: str, axis_key: str, split: int) -> pd.DataFrame:
    path = prediction_dir(model, source, axis_key, split) / "test_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    pred = pd.read_parquet(path)
    pred["y_center"] = pred["y_median"] if "y_median" in pred else pred["y_mean"]
    pred[GROUP] = pred[GROUP].astype(int)
    return pred[pred[ANGLE].between(0, MAX_ANGLE)]


def interval_score(y: pd.Series, lo: pd.Series, hi: pd.Series, alpha: float = 0.1) -> float:
    penalty = (2 / alpha) * ((lo - y).clip(lower=0) + (y - hi).clip(lower=0))
    return float(((hi - lo) + penalty).mean())


def gaussian_nlpd(y: pd.Series, center: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    """NLPD of a Gaussian approximation of a 90% interval (thesis Eq. 4.1)."""
    sigma = ((hi - lo) / (2 * 1.64485)).clip(lower=1e-9)
    return float((0.5 * np.log(2 * np.pi * sigma**2) + (y - center) ** 2 / (2 * sigma**2)).mean())


@st.cache_data(show_spinner=False)
def load_metrics() -> pd.DataFrame:
    """One row per model x training source x axis, as reported in the thesis."""
    rows = []
    for axis_name, axis in AXES.items():
        for source, source_meta in SOURCES.items():
            for model in MODELS:
                run = prediction_dir(model, source, axis["key"], axis["split"])
                if not (run / "metrics.parquet").exists():
                    continue
                m = pd.read_parquet(run / "metrics.parquet").iloc[0]
                # Unclipped predictions, so derived scores match the thesis tables exactly.
                pred = pd.read_parquet(run / "test_predictions.parquet")
                center = pred["y_median"] if "y_median" in pred else pred["y_mean"]
                y, lo, hi, c = pred["y_true"], pred["y_lower"], pred["y_upper"], center
                rows.append(
                    {
                        "Axis": axis_name,
                        "Model": model,
                        "source": source,
                        "Training data": source_meta["label"],
                        "Coverage [%]": m["coverage_percent"],
                        "Calibration error [pp]": abs(m["coverage_percent"] - NOMINAL),
                        "Mean PI width [mm]": m["mean_interval_width"],
                        "MAE [mm]": m["mae"],
                        "RMSE [mm]": m["rmse"],
                        "Interval score": m.get("interval_score", interval_score(y, lo, hi)),
                        "NLPD": m.get("negative_log_predictive_density", gaussian_nlpd(y, c, lo, hi)),
                    }
                )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_rf_summary() -> dict[str, float]:
    path = DATA / "rf_augmented" / "final_model_results_raw.parquet"
    if not path.exists():
        return {}
    row = pd.read_parquet(path).iloc[0]
    return {k: float(row[k]) for k in ("r2_main", "r2_secondary", "test_r2_main", "test_r2_secondary")}


@st.cache_data(show_spinner=False)
def load_signals(experiment_id: int, max_points: int = 1200) -> pd.DataFrame:
    frames = []
    for name, columns in SIGNALS.items():
        df = pd.read_parquet(
            DATA / "processed" / f"{name}.parquet",
            columns=[EXP, "Time_[s]", *columns],
            filters=[(EXP, "==", int(experiment_id))],
        )
        frames.append(df.set_index("Time_[s]").drop(columns=EXP))
    signals = pd.concat(frames, axis=1).sort_index()
    step = max(1, int(np.ceil(len(signals) / max_points)))
    return signals.iloc[::step]


@st.cache_data(show_spinner=False)
def within_group_spread(source: str, axis_col: str) -> pd.Series:
    """Per group: pointwise std across profiles, averaged over the 45 points.

    Real data uses the physical repetitions; augmented sources use only the
    generated profiles, so the comparison isolates what augmentation adds.
    """
    df = load_geometry(source)
    df = df[df["synthetic"]] if source != "real" else df
    counts = df.groupby(GROUP)[EXP].nunique()
    df = df[df[GROUP].isin(counts[counts >= 2].index)]
    return df.groupby([GROUP, ANGLE])[axis_col].std().groupby(level=0).mean()


@st.cache_data(show_spinner=False)
def pairwise_variability(axis_col: str) -> pd.DataFrame:
    """Mean pairwise Euclidean distance between physical repetitions, per group."""
    rows = []
    for group_id, part in load_geometry("real").groupby(GROUP):
        curves = part.pivot_table(index=EXP, columns=ANGLE, values=axis_col).dropna(axis=1).to_numpy()
        if len(curves) < 2:
            continue
        diff = curves[:, None, :] - curves[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))[np.triu_indices(len(curves), k=1)]
        rows.append({GROUP: group_id, "S_g": dist.mean(), "runs": len(curves)})
    out = pd.DataFrame(rows)
    return out.merge(load_setup()[[GROUP, *PARAMETERS]], on=GROUP)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def kpi(column, label: str, value: str, note: str = "", note_class: str = "") -> None:
    column.markdown(
        f'<div class="tb-kpi"><div class="l">{label}</div><div class="v">{value}</div>'
        f'<div class="n {note_class}">{note}</div></div>',
        unsafe_allow_html=True,
    )


def add_profiles(fig, df, axis_col, color, row=None, col=None, width=1.6, dash="solid", name=None, legend=False):
    for i, (exp_id, part) in enumerate(df.groupby(EXP)):
        part = part.sort_values(ANGLE)
        fig.add_trace(
            go.Scatter(
                x=part[ANGLE], y=part[axis_col], mode="lines",
                line=dict(color=color, width=width, dash=dash),
                name=name or f"Experiment {exp_id}", legendgroup=name,
                showlegend=legend and i == 0,
                hovertemplate=f"Experiment {exp_id}<br>angle %{{x}}<br>%{{y:.3f}} mm<extra></extra>",
            ),
            row=row, col=col,
        )


def add_interval(fig, pred, model, row, col, legend):
    summary = pred.groupby(ANGLE)[["y_lower", "y_upper", "y_center"]].mean().reset_index()
    color = MODELS[model]["color"]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([summary[ANGLE], summary[ANGLE][::-1]]),
            y=pd.concat([summary["y_upper"], summary["y_lower"][::-1]]),
            fill="toself", fillcolor=rgba(color, 0.18), line=dict(width=0),
            name=f"{model} 90% interval", legendgroup=f"{model}-band", showlegend=legend, hoverinfo="skip",
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=summary[ANGLE], y=summary["y_center"], mode="lines", line=dict(color=color, width=2.5),
            name=f"{model} prediction", legendgroup=f"{model}-line", showlegend=legend,
            hovertemplate=f"{model}<br>angle %{{x}}<br>%{{y:.3f}} mm<extra></extra>",
        ),
        row=row, col=col,
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def render_overview(metrics: pd.DataFrame) -> None:
    setup, real = load_setup(), load_geometry("real")
    rf = load_rf_summary()

    k = st.columns(5)
    kpi(k[0], "Bending experiments", f"{real[EXP].nunique()}", f"{setup[GROUP].nunique()} nominal setup groups")
    kpi(k[1], "Experiments per group", "1 – 6", "few physical repetitions")
    kpi(k[2], "Profile resolution", f"{real[ANGLE].nunique()} points", "Main + Secondary axis")
    if rf:
        kpi(k[3], "RF geometry R² (Main / Sec.)", f"{rf['r2_main']:.3f} / {rf['r2_secondary']:.3f}",
            f"held-out groups: {rf['test_r2_main']:.2f} / {rf['test_r2_secondary']:.2f}")
    n_aug = load_geometry(INTERP_KEY)[EXP].nunique()
    kpi(k[4], "Profiles after augmentation", f"{n_aug:,}", f"+{n_aug - real[EXP].nunique():,} synthetic per strategy")
    st.write("")

    st.markdown("#### Coverage vs. interval width — every model and training source at a glance")
    st.caption(
        "Each point is one model trained on one data source and evaluated on held-out setup groups. "
        "A good interval sits **on the 90% line and as far left as possible** (calibrated and sharp)."
    )
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.08,
        subplot_titles=[f"{name} · held out {a['held_out']}" for name, a in AXES.items()],
    )
    for c, axis_name in enumerate(AXES, start=1):
        fig.add_hline(y=NOMINAL, line=dict(color="#94a3b8", dash="dot"), row=1, col=c)
        part = metrics[metrics["Axis"].eq(axis_name)]
        for _, r in part.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[r["Mean PI width [mm]"]], y=[r["Coverage [%]"]], mode="markers",
                    marker=dict(
                        size=16, color=SOURCES[r["source"]]["color"], symbol=MODELS[r["Model"]]["symbol"],
                        line=dict(color="white", width=1.5),
                    ),
                    name=f"{r['Model']} · {r['Training data']}", legendgroup=f"{r['Model']}{r['source']}",
                    showlegend=c == 1,
                    hovertemplate=(
                        f"<b>{r['Model']} · {r['Training data']}</b><br>coverage %{{y:.1f}}%<br>"
                        f"width %{{x:.3f}} mm<br>MAE {r['MAE [mm]']:.3f} mm<extra></extra>"
                    ),
                ),
                row=1, col=c,
            )
        fig.update_xaxes(title_text="Mean interval width [mm]  (narrower →  better)", row=1, col=c)
    fig.update_yaxes(title_text="Empirical coverage [%]", row=1, col=1)
    fig.update_layout(height=430, margin=dict(t=90), legend=dict(y=1.14))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def render_data(axis_name: str, group_id: int) -> None:
    axis = AXES[axis_name]
    setup, real = load_setup(), load_geometry("real")
    group = real[real[GROUP].eq(group_id)]

    left, right = st.columns([2, 1])
    with left:
        fig = go.Figure()
        add_profiles(fig, group, axis["col"], rgba(MEASURED, 0.75), name="Physical repetition", legend=True)
        mean = group.groupby(ANGLE)[axis["col"]].mean()
        fig.add_trace(go.Scatter(x=mean.index, y=mean.values, mode="lines", name="Group mean",
                                 line=dict(color="#2563eb", width=3, dash="dash")))
        fig.update_layout(
            title=f"Group {group_id}: {group[EXP].nunique()} physical repetitions of the same nominal setup",
            height=420, xaxis_title="Measurement position (angle / distance)", yaxis_title=axis["col"],
            margin=dict(t=90), legend=dict(y=1.04),
        )
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with right:
        row = setup[setup[GROUP].eq(group_id)].iloc[0]
        st.markdown("**Bending setup of this group**")
        st.dataframe(
            pd.DataFrame({"Parameter": PARAMETERS, "Value": [row[p] for p in PARAMETERS]}),
            hide_index=True, width="stretch",
        )
        sizes = real.groupby(GROUP)[EXP].nunique()
        dist = sizes.value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=dist.index, y=dist.values, text=dist.values, textposition="outside",
            marker_color=["#2563eb" if n == sizes[group_id] else "#cbd5e1" for n in dist.index],
            hovertemplate="%{y} groups with %{x} experiments<extra></extra>",
        ))
        fig.update_layout(title=f"Experiments per setup group (this group: {sizes[group_id]})", height=240,
                          margin=dict(t=40), xaxis=dict(dtick=1, title=None),
                          yaxis=dict(visible=False, range=[0, dist.max() * 1.25]))
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    first_run = int(group[EXP].min())
    st.markdown(f"#### Process signals recorded during experiment {first_run}")
    st.caption("The 12 machine-torque and movement channels whose statistical features feed the geometry models.")
    signals = load_signals(first_run)
    labels = {**SIGNALS["machine"], **SIGNALS["movement"]}
    fig = make_subplots(rows=3, cols=4, shared_xaxes=True, vertical_spacing=0.09, horizontal_spacing=0.05,
                        subplot_titles=list(labels.values()))
    for i, (column, label) in enumerate(labels.items()):
        color = "#2563eb" if column in SIGNALS["machine"] else "#0d9488"
        fig.add_trace(go.Scatter(x=signals.index, y=signals[column], mode="lines", line=dict(color=color, width=1.4),
                                 showlegend=False, hovertemplate=f"{label}<br>t=%{{x:.1f}} s<br>%{{y:.2f}}<extra></extra>"),
                      row=i // 4 + 1, col=i % 4 + 1)
    fig.update_annotations(font_size=12)
    fig.update_xaxes(title_text="Time [s]", row=3)
    fig.update_layout(height=560, margin=dict(t=40))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)


def render_augmentation(axis_name: str, group_id: int) -> None:
    axis = AXES[axis_name]
    st.caption(
        "Both strategies fill every group up to 15 profiles using the trained Random Forest geometry models. "
        "Dark lines are physical experiments, coloured lines are generated profiles."
    )
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[s["label"] for s in SOURCES.values()])
    for c, (source, meta) in enumerate(SOURCES.items(), start=1):
        df = load_geometry(source)
        df = df[df[GROUP].eq(group_id)]
        add_profiles(fig, df[df["synthetic"]], axis["col"], rgba(meta["color"], 0.55), 1, c, width=1.2,
                     name="Generated profile", legend=c == 2)
        add_profiles(fig, df[~df["synthetic"]], axis["col"], MEASURED, 1, c, width=2,
                     name="Physical experiment", legend=c == 1)
    fig.update_xaxes(title_text="Measurement position")
    fig.update_yaxes(title_text=axis["col"], col=1)
    fig.update_layout(height=420, margin=dict(t=100), legend=dict(y=1.16))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    st.markdown("#### Generated profiles are less variable than physical repetitions")
    left, right = st.columns([2, 1])
    spreads = {s: within_group_spread(s, axis["col"]) for s in SOURCES}
    with left:
        fig = go.Figure()
        for source, values in spreads.items():
            meta = SOURCES[source]
            fig.add_trace(go.Box(
                y=values.values, name=meta["label"], marker_color=meta["color"], boxmean=True,
                boxpoints="outliers", line=dict(width=1.5), fillcolor=rgba(meta["color"], 0.18),
                customdata=values.index, hovertemplate="Group %{customdata}<br>%{y:.4f} mm<extra></extra>",
            ))
        fig.update_layout(height=360, showlegend=False, yaxis_title="Mean within-group std [mm]",
                          title=f"{axis_name}: spread per setup group (one value per group)")
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
    with right:
        base = spreads["real"].mean()
        st.write("")
        kpi(st, "Physical repetitions", f"{base:.4f} mm", "mean within-group std (reference)")
        for source in (INTERP_KEY, SENSOR_KEY):
            value = spreads[source].mean()
            st.write("")
            kpi(st, SOURCES[source]["label"], f"{value:.4f} mm", f"{100 * (value / base - 1):+.0f}% spread vs real", "good")


def render_intervals(axis_name: str, group_id: int, metrics: pd.DataFrame) -> None:
    axis = AXES[axis_name]
    st.caption(
        f"Test groups contain a setting never seen in training (**{axis['held_out']}**). "
        f"Bands are the 90% prediction intervals for group {group_id}; dark lines are its measured experiments. "
        "Panel titles give the share of measured points inside the band for this group."
    )
    titles, panels = [], []
    for model in MODELS:
        for source, meta in SOURCES.items():
            pred = load_predictions(model, source, axis["key"], axis["split"])
            pred = pred[pred[GROUP].eq(group_id) & pred[EXP].lt(SYNTHETIC_ID_START)] if not pred.empty else pred
            inside = ((pred["y_true"] >= pred["y_lower"]) & (pred["y_true"] <= pred["y_upper"])).mean() * 100 \
                if not pred.empty else float("nan")
            titles.append(f"<b>{model}</b> · {meta['label']} — {inside:.0f}% inside")
            panels.append((model, pred))

    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes="all",
                        horizontal_spacing=0.03, vertical_spacing=0.1, subplot_titles=titles)
    for i, (model, pred) in enumerate(panels):
        r, c = i // 3 + 1, i % 3 + 1
        if pred.empty:
            continue
        add_interval(fig, pred, model, r, c, legend=c == 1)
        measured = pred.rename(columns={"y_true": "measured"})
        add_profiles(fig, measured, "measured", rgba(MEASURED, 0.7), r, c, width=1.4,
                     name="Measured experiment", legend=i == 0)
    fig.update_annotations(font_size=12)
    fig.update_xaxes(title_text="Measurement position", row=2)
    fig.update_yaxes(range=[21, 23])
    fig.update_yaxes(title_text=axis["col"], col=1)
    fig.update_layout(height=620, margin=dict(t=80), legend=dict(y=1.08))
    st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)

    st.markdown(f"#### {axis_name}: all held-out test groups")
    table = metrics[metrics["Axis"].eq(axis_name)].drop(columns=["Axis", "source"]).reset_index(drop=True)
    lower_is_better = [c for c in table.columns if c not in ("Model", "Training data", "Coverage [%]")]
    styled = (
        table.style.format(precision=3)
        .format({"Coverage [%]": "{:.1f}", "Calibration error [pp]": "{:.1f}"})
        .highlight_min(subset=lower_is_better, props="background-color:#dcfce7;font-weight:600")
    )
    st.dataframe(styled, hide_index=True, width="stretch")
    st.caption("Green = best value in the column. Calibration error is |coverage − 90|. "
               "QRF NLPD uses a Gaussian approximation of the 90% interval (thesis Eq. 4.1).")


def render_variability(axis_name: str) -> None:
    axis = AXES[axis_name]
    ranking = pd.Series(VARIABILITY_RANKING[axis["key"]]).sort_values()
    left, right = st.columns([1, 1.4])
    with left:
        fig = go.Figure(go.Bar(
            x=ranking.values, y=ranking.index, orientation="h",
            marker_color=["#2563eb" if v == ranking.max() else "#94a3b8" for v in ranking.values],
            text=[f"{v:.4f}" for v in ranking.values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.5f}<extra></extra>",
        ))
        fig.update_layout(height=380, title="Which parameter predicts repeatability?",
                          xaxis=dict(title="Permutation importance (Δ MAE)", range=[min(0, ranking.min()) - 0.004,
                                                                                     ranking.max() * 1.25]))
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption("Held-out permutation importance of a Random Forest predicting within-group variability "
                   "from the six setup parameters (thesis Table 4.20).")
    with right:
        parameter = st.pills(
            "Bending parameter", PARAMETERS, default=ranking.index[-1], key=f"param_{axis['key']}",
            label_visibility="collapsed",
        ) or ranking.index[-1]
        data = pairwise_variability(axis["col"])
        fig = go.Figure()
        levels = sorted(data[parameter].unique())
        for level in levels:
            values = data[data[parameter].eq(level)]
            fig.add_trace(go.Box(
                y=values["S_g"], name=f"{level:g}", boxpoints="all", jitter=0.4, pointpos=0,
                marker=dict(color="#2563eb", size=6, opacity=0.6), line=dict(color="#334155", width=1.2),
                fillcolor="rgba(0,0,0,0)", customdata=values[[GROUP, "runs"]],
                hovertemplate="Group %{customdata[0]} (%{customdata[1]} experiments)<br>S_g = %{y:.3f} mm<extra></extra>",
            ))
        fig.update_layout(height=340, showlegend=False, title=f"Variability across levels of {parameter}",
                          xaxis_title=parameter, yaxis_title="Mean pairwise distance S_g [mm]",
                          xaxis_type="category")
        st.plotly_chart(fig, width="stretch", config=PLOT_CONFIG)
        st.caption(
            f"Each dot is one setup group (physical experiments only). Adjusted relationship with variability on the "
            f"{axis_name}: **{VARIABILITY_DIRECTION[axis['key']][parameter]}** (thesis Table 4.21)."
        )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def main() -> None:
    st.markdown(
        '<div class="tb-title">Uncertainty-Aware Prediction Intervals for Rotary Tube Bending</div>'
        '<div class="tb-sub">Master thesis · Soroor Eskandari Khorram Abadi · University of Siegen, Ubicomp · '
        'TubeBEND dataset</div>',
        unsafe_allow_html=True,
    )
    c1, c2, _ = st.columns([1.1, 1.1, 3], vertical_alignment="bottom")
    axis_name = c1.segmented_control("Geometry axis", list(AXES), default=list(AXES)[0], key="axis") \
        or list(AXES)[0]
    axis = AXES[axis_name]
    test_groups = sorted(load_predictions("QRF", "real", axis["key"], axis["split"])[GROUP].unique())
    runs = load_geometry("real").groupby(GROUP)[EXP].nunique()
    if st.session_state.get("group") not in test_groups:
        st.session_state["group"] = 43 if 43 in test_groups else test_groups[0]
    group_id = c2.selectbox(
        "Test setup group", test_groups, key="group",
        format_func=lambda g: f"Group {g} · {runs.get(g, 0)} experiments",
        help=f"Groups held out for evaluation ({axis['held_out']}). Used by every tab.",
    )

    metrics = load_metrics()
    tabs = st.tabs(["Overview", "1 · Data", "2 · Augmentation", "3 · Prediction intervals", "4 · Variability"])
    with tabs[0]:
        render_overview(metrics)
    with tabs[1]:
        render_data(axis_name, int(group_id))
    with tabs[2]:
        render_augmentation(axis_name, int(group_id))
    with tabs[3]:
        render_intervals(axis_name, int(group_id), metrics)
    with tabs[4]:
        render_variability(axis_name)


main()
