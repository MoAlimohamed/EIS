"""
⚡ EIS Full Pipeline — Complete End-to-End Streamlit Application
================================================================
Covers all stages:
  1. 🔬 Data Generation  — Simulate EIS spectra for all 5 circuits
  2. 🧠 Train Models     — Train the classifier + 5 regression models
  3. 📊 Evaluation       — Confusion matrix, accuracy, MAE plots
  4. ⚡ EIS Analyzer     — Upload real data → classify → predict parameters

Feature vector (per frequency point, 6 features):
    [Z_imag, phase(°), |Z|, -Z_imag, -phase, -|Z|]
"""

# ── Imports ────────────────────────────────────────────────────────────────
import io, os, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EIS Full Pipeline",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

N_FREQ       = 100
F_MIN, F_MAX = 0.01, 1e5
R_RANGE      = (1e-1, 1e4)
Q_RANGE      = (1e-5, 1e-3)
ALPHA_RANGE  = (0.8,  1.0)
SIGMA_RANGE  = (1.0,  1e3)
Q_SCALE      = 1e6          # scale Q values for training stability

# Parameter names displayed to the user (after removing alpha columns)
PARAM_NAMES = {
    "C1": ["R1 (Ω)", "R2 (Ω)", "Q1 (F)"],
    "C2": ["R1 (Ω)", "R2 (Ω)", "R3 (Ω)", "Q1 (F)", "Q2 (F)"],
    "C3": ["Rs (Ω)", "R1 (Ω)", "Q1 (F)", "Sigma (Ω·s⁻⁰·⁵)"],
    "C4": ["Rs (Ω)", "R1 (Ω)", "R2 (Ω)", "Q1 (F)", "Q2 (F)", "Sigma (Ω·s⁻⁰·⁵)"],
    "C5": ["Rs (Ω)", "R1 (Ω)", "R2 (Ω)", "Q1 (F)", "Q2 (F)", "Sigma (Ω·s⁻⁰·⁵)"],
}
# Which columns of Zparam to select (drops ideality-factor columns)
PARAM_IDX = {
    "C1": [0, 1, 3],              # R1, R2, Q1      (skip alpha@2)
    "C2": [0, 1, 2, 4, 6],        # R1,R2,R3,Q1,Q2  (skip alpha@3,5)
    "C3": [0, 1, 3, 4],           # Rs,R1,Q1,Sigma  (skip alpha@2)
    "C4": [0, 1, 2, 4, 6, 7],     # Rs,R1,R2,Q1,Q2,Sigma (skip alpha@3,5)
    "C5": [0, 1, 2, 4, 6, 7],     # same
}
# Which of those selected columns are Q-type (need × Q_SCALE for training)
Q_COLS = {
    "C1": [2],
    "C2": [3, 4],
    "C3": [2],
    "C4": [3, 4],
    "C5": [3, 4],
}
CIRCUIT_TOPOLOGY = {
    "C1": "R1 + (R2 ∥ Q1)",
    "C2": "R1 + (R2∥Q1) + (R3∥Q2)",
    "C3": "R1 + (Q1 ∥ (R2 + W))",
    "C4": "R1 + (R2∥Q1) + (Q2∥(R3+W))",
    "C5": "R1 + (Q1 ∥ (R2 + (Q2∥(R3+W))))",
}

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.page-header{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);
  color:#e0f7fa;padding:1.6rem 2rem;border-radius:12px;margin-bottom:1.2rem;
  box-shadow:0 4px 20px rgba(0,0,0,.4);}
.page-header h1{margin:0;font-size:2rem;letter-spacing:1px;}
.page-header p{margin:.3rem 0 0;opacity:.85;font-size:.95rem;}
.card{background:#1e2a3a;border:1px solid #2e4057;border-radius:10px;
  padding:1.2rem 1.5rem;margin-bottom:1rem;}
.card-title{font-size:1rem;font-weight:700;color:#4fc3f7;
  border-bottom:1px solid #2e4057;padding-bottom:.35rem;margin-bottom:.8rem;}
.metric-row{display:flex;flex-wrap:wrap;gap:10px;margin-top:.5rem;}
.metric-card{background:linear-gradient(135deg,#1a2a3a,#0d1b2a);
  border:1px solid #0288d1;border-radius:8px;padding:.8rem 1rem;
  min-width:140px;flex:1;text-align:center;}
.metric-label{font-size:.75rem;color:#90caf9;margin-bottom:3px;}
.metric-value{font-size:1.25rem;font-weight:bold;color:#e1f5fe;}
.step-flow{display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;margin:.5rem 0 1rem;}
.step-pill{background:#0d2137;border:1px solid #0288d1;color:#4fc3f7;
  border-radius:20px;padding:.25rem .8rem;font-size:.8rem;}
.step-arrow{color:#546e7a;font-size:1rem;}
.success-badge{background:linear-gradient(90deg,#006064,#00acc1);color:white;
  padding:.35rem 1rem;border-radius:20px;font-size:1.1rem;font-weight:bold;
  letter-spacing:2px;display:inline-block;}
.warn{color:#ffb74d;font-size:.85rem;}
section[data-testid="stSidebar"]{background:#0d1b2a;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT SIMULATION FUNCTIONS  (ported from EIS_data_simulation.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

def _log_rand(lo, hi, n): return np.exp(np.log(lo) + (np.log(hi)-np.log(lo))*np.random.rand(n))
def _lin_rand(lo, hi, n): return lo + (hi-lo)*np.random.rand(n)

def _ZR(R):                     return R
def _ZQ(Q, alpha, omega):       return 1.0 / (Q * (1j*omega)**alpha)
def _ZW(sigma, omega):          return (sigma*np.sqrt(2)) / np.sqrt(1j*omega)

def _batch_ZR(R, omega):
    return np.ones((len(R), len(omega))) * R[:, None]

def _batch_ZQ(Q, alpha, omega):
    return 1.0 / (Q[:,None] * (1j * omega[None,:])**alpha[:,None])

def _batch_ZW(sigma, omega):
    return (sigma[:,None]*np.sqrt(2)) / np.sqrt(1j * omega[None,:])


def simulate_circuit(circuit: str, n: int, omega: np.ndarray):
    """Return (Z_complex, Zparam_raw) for one circuit type."""
    R1 = _log_rand(*R_RANGE, n);  Zr1 = _batch_ZR(R1, omega)
    R2 = _log_rand(*R_RANGE, n);  Zr2 = _batch_ZR(R2, omega)
    a1 = np.round(_lin_rand(*ALPHA_RANGE, n), 3)
    Q1 = _log_rand(*Q_RANGE,  n);  Zq1 = _batch_ZQ(Q1, a1, omega)

    if circuit == "C1":
        Z = Zr1 + 1/(1/Zr2 + 1/Zq1)
        p = np.column_stack([R1, R2, a1, Q1])

    elif circuit == "C2":
        R3 = _log_rand(*R_RANGE, n); Zr3 = _batch_ZR(R3, omega)
        a2 = np.round(_lin_rand(*ALPHA_RANGE, n), 3)
        Q2 = _log_rand(*Q_RANGE, n); Zq2 = _batch_ZQ(Q2, a1, omega)
        Z  = Zr1 + 1/(1/Zr2+1/Zq1) + 1/(1/Zr3+1/Zq2)
        p  = np.column_stack([R1, R2, R3, a1, Q1, a2, Q2])

    elif circuit == "C3":
        sigma = _log_rand(*SIGMA_RANGE, n); Zw = _batch_ZW(sigma, omega)
        Z = Zr1 + 1/(1/Zq1 + 1/(Zr2+Zw))
        p = np.column_stack([R1, R2, a1, Q1, sigma])

    elif circuit == "C4":
        a2 = np.round(_lin_rand(*ALPHA_RANGE, n), 3)
        Q2 = _log_rand(*Q_RANGE, n); Zq2 = _batch_ZQ(Q2, a1, omega)
        R3 = _log_rand(*R_RANGE, n); Zr3 = _batch_ZR(R3, omega)
        sigma = _log_rand(*SIGMA_RANGE, n); Zw = _batch_ZW(sigma, omega)
        Z  = Zr1 + 1/(1/Zr2+1/Zq1) + 1/(1/Zq2+1/(Zr3+Zw))
        p  = np.column_stack([R1, R2, R3, a1, Q1, a2, Q2, sigma])

    elif circuit == "C5":
        a2 = np.round(_lin_rand(*ALPHA_RANGE, n), 3)
        Q2 = _log_rand(*Q_RANGE, n); Zq2 = _batch_ZQ(Q2, a1, omega)
        R3 = _log_rand(*R_RANGE, n); Zr3 = _batch_ZR(R3, omega)
        sigma = _log_rand(*SIGMA_RANGE, n); Zw = _batch_ZW(sigma, omega)
        Z  = Zr1 + 1/(1/Zq1 + 1/(Zr2+1/(1/(Zr3+Zw)+1/Zq2)))
        p  = np.column_stack([R1, R2, R3, a1, Q1, a2, Q2, sigma])

    return Z, p


def build_x_features(Z: np.ndarray) -> np.ndarray:
    """Z shape (n,100) → x shape (n,100,6): [Zimag, phase°, |Z|, negatives]"""
    zimag  = Z.imag
    phase  = np.degrees(np.arctan2(Z.imag, Z.real))
    mag    = np.abs(Z)
    base   = np.stack([zimag, phase, mag], axis=2)        # (n,100,3)
    aug    = np.concatenate([base, -base], axis=2)         # (n,100,6)
    return aug.astype(np.float32)


def build_y_regression(Zparam: np.ndarray, circuit: str) -> np.ndarray:
    """Select relevant columns + scale Q values."""
    y = Zparam[:, PARAM_IDX[circuit]].copy()
    for col in Q_COLS[circuit]:
        y[:, col] *= Q_SCALE
    return y.astype(np.float32)


def inverse_scale_y(y_pred: np.ndarray, circuit: str) -> np.ndarray:
    """Undo Q scaling on predicted values for display."""
    y = y_pred.copy()
    for col in Q_COLS[circuit]:
        y[col] /= Q_SCALE
    return y

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURE FUNCTIONS  (ported from Classification & Regression nbs)
# ══════════════════════════════════════════════════════════════════════════════

def build_classifier(input_shape=(N_FREQ, 6)):
    import tensorflow as tf
    keras = tf.keras
    init  = tf.keras.initializers.HeNormal()
    inp   = keras.Input(input_shape)
    x = keras.layers.Conv1D(64,  32, padding="same", activation="relu", kernel_initializer=init)(inp)
    x = keras.layers.Conv1D(128, 16, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Conv1D(256,  8, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Conv1D(512,  4, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Conv1D(768,  2, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.SpatialDropout1D(0.7)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(1024, activation="relu", kernel_initializer=init)(x)
    out = keras.layers.Dense(5, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_regression(n_outputs: int, input_shape=(N_FREQ, 6)):
    import tensorflow as tf
    keras = tf.keras
    init  = tf.keras.initializers.HeNormal()
    inp   = keras.Input(input_shape)
    x = keras.layers.Conv1D(64,  32, padding="same", activation="relu", kernel_initializer=init)(inp)
    x = keras.layers.Conv1D(128, 16, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Conv1D(256,  8, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Conv1D(512,  4, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Conv1D(768,  2, padding="same", activation="relu", kernel_initializer=init)(x)
    x = keras.layers.SpatialDropout1D(0.3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Dense(512, activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Dense(64,  activation="relu", kernel_initializer=init)(x)
    x = keras.layers.Dense(64,  activation="relu", kernel_initializer=init)(x)
    out = keras.layers.Dense(n_outputs)(x)
    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# ══════════════════════════════════════════════════════════════════════════════
#  KERAS PROGRESS CALLBACK  (live updates inside Streamlit)
# ══════════════════════════════════════════════════════════════════════════════

def make_progress_callback(total_epochs, progress_bar, status_text, chart_placeholder):
    import tensorflow as tf

    class _CB(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self._hist = {"loss":[], "val_loss":[], "metric":[], "val_metric":[]}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            frac = (epoch+1)/total_epochs
            progress_bar.progress(min(frac, 1.0))

            for k, src in [("loss","loss"),("val_loss","val_loss"),
                           ("metric", list({k for k in logs if k not in ("loss","val_loss") and not k.startswith("val_")})),
                           ("val_metric", list({k for k in logs if k.startswith("val_") and k!="val_loss"}))]:
                if isinstance(src, list):
                    val = logs.get(src[0]) if src else None
                else:
                    val = logs.get(src)
                if val is not None:
                    self._hist[k].append(float(val))

            loss = logs.get("loss", 0)
            vl   = logs.get("val_loss", 0)
            status_text.markdown(
                f"**Epoch {epoch+1}/{total_epochs}** &nbsp;|&nbsp; "
                f"loss: `{loss:.5f}` &nbsp;|&nbsp; val_loss: `{vl:.5f}`"
            )

            if (epoch+1) % max(1, total_epochs//20) == 0 and len(self._hist["loss"]) > 1:
                epochs_done = list(range(1, len(self._hist["loss"])+1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=epochs_done, y=self._hist["loss"],
                    name="Train Loss", line=dict(color="#00e5ff", width=2)))
                fig.add_trace(go.Scatter(x=epochs_done, y=self._hist["val_loss"],
                    name="Val Loss", line=dict(color="#ff7043", width=2)))
                fig.update_layout(
                    xaxis_title="Epoch", yaxis_title="Loss",
                    paper_bgcolor="#101e2c", plot_bgcolor="#101e2c",
                    font=dict(color="#cfd8dc"), height=250,
                    margin=dict(l=50,r=20,t=30,b=40),
                    legend=dict(bgcolor="#0d1b2a"),
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)
    return _CB()

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL SAVE / LOAD  (bytes for session state + download)
# ══════════════════════════════════════════════════════════════════════════════

def model_to_bytes(model) -> bytes:
    buf = io.BytesIO()
    model.save(buf, save_format="h5")
    return buf.getvalue()

@st.cache_resource(show_spinner="Loading model…")
def model_from_bytes(b: bytes):
    import tensorflow as tf
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        f.write(b); tmp = f.name
    m = tf.keras.models.load_model(tmp, compile=False)
    os.unlink(tmp)
    return m

# ══════════════════════════════════════════════════════════════════════════════
#  SHARED PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_DARK = dict(paper_bgcolor="#101e2c", plot_bgcolor="#101e2c",
             font=dict(color="#cfd8dc"))

def _dark_fig(**kw):
    fig = go.Figure(**kw)
    fig.update_layout(**_DARK, margin=dict(l=55,r=20,t=45,b=50))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚡ EIS Full Pipeline")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home",
         "🔬 Data Generation",
         "🧠 Train Models",
         "📊 Evaluation",
         "⚡ EIS Analyzer"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Status indicators
    has_data   = "sim_data" in st.session_state
    has_cls    = "cls_model_bytes" in st.session_state
    has_all_reg= all(f"reg_model_bytes_{c}" in st.session_state for c in ["C1","C2","C3","C4","C5"])

    def _dot(ok): return "🟢" if ok else "🔴"

    st.markdown(
        f"{_dot(has_data)} Simulation data\n\n"
        f"{_dot(has_cls)} Classifier model\n\n"
        f"{_dot(has_all_reg)} All regression models"
    )
    st.markdown("---")
    st.markdown('<p style="color:#546e7a;font-size:.75rem;">Data & models are kept in session memory. Refresh = reset.</p>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":
    st.markdown("""
    <div class="page-header">
      <h1>⚡ EIS Full Pipeline</h1>
      <p>Complete end-to-end platform: simulate → train → evaluate → analyze</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-flow">
      <span class="step-pill">① Generate Data</span><span class="step-arrow">→</span>
      <span class="step-pill">② Train Classifier</span><span class="step-arrow">→</span>
      <span class="step-pill">③ Train 5 Regression Models</span><span class="step-arrow">→</span>
      <span class="step-pill">④ Evaluate</span><span class="step-arrow">→</span>
      <span class="step-pill">⑤ Analyze Real Data</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><div class="card-title">🔬 5 Equivalent Circuit Models</div>', unsafe_allow_html=True)
        for c, t in CIRCUIT_TOPOLOGY.items():
            pnames = " · ".join(PARAM_NAMES[c])
            st.markdown(f"**{c}** — `{t}`  \n<span style='color:#90caf9;font-size:.8rem'>{pnames}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><div class="card-title">📐 Model Input / Output</div>', unsafe_allow_html=True)
        st.markdown("""
| | Details |
|---|---|
| **Input shape** | `(1, 100, 6)` |
| **Features** | Z_imag, Phase°, abs(Z), −Z_imag, −Phase, −abs(Z) | 
| **Freq points** | 100 log-spaced (0.01 → 10⁵ Hz) |
| **Classifier out** | 5-class softmax |
| **Regression out** | 3–6 circuit parameters |
""")
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("👈  Use the sidebar to navigate through each stage of the pipeline.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Data Generation":
    st.markdown("""
    <div class="page-header">
      <h1>🔬 Data Generation</h1>
      <p>Simulate synthetic EIS spectra for all 5 equivalent circuit topologies</p>
    </div>
    """, unsafe_allow_html=True)

    # Config
    st.markdown('<div class="card"><div class="card-title">⚙️ Simulation Settings</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.slider("Samples per circuit", 200, 5000, 1000, 100,
                              help="More = better models but slower training")
    with c2:
        f_min_exp = st.slider("Min frequency (10^x Hz)", -2, 1, -2)
        f_max_exp = st.slider("Max frequency (10^x Hz)",  3, 6,  5)
    with c3:
        seed = st.number_input("Random seed", 0, 9999, 42)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("▶️  Generate Simulation Data", type="primary"):
        np.random.seed(int(seed))
        freq  = np.logspace(f_min_exp, f_max_exp, N_FREQ)
        omega = 2 * np.pi * freq

        circuits = ["C1", "C2", "C3", "C4", "C5"]
        prog = st.progress(0.0)
        status = st.empty()

        # Classification data
        x_cls_list, y_cls_list = [], []
        # Regression data per circuit
        reg_data = {}

        for idx, circ in enumerate(circuits):
            status.text(f"Simulating {circ} ({n_samples} samples)…")
            Z, Zparam = simulate_circuit(circ, n_samples, omega)
            x_feat = build_x_features(Z)
            y_reg  = build_y_regression(Zparam, circ)

            x_cls_list.append(x_feat)
            y_cls_list.append(np.full(n_samples, idx, dtype=np.int32))

            reg_data[circ] = {"x": x_feat, "y": y_reg, "Z": Z, "freq": freq}
            prog.progress((idx+1)/5)

        x_cls = np.concatenate(x_cls_list, axis=0)
        y_cls = np.concatenate(y_cls_list, axis=0)

        # Shuffle
        perm  = np.random.permutation(len(x_cls))
        x_cls, y_cls = x_cls[perm], y_cls[perm]

        st.session_state["sim_data"] = {
            "x_cls": x_cls, "y_cls": y_cls,
            "reg": reg_data, "freq": freq,
            "n_samples": n_samples,
        }
        status.success(f"✅  Done! {len(x_cls):,} total spectra generated across 5 circuits.")
        prog.progress(1.0)

    # ── Preview ──────────────────────────────────────────────────────────────
    if "sim_data" in st.session_state:
        d = st.session_state["sim_data"]
        st.markdown("---")
        st.markdown('<div class="card"><div class="card-title">📊 Data Preview</div>', unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["Class Distribution", "Nyquist Samples", "Parameter Distributions"])

        with tab1:
            cnts = np.bincount(d["y_cls"])
            fig  = go.Figure(go.Bar(
                x=[f"C{i+1}" for i in range(5)], y=cnts,
                marker_color=["#00acc1","#26c6da","#4dd0e1","#80deea","#b2ebf2"],
            ))
            fig.update_layout(**_DARK, title="Samples per Circuit",
                              xaxis_title="Circuit", yaxis_title="Count", height=300,
                              margin=dict(l=50,r=20,t=40,b=40))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            sel = st.selectbox("Circuit to preview", ["C1","C2","C3","C4","C5"])
            n_show = min(5, d["n_samples"])
            Zs  = d["reg"][sel]["Z"][:n_show]
            fig = go.Figure()
            colors = px.colors.sequential.Blues[3:]
            for i, Zi in enumerate(Zs):
                fig.add_trace(go.Scatter(
                    x=Zi.real, y=-Zi.imag,
                    mode="lines+markers", name=f"Sample {i+1}",
                    line=dict(color=colors[i % len(colors)], width=1.5),
                    marker=dict(size=3),
                ))
            fig.update_layout(**_DARK, title=f"Nyquist — {sel} (first {n_show} samples)",
                              xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)", height=350,
                              margin=dict(l=60,r=20,t=40,b=50))
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            sel2 = st.selectbox("Circuit", ["C1","C2","C3","C4","C5"], key="dist_c")
            y_raw = d["reg"][sel2]["y"].copy()
            # Inverse-scale Q columns for display
            for col in Q_COLS[sel2]:
                y_raw[:, col] /= Q_SCALE
            pnames = PARAM_NAMES[sel2]
            n_p    = len(pnames)
            cols_p = st.columns(min(n_p, 3))
            for i, (pname, vals) in enumerate(zip(pnames, y_raw.T)):
                with cols_p[i % 3]:
                    fig = go.Figure(go.Histogram(
                        x=np.log10(np.abs(vals)+1e-20),
                        marker_color="#00acc1", opacity=0.8,
                    ))
                    fig.update_layout(**_DARK, title=f"log₁₀({pname})",
                                      height=220, margin=dict(l=40,r=10,t=35,b=35),
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — TRAIN MODELS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧠 Train Models":
    st.markdown("""
    <div class="page-header">
      <h1>🧠 Train Models</h1>
      <p>Train the circuit classifier and all 5 regression models on the simulated data</p>
    </div>
    """, unsafe_allow_html=True)

    if "sim_data" not in st.session_state:
        st.warning("⚠️  No simulation data found. Go to **🔬 Data Generation** first.")
        st.stop()

    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    d = st.session_state["sim_data"]

    tab_cls, tab_reg = st.tabs(["🔵  Classifier", "🟠  Regression Models (C1–C5)"])

    # ── CLASSIFIER TAB ──────────────────────────────────────────────────────
    with tab_cls:
        st.markdown('<div class="card"><div class="card-title">⚙️ Classifier Training Settings</div>',
                    unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)
        with cc1: cls_epochs = st.slider("Epochs", 5, 400, 50, 5, key="cls_ep")
        with cc2: cls_batch  = st.slider("Batch size", 32, 2048, 256, 32, key="cls_bs")
        with cc3: cls_split  = st.slider("Val split", 0.1, 0.4, 0.2, 0.05, key="cls_sp")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("▶️  Train Classifier", type="primary"):
            x = d["x_cls"]
            y = tf.keras.utils.to_categorical(d["y_cls"], 5)
            x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=cls_split, random_state=42)

            model = build_classifier()
            st.markdown(f"**Parameters:** `{model.count_params():,}`")

            prog   = st.progress(0.0)
            status = st.empty()
            chart  = st.empty()

            cb = make_progress_callback(cls_epochs, prog, status, chart)
            lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0)

            model.fit(x_tr, y_tr, epochs=cls_epochs, batch_size=cls_batch,
                      validation_data=(x_te, y_te), callbacks=[cb, lr_cb], verbose=0)

            st.session_state["cls_model_bytes"] = model_to_bytes(model)
            st.session_state["cls_history"]     = cb._hist
            st.session_state["cls_test_data"]   = (x_te, y_te)

            st.success("✅  Classifier trained and saved!")

            ev = model.evaluate(x_te, y_te, verbose=0)
            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-card"><div class="metric-label">Test Loss</div>'
                f'<div class="metric-value">{ev[0]:.4f}</div></div>'
                f'<div class="metric-card"><div class="metric-label">Test Accuracy</div>'
                f'<div class="metric-value">{ev[1]*100:.1f}%</div></div>'
                f'</div>', unsafe_allow_html=True,
            )

        if "cls_model_bytes" in st.session_state:
            st.download_button("⬇️ Download Classifier.h5",
                data=st.session_state["cls_model_bytes"],
                file_name="Classifier.h5", mime="application/octet-stream")

    # ── REGRESSION TAB ──────────────────────────────────────────────────────
    with tab_reg:
        st.markdown('<div class="card"><div class="card-title">⚙️ Regression Training Settings</div>',
                    unsafe_allow_html=True)
        rc1, rc2, rc3 = st.columns(3)
        with rc1: reg_epochs = st.slider("Epochs", 5, 400, 50, 5, key="reg_ep")
        with rc2: reg_batch  = st.slider("Batch size", 32, 2048, 256, 32, key="reg_bs")
        with rc3: reg_split  = st.slider("Val split", 0.1, 0.4, 0.2, 0.05, key="reg_sp")

        circuits_to_train = st.multiselect(
            "Circuits to train", ["C1","C2","C3","C4","C5"],
            default=["C1","C2","C3","C4","C5"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("▶️  Train Regression Models", type="primary"):
            for circ in circuits_to_train:
                st.markdown(f"### Training **{circ}** — `{CIRCUIT_TOPOLOGY[circ]}`")
                xr = d["reg"][circ]["x"]
                yr = d["reg"][circ]["y"]
                x_tr, x_te, y_tr, y_te = train_test_split(xr, yr, test_size=reg_split, random_state=42)

                n_out = len(PARAM_NAMES[circ])
                model = build_regression(n_out)
                st.markdown(f"**Parameters:** `{model.count_params():,}` &nbsp;|&nbsp; **Outputs:** {n_out}")

                prog   = st.progress(0.0)
                status = st.empty()
                chart  = st.empty()

                cb = make_progress_callback(reg_epochs, prog, status, chart)
                lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=0)

                model.fit(x_tr, y_tr, epochs=reg_epochs, batch_size=reg_batch,
                          validation_data=(x_te, y_te), callbacks=[cb, lr_cb], verbose=0)

                st.session_state[f"reg_model_bytes_{circ}"] = model_to_bytes(model)
                st.session_state[f"reg_history_{circ}"]     = cb._hist
                st.session_state[f"reg_test_{circ}"]        = (x_te, y_te)

                ev = model.evaluate(x_te, y_te, verbose=0)
                st.success(f"✅  {circ} trained!  Test MAE = `{ev[1]:.5f}`")
                st.download_button(f"⬇️ Download Reg{circ}.h5",
                    data=st.session_state[f"reg_model_bytes_{circ}"],
                    file_name=f"Reg{circ}.h5", mime="application/octet-stream",
                    key=f"dl_{circ}")
                st.markdown("---")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Evaluation":
    st.markdown("""
    <div class="page-header">
      <h1>📊 Model Evaluation</h1>
      <p>Confusion matrix, accuracy curves, and per-parameter regression error analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if "cls_model_bytes" not in st.session_state:
        st.warning("⚠️  No trained models found. Complete **🧠 Train Models** first.")
        st.stop()

    import tensorflow as tf

    tab_cls_eval, tab_reg_eval = st.tabs(["Classifier", "Regression"])

    # ── CLASSIFIER EVALUATION ───────────────────────────────────────────────
    with tab_cls_eval:
        cls_model = model_from_bytes(st.session_state["cls_model_bytes"])
        x_te, y_te = st.session_state.get("cls_test_data", (None, None))

        if x_te is None:
            st.info("Test data not available (re-run training or regenerate data).")
        else:
            ev = cls_model.evaluate(x_te, y_te, verbose=0)
            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-card"><div class="metric-label">Test Loss</div>'
                f'<div class="metric-value">{ev[0]:.4f}</div></div>'
                f'<div class="metric-card"><div class="metric-label">Test Accuracy</div>'
                f'<div class="metric-value">{ev[1]*100:.2f}%</div></div>'
                f'</div>', unsafe_allow_html=True,
            )

            y_pred   = cls_model.predict(x_te, verbose=0)
            y_true_c = np.argmax(y_te, axis=1)
            y_pred_c = np.argmax(y_pred, axis=1)

            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true_c, y_pred_c)
            labs = ["C1","C2","C3","C4","C5"]
            fig = go.Figure(go.Heatmap(
                z=cm, x=labs, y=labs,
                colorscale="Blues", showscale=True,
                text=cm, texttemplate="%{text}",
                textfont=dict(size=14),
            ))
            fig.update_layout(**_DARK, title="Confusion Matrix",
                              xaxis_title="Predicted", yaxis_title="True",
                              height=400, margin=dict(l=60,r=20,t=50,b=50))
            st.plotly_chart(fig, use_container_width=True)

            # Training curves
            if "cls_history" in st.session_state:
                h = st.session_state["cls_history"]
                ep = list(range(1, len(h["loss"])+1))
                fig2 = make_subplots(rows=1, cols=2, subplot_titles=["Loss","Accuracy"])
                fig2.add_trace(go.Scatter(x=ep, y=h["loss"],    name="Train",      line=dict(color="#00e5ff")), 1,1)
                fig2.add_trace(go.Scatter(x=ep, y=h["val_loss"],name="Validation", line=dict(color="#ff7043")), 1,1)
                if h["metric"]:
                    fig2.add_trace(go.Scatter(x=ep[:len(h["metric"])], y=h["metric"],     name="Train Acc",  line=dict(color="#69f0ae")), 1,2)
                    fig2.add_trace(go.Scatter(x=ep[:len(h["val_metric"])], y=h["val_metric"], name="Val Acc",    line=dict(color="#ffab40")), 1,2)
                fig2.update_layout(**_DARK, height=300, margin=dict(l=50,r=20,t=40,b=40))
                st.plotly_chart(fig2, use_container_width=True)

    # ── REGRESSION EVALUATION ───────────────────────────────────────────────
    with tab_reg_eval:
        sel_c = st.selectbox("Select circuit", ["C1","C2","C3","C4","C5"])
        key_b = f"reg_model_bytes_{sel_c}"
        key_t = f"reg_test_{sel_c}"

        if key_b not in st.session_state:
            st.info(f"No trained regression model for {sel_c}.")
        else:
            reg_model = model_from_bytes(st.session_state[key_b])
            x_te_r, y_te_r = st.session_state[key_t]

            ev = reg_model.evaluate(x_te_r, y_te_r, verbose=0)
            st.markdown(
                f'<div class="metric-row">'
                f'<div class="metric-card"><div class="metric-label">Test MSE</div>'
                f'<div class="metric-value">{ev[0]:.5f}</div></div>'
                f'<div class="metric-card"><div class="metric-label">Test MAE</div>'
                f'<div class="metric-value">{ev[1]:.5f}</div></div>'
                f'</div>', unsafe_allow_html=True,
            )

            y_pred_r = reg_model.predict(x_te_r, verbose=0)

            pnames  = PARAM_NAMES[sel_c]
            ncols   = min(len(pnames), 3)
            cols_ev = st.columns(ncols)

            for i, pname in enumerate(pnames):
                true_v = y_te_r[:, i].copy()
                pred_v = y_pred_r[:, i].copy()
                if i in Q_COLS[sel_c]:
                    true_v /= Q_SCALE; pred_v /= Q_SCALE

                mae_v = np.mean(np.abs(true_v - pred_v))
                mape  = np.mean(np.abs((true_v - pred_v) / (np.abs(true_v)+1e-20)))*100

                with cols_ev[i % ncols]:
                    fig = go.Figure()
                    lim = [min(true_v.min(), pred_v.min()), max(true_v.max(), pred_v.max())]
                    fig.add_trace(go.Scatter(
                        x=true_v[:500], y=pred_v[:500], mode="markers",
                        marker=dict(color="#00acc1", size=3, opacity=0.5),
                        name="Samples",
                    ))
                    fig.add_trace(go.Scatter(
                        x=lim, y=lim, mode="lines",
                        line=dict(color="#ff7043", dash="dash", width=1.5), name="Ideal",
                    ))
                    fig.update_layout(**_DARK, title=f"{pname}<br><sup>MAE={mae_v:.3g} | MAPE={mape:.1f}%</sup>",
                                      xaxis_title="True", yaxis_title="Predicted",
                                      height=260, margin=dict(l=50,r=10,t=55,b=40),
                                      showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — EIS ANALYZER  (upload real data → classify → predict)
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚡ EIS Analyzer":
    st.markdown("""
    <div class="page-header">
      <h1>⚡ EIS Analyzer</h1>
      <p>Upload real EIS data → identify equivalent circuit → predict component parameters</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="step-flow">
      <span class="step-pill">① Upload CSV</span><span class="step-arrow">→</span>
      <span class="step-pill">② Visualize</span><span class="step-arrow">→</span>
      <span class="step-pill">③ Classify Circuit</span><span class="step-arrow">→</span>
      <span class="step-pill">④ Predict Parameters</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Model source selection ───────────────────────────────────────────────
    with st.expander("🔧 Model Source (click to configure)", expanded=False):
        model_src = st.radio("Use models from:", ["Session (trained in this app)", "Upload .h5 files manually"], horizontal=True)

        uploaded_cls, uploaded_reg = None, {}
        if model_src == "Upload .h5 files manually":
            st.markdown("Upload your pre-trained `.h5` files:")
            uploaded_cls = st.file_uploader("Classifier.h5", type=["h5"], key="anz_cls")
            for c in ["C1","C2","C3","C4","C5"]:
                uploaded_reg[c] = st.file_uploader(f"Reg{c}.h5", type=["h5"], key=f"anz_{c}")

    # ── Resolve models ───────────────────────────────────────────────────────
    def _get_model(key_bytes, upload_file):
        if upload_file is not None:
            return model_from_bytes(upload_file.read())
        if key_bytes in st.session_state:
            return model_from_bytes(st.session_state[key_bytes])
        return None

    # ── Data upload ──────────────────────────────────────────────────────────
    st.markdown('<div class="card"><div class="card-title">📂 Upload EIS Data (CSV)</div>', unsafe_allow_html=True)
    st.markdown('<span class="warn">Required columns: <b>Frequency</b>, <b>Z_real</b>, <b>Z_imag</b></span>', unsafe_allow_html=True)
    data_file = st.file_uploader("EIS CSV file", type=["csv"], key="anz_data")
    st.markdown("</div>", unsafe_allow_html=True)

    df_eis = None
    if data_file:
        try:
            df_eis = pd.read_csv(data_file)
            df_eis.columns = [c.strip() for c in df_eis.columns]
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")

    # ── Visualisation ────────────────────────────────────────────────────────
    if df_eis is not None:
        req = {"Frequency","Z_real","Z_imag"}
        if not req.issubset(df_eis.columns):
            st.error(f"Missing columns: {req - set(df_eis.columns)}")
        else:
            st.markdown('<div class="card"><div class="card-title">📊 Section 1 — Data Preview</div>', unsafe_allow_html=True)
            ca, cb_ = st.columns([3,1])
            with ca: st.dataframe(df_eis.head(12), use_container_width=True, height=280)
            with cb_:
                st.markdown(f"**Rows:** {len(df_eis)}")
                st.markdown(f"**Freq range:** {df_eis['Frequency'].min():.2e} – {df_eis['Frequency'].max():.2e} Hz")
                mag = np.sqrt(df_eis['Z_real']**2 + df_eis['Z_imag']**2)
                st.markdown(f"**|Z| range:** {mag.min():.3g} – {mag.max():.3g} Ω")

            tab_nyq, tab_bode = st.tabs(["Nyquist Plot", "Bode Plot"])
            with tab_nyq:
                fig = go.Figure(go.Scatter(
                    x=df_eis["Z_real"], y=-df_eis["Z_imag"], mode="lines+markers",
                    line=dict(color="#00e5ff",width=2), marker=dict(size=5),
                ))
                fig.update_layout(**_DARK, title="Nyquist Plot",
                    xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)",
                    height=380, margin=dict(l=60,r=20,t=45,b=55),
                    xaxis=dict(showgrid=True, gridcolor="#1c2d3a"),
                    yaxis=dict(showgrid=True, gridcolor="#1c2d3a", scaleanchor="x", scaleratio=1),
                )
                st.plotly_chart(fig, use_container_width=True)

            with tab_bode:
                freq_b = df_eis["Frequency"].values
                mag_b  = np.sqrt(df_eis["Z_real"]**2 + df_eis["Z_imag"]**2)
                ph_b   = np.degrees(np.arctan2(df_eis["Z_imag"], df_eis["Z_real"]))
                fig2   = make_subplots(specs=[[{"secondary_y":True}]])
                fig2.add_trace(go.Scatter(x=freq_b,y=mag_b, name="|Z|",
                    line=dict(color="#00e5ff",width=2)), secondary_y=False)
                fig2.add_trace(go.Scatter(x=freq_b,y=ph_b, name="Phase°",
                    line=dict(color="#ff7043",width=2)), secondary_y=True)
                fig2.update_xaxes(type="log", title="Frequency (Hz)",
                    showgrid=True, gridcolor="#1c2d3a")
                fig2.update_yaxes(title="|Z| (Ω)", secondary_y=False, showgrid=True, gridcolor="#1c2d3a")
                fig2.update_yaxes(title="Phase (°)", secondary_y=True, showgrid=False)
                fig2.update_layout(**_DARK, height=350, margin=dict(l=60,r=60,t=40,b=50))
                st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Preprocessing ─────────────────────────────────────────────────
            def preprocess_df(df):
                df = df[["Frequency","Z_real","Z_imag"]].dropna().sort_values("Frequency").reset_index(drop=True)
                if len(df) < 3: return None
                if len(df) != N_FREQ:
                    from scipy.interpolate import interp1d
                    lf_o = np.log10(df["Frequency"].values)
                    lf_n = np.linspace(lf_o[0], lf_o[-1], N_FREQ)
                    df2  = pd.DataFrame({"Frequency": 10**lf_n})
                    for col in ["Z_real","Z_imag"]:
                        df2[col] = interp1d(lf_o, df[col].values)(lf_n)
                    df = df2
                freq_v = df["Frequency"].values
                zr = df["Z_real"].values; zi = df["Z_imag"].values
                # Build features: [Z_imag, phase°, |Z|] × 2 (augmented)
                zimag = zi
                phase = np.degrees(np.arctan2(zi, zr))
                mag   = np.sqrt(zr**2 + zi**2)
                base  = np.stack([zimag, phase, mag], axis=1)          # (100,3)
                aug   = np.concatenate([base, -base], axis=1)          # (100,6)
                return aug[np.newaxis].astype(np.float32)               # (1,100,6)

            x_input = preprocess_df(df_eis)

            # ── Classification ────────────────────────────────────────────────
            st.markdown('<div class="card"><div class="card-title">🔍 Section 2 — Circuit Classification</div>',
                        unsafe_allow_html=True)
            run_btn = st.button("🔍  Identify Equivalent Circuit", type="primary",
                                disabled=(x_input is None))

            if run_btn:
                cls_model = _get_model("cls_model_bytes", uploaded_cls)
                if cls_model is None:
                    st.warning("⚠️  No classifier model available. Train one first or upload Classifier.h5.")
                else:
                    probs = cls_model.predict(x_input, verbose=0)[0]
                    pred  = int(np.argmax(probs))
                    circ  = f"C{pred+1}"
                    st.session_state["anz_circuit"] = circ
                    st.session_state["anz_probs"]   = probs
                    st.session_state["anz_x"]       = x_input

            if "anz_circuit" in st.session_state:
                circ  = st.session_state["anz_circuit"]
                probs = st.session_state["anz_probs"]

                c_res, c_bars = st.columns(2)
                with c_res:
                    st.markdown(f'<span class="success-badge">{circ}</span>', unsafe_allow_html=True)
                    st.success(f"✅ **{circ}** identified  —  `{CIRCUIT_TOPOLOGY[circ]}`  "
                               f"({probs[int(circ[1])-1]*100:.1f}% confidence)")
                with c_bars:
                    st.markdown("**Class probabilities**")
                    for i in np.argsort(probs)[::-1]:
                        pct = probs[i]*100
                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                            f'<span style="min-width:26px;color:#cfd8dc;font-weight:600">C{i+1}</span>'
                            f'<div style="flex:1;height:12px;background:#1c2d3a;border-radius:6px;overflow:hidden">'
                            f'<div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,#006064,#00e5ff);border-radius:6px"></div>'
                            f'</div><span style="min-width:44px;text-align:right;color:#80cbc4;font-size:.82rem">{pct:.1f}%</span></div>',
                            unsafe_allow_html=True,
                        )
            st.markdown("</div>", unsafe_allow_html=True)

            # ── Regression ────────────────────────────────────────────────────
            st.markdown('<div class="card"><div class="card-title">📐 Section 3 — Parameter Prediction</div>',
                        unsafe_allow_html=True)
            if "anz_circuit" in st.session_state:
                circ   = st.session_state["anz_circuit"]
                x_inp  = st.session_state["anz_x"]
                reg_m  = _get_model(f"reg_model_bytes_{circ}", uploaded_reg.get(circ))
                if reg_m is None:
                    st.warning(f"⚠️  No regression model for {circ}. Train it first or upload Reg{circ}.h5.")
                else:
                    raw  = reg_m.predict(x_inp, verbose=0)[0]
                    vals = inverse_scale_y(raw, circ)
                    pnames = PARAM_NAMES[circ]

                    # Metric cards
                    cards = "".join(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">{n}</div>'
                        f'<div class="metric-value">{v:.4g}</div>'
                        f'</div>'
                        for n, v in zip(pnames, vals)
                    )
                    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)
                    st.markdown(" ")

                    res_df = pd.DataFrame({"Parameter": pnames, "Predicted Value": vals})
                    st.dataframe(res_df.style.format({"Predicted Value":"{:.6g}"}),
                                 use_container_width=True, hide_index=True)

                    st.download_button("⬇️ Download Results CSV",
                        data=res_df.to_csv(index=False).encode(),
                        file_name=f"EIS_params_{circ}.csv", mime="text/csv")
            else:
                st.info("Run the classification step above to unlock parameter prediction.")
            st.markdown("</div>", unsafe_allow_html=True)
