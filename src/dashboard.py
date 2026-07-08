"""
dashboard.py  -  "SENTINEL / vibration" styled real-time monitor.

Three live modes, all functional:
  * CSV Replay     : upload a Phyphox CSV, scored window-by-window
  * Phyphox Live   : poll phone accelerometer over LAN
  * ESP32 Edge Feed: poll telemetry_server.py for live on-chip IF scores

Metric cards read REAL numbers from data/eval_results.npz (evaluate.py).
Run from project root:  streamlit run src/dashboard.py
"""

import os, sys, io, time
import numpy as np
import pandas as pd
import joblib
import requests
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_utils import (
    LOF_PATH, RESULTS_PATH, AXES, TARGET_FS, WINDOW_SIZE, STEP_SIZE,
    resample_to_target, estimate_fs, extract_features_one_window,
)

st.set_page_config(page_title="SENTINEL / vibration", page_icon="📡", layout="wide")

# ── Theme ────────────────────────────────────────────────────────────────────
ACCENT = "#2DD4BF"
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap');
.stApp { background:#070B12; color:#E5ECF3; font-family:'Inter',sans-serif; }
#MainMenu, header, footer {visibility:hidden;}
.block-container {padding-top:1.5rem; max-width:1200px;}
.topbar {display:flex; justify-content:space-between; align-items:center;
         border-bottom:1px solid rgba(148,163,184,.12); padding-bottom:14px; margin-bottom:26px;}
.brand {font-weight:800; letter-spacing:.5px; font-size:15px;}
.brand .dim {color:#5B6B7E; font-weight:600;}
.status {font-family:'JetBrains Mono',monospace; font-size:12px; color:#7C8A9A;}
.status b {color:#2DD4BF;} .status .warn {color:#F59E0B;}
.eyebrow {color:#2DD4BF; font-family:'JetBrains Mono',monospace; font-size:12px;
          letter-spacing:2px; font-weight:700; margin-bottom:10px;}
.h1 {font-size:46px; font-weight:800; line-height:1.05; margin:0;}
.h1 .cy {color:#2DD4BF;}
.sub {color:#7C8A9A; font-size:14px; max-width:620px; margin-top:14px; line-height:1.5;}
.card {background:linear-gradient(160deg,rgba(20,28,40,.7),rgba(12,18,28,.7));
       border:1px solid rgba(148,163,184,.12); border-radius:16px; padding:22px; height:100%;}
.card .lbl {color:#5B6B7E; font-size:11px; font-weight:700; letter-spacing:1.2px;}
.card .val {font-size:38px; font-weight:800; margin:8px 0 4px;}
.card .foot {color:#5B6B7E; font-size:12px;}
.stButton>button {background:rgba(20,28,40,.6); border:1px solid rgba(148,163,184,.14);
       color:#9AA8B6; border-radius:14px; padding:16px; width:100%; text-align:left;
       font-weight:600; transition:.2s;}
.stButton>button:hover {border-color:rgba(45,212,191,.5); color:#E5ECF3;}
[data-testid="stTextInput"] input {font-family:'JetBrains Mono',monospace; color:#2DD4BF;
       background:rgba(10,16,24,.8); border:1px solid rgba(148,163,184,.14);}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_lof():
    if not os.path.exists(LOF_PATH):
        return None, None
    p = joblib.load(LOF_PATH); return p["model"], p["threshold"]

def load_metrics():
    if not os.path.exists(RESULTS_PATH):
        return None
    return np.load(RESULTS_PATH, allow_pickle=True)

lof_model, lof_threshold = load_lof()
metrics = load_metrics()
if lof_model is None:
    st.error("No trained LOF model. Run preprocess → features → make_split → train.")
    st.stop()

if "mode" not in st.session_state:
    st.session_state.mode = "ESP32 Edge Feed"

# live header status (filled by edge feed)
node = st.session_state.get("hdr_node", "—")
ev   = st.session_state.get("hdr_events", 0)
an   = st.session_state.get("hdr_anom", 0)

st.markdown(f"""
<div class="topbar">
  <div class="brand">📡 SENTINEL<span class="dim">/vibration</span></div>
  <div class="status">node: <b>{node}</b> &nbsp;&nbsp; events: {ev} &nbsp;&nbsp;
       anomalies: <span class="warn">{an}</span></div>
</div>
<div class="eyebrow">● REAL-TIME TELEMETRY · V2.1</div>
<div class="h1">Structural Vibration<br><span class="cy">Anomaly Monitor</span></div>
<div class="sub">Two-tier detection pipeline. <b>Edge</b>: Isolation Forest on ESP32.
  <b>Cloud</b>: Local Outlier Factor on aggregated feature vectors.</div>
""", unsafe_allow_html=True)

# ── Metric cards (real numbers) ──────────────────────────────────────────────
auc = f1 = None
if metrics is not None:
    names = list(metrics["model_names"])
    if "Local Outlier Factor" in names:
        i = names.index("Local Outlier Factor")
        auc = float(metrics["holdout_auc"][i]); f1 = float(metrics["holdout_f1"][i])

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="medium")
c1.markdown(f"""<div class="card"><div class="lbl">LOF HELD-OUT AUC</div>
  <div class="val">{auc:.3f}</div><div class="foot">held-out recording split</div></div>"""
  if auc is not None else """<div class="card"><div class="lbl">LOF HELD-OUT AUC</div>
  <div class="val">—</div><div class="foot">run evaluate.py</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="card"><div class="lbl">LOF HELD-OUT F1</div>
  <div class="val">{f1:.3f}</div><div class="foot">precision/recall balanced</div></div>"""
  if f1 is not None else """<div class="card"><div class="lbl">LOF HELD-OUT F1</div>
  <div class="val">—</div><div class="foot">run evaluate.py</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="card"><div class="lbl">DECISION THRESHOLD</div>
  <div class="val">{lof_threshold:.4f}</div><div class="foot">contamination = 0.05</div></div>""",
  unsafe_allow_html=True)

# ── Mode selector ────────────────────────────────────────────────────────────
st.markdown("<div style='height:26px'></div><div class='lbl' style='color:#5B6B7E;font-size:11px;letter-spacing:1.2px;font-weight:700'>DEMO MODE</div>", unsafe_allow_html=True)
m1, m2, m3 = st.columns(3, gap="medium")
if m1.button("☰  CSV Replay\nReplay recorded CSV"): st.session_state.mode = "CSV Replay"
if m2.button("📶  Phyphox Live\nPhone IMU via WiFi"): st.session_state.mode = "Phyphox Live"
if m3.button("⚙  ESP32 Edge Feed\nLive IF scores over HTTP"): st.session_state.mode = "ESP32 Edge Feed"
mode = st.session_state.mode
st.markdown(f"<div style='color:#2DD4BF;font-family:JetBrains Mono;font-size:12px;margin-top:6px'>▸ active: {mode}</div>", unsafe_allow_html=True)
st.divider()


def edge_fig(xs, ys, thr, title_node):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
        line=dict(color=ACCENT, width=2), fill="tozeroy",
        fillcolor="rgba(45,212,191,.05)", name="IF score"))
    ax = [x for x, y in zip(xs, ys) if y < thr]; ay = [y for y in ys if y < thr]
    if ax: fig.add_trace(go.Scatter(x=ax, y=ay, mode="markers",
        marker=dict(color="#F43F5E", size=7), name="anomaly"))
    fig.add_hline(y=thr, line_dash="dash", line_color="#F59E0B")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7C8A9A"), xaxis_title="event #", yaxis_title="IF score",
        xaxis=dict(gridcolor="rgba(148,163,184,.06)"),
        yaxis=dict(gridcolor="rgba(148,163,184,.06)"),
        legend=dict(orientation="h", y=1.1, x=1, xanchor="right"))
    return fig


# ── CSV Replay ───────────────────────────────────────────────────────────────
if mode == "CSV Replay":
    up = st.file_uploader("Upload a Phyphox accelerometer CSV", type=["csv"])
    speed = st.slider("Replay delay per window (s)", 0.0, 0.5, 0.05)
    if up and st.button("▶ Run replay"):
        df = pd.read_csv(io.BytesIO(up.getvalue())).iloc[:, :4]
        df.columns = ["time"] + AXES; df = df.dropna()
        fs = estimate_fs(df["time"].values)
        sig = resample_to_target(df[AXES].values, fs, TARGET_FS)
        sig = sig - sig.mean(axis=0)
        st.caption(f"Detected ~{fs:.0f} Hz → resampled to {TARGET_FS} Hz")
        plot, stat = st.empty(), st.empty()
        xs, ys, na = [], [], 0
        total = (len(sig) - WINDOW_SIZE)//STEP_SIZE + 1
        for k, s0 in enumerate(range(0, len(sig)-WINDOW_SIZE+1, STEP_SIZE)):
            w = sig[s0:s0+WINDOW_SIZE]
            sc = float(lof_model.score_samples(extract_features_one_window(w).reshape(1,-1))[0])
            xs.append(k); ys.append(sc); na += sc < lof_threshold
            if k % 3 == 0 or k == total-1:
                plot.plotly_chart(edge_fig(xs, ys, lof_threshold, "CSV"), use_container_width=True)
                stat.write(f"window {k+1}/{total} • anomalies {na} ({100*na/(k+1):.1f}%)")
            time.sleep(speed)
        st.success(f"Done. {na}/{total} windows anomalous.")

# ── Phyphox Live ─────────────────────────────────────────────────────────────
elif mode == "Phyphox Live":
    st.markdown("In Phyphox: open an **Acceleration** experiment → menu → **Allow remote access**.")
    base = st.text_input("Phyphox remote URL", "http://192.168.1.50:8080")
    bufs = st.text_input("Buffer names (x,y,z)", "accX,accY,accZ")
    poll = st.slider("Poll interval (s)", 0.2, 2.0, 0.5)
    run = st.checkbox("● Stream live")
    bx, by, bz = [b.strip() for b in bufs.split(",")]
    plot, stat = st.empty(), st.empty()
    buf, xs, ys, k, na = [], [], [], 0, 0
    while run:
        try:
            r = requests.get(f"{base}/get?{bx}=full&{by}=full&{bz}=full", timeout=2).json()["buffer"]
            xx, yy, zz = r[bx]["buffer"], r[by]["buffer"], r[bz]["buffer"]
            for i in range(min(len(xx), len(yy), len(zz))): buf.append([xx[i], yy[i], zz[i]])
        except Exception as e:
            stat.error(f"Cannot reach Phyphox: {e}"); time.sleep(poll); continue
        while len(buf) >= WINDOW_SIZE:
            w = np.array(buf[:WINDOW_SIZE], float); w -= w.mean(axis=0)
            sc = float(lof_model.score_samples(extract_features_one_window(w).reshape(1,-1))[0])
            xs.append(k); ys.append(sc); k += 1; na += sc < lof_threshold; buf = buf[STEP_SIZE:]
        if ys:
            plot.plotly_chart(edge_fig(xs[-200:], ys[-200:], lof_threshold, "phone"), use_container_width=True)
            stat.write(f"windows {k} • anomalies {na}")
        time.sleep(poll)

# ── ESP32 Edge Feed ──────────────────────────────────────────────────────────
else:
    server = st.text_input("Telemetry endpoint", "http://127.0.0.1:8765")
    poll = st.slider("Poll interval (s)", 0.3, 2.0, 0.5)
    run = st.checkbox("● Show edge feed")
    st.caption("Start telemetry_server.py first. Before the MPU6050 arrives, run fake_esp32.py.")
    plot, stat = st.empty(), st.empty()
    while run:
        try:
            r = requests.get(f"{server}/api/v1/latest?n=200", timeout=2).json()
            events, thr = r.get("events", []), r.get("threshold", lof_threshold)
        except Exception as e:
            stat.error(f"Cannot reach telemetry server: {e}"); time.sleep(poll); continue
        if events:
            xs = list(range(len(events))); ys = [e["score"] for e in events]
            na = sum(e["anomaly"] for e in events)
            st.session_state.hdr_node = events[-1]["node"]
            st.session_state.hdr_events = len(events); st.session_state.hdr_anom = na
            plot.plotly_chart(edge_fig(xs, ys, thr, events[-1]["node"]), use_container_width=True)
            stat.write(f"node {events[-1]['node']} • events {len(events)} • anomalies {na}")
        else:
            stat.info("Connected — waiting for the ESP32 (or fake_esp32.py)…")
        time.sleep(poll)