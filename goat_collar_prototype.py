"""
Goat Collar — Prototype
CSC 3118: Bio-Inspired Medicinal Plant Species Crowd-sourcing Using Goats
for Medical Heritage Preservation (Makerere University)

The combined, deployed system as a field dashboard:
  collar motion  -> M1 (integer random forest)  -> "is the goat eating?"
  camera photo   -> M2 (pruned INT8 CNN)         -> "which plant?"  (or "unidentified")
  every detection -> a review queue for a person to confirm or reject

WHAT IS REAL AND WHAT IS SIMULATED
  Real:      both models; every motion window is a held-out Kamminga test recording
             scored by M1; every photo is a held-out leaf test photo scored by M2.
  Simulated: the herd (goat names, collars, battery), the dates and times, and the
             GPS positions. There is no real collar deployment yet.

HOW TO RUN (from the folder that holds outputs_kamminga and plant_results):
    python -m pip install streamlit
    python -m streamlit run goat_collar_prototype.py
"""
import datetime as dt
import glob
import json
import os
import time

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from PIL import Image, ImageOps

# =============================================================================
# SETTINGS YOU MAY EDIT
# =============================================================================
M1_BUNDLE = os.path.join("outputs_kamminga", "deploy", "M1_collar_model.joblib")
M1_WINDOWS = os.path.join("outputs_kamminga", "windows_kamminga.npz")
M2_MODEL = os.path.join("plant_results", "deploy", "M2_plant_model.tflite")
M2_META = os.path.join("plant_results", "deploy", "M2_plant_model.json")
M2_SPLITS = os.path.join("plant_results", "splits")
# If the photo paths saved in the split CSV don't exist on this computer, the app
# also looks for the photos under these folders (using the CSV's "rel" column).
PHOTO_ROOTS = ["data/Raw", "Raw", "data", "."]

SITE = "Kabanyolo (simulated)"          # shown in the header
SITE_LAT, SITE_LON = 0.4650, 32.6120     # map centre
N_DAYS = 9                               # days of simulated field data
BOUTS_PER_DAY = 14                       # activity bouts sampled per goat per day
MAX_PHOTOS = 300                         # test photos scored once at start-up

GOATS = [  # name, collar, simulated battery %, collar offline from day index (None = active)
    ("Nakato", "C-01", 82, None), ("Kato", "C-02", 64, None), ("Babirye", "C-03", 91, None),
    ("Wasswa", "C-04", 47, None), ("Namukasa", "C-05", 73, None), ("Mukisa", "C-06", 9, 5),
]

SCIENTIFIC = {"aloe vera": "Aloe vera", "moringa": "Moringa oleifera", "tulsi": "Ocimum tenuiflorum",
              "neem": "Azadirachta indica", "hibiscus": "Hibiscus rosa-sinensis",
              "bitter leaf": "Vernonia amygdalina", "mango": "Mangifera indica",
              "guava": "Psidium guajava", "lemon": "Citrus limon"}

ACCENT = "#0d9488"          # teal: readable on both light and dark backgrounds
WARM = "#e07a2e"            # ochre: "really grazing" shading
MUTE = "#8a948f"            # neutral grey for chart text and rules
GRID = "rgba(128,128,128,0.18)"
SPECIES_COLORS = ["#14b8a6", "#f59e0b", "#6366f1", "#ef4444", "#22c55e", "#ec4899", "#0ea5e9", "#a8a29e"]

st.set_page_config(page_title="Goat Collar Prototype", page_icon="🐐", layout="wide")


# =============================================================================
# STYLE (works in light and dark mode)
# =============================================================================
LIGHT = """--bg:#f6f4ef; --surface:#ffffff; --surface2:#faf9f6; --border:#e7e2d8; --ink:#1b2320; --muted:#5b6762;
  --faint:#8a948f; --accent:#0f766e; --accent-soft:#dcf1ee; --warn:#b45309; --warn-soft:#fdf1e1;
  --bad:#b42318; --bad-soft:#fdeceb; --ok:#15803d; --ok-soft:#e2f4e7; --info:#4f46e5; --info-soft:#e9e8fd;"""
DARK = """--bg:#101416; --surface:#181e21; --surface2:#1d2428; --border:#2b3439; --ink:#e9eeec; --muted:#a6b1ad;
  --faint:#7f8a86; --accent:#2dd4bf; --accent-soft:#12332f; --warn:#f5a524; --warn-soft:#33270f;
  --bad:#f87171; --bad-soft:#3a1c1c; --ok:#4ade80; --ok-soft:#15301f; --info:#a5b4fc; --info-soft:#23244a;"""


def theme_type():
    try:
        return st.context.theme.type          # "light" / "dark" (Streamlit 1.46+)
    except Exception:
        return None


_tt = theme_type()
if _tt == "dark":
    VARS = f":root {{ {DARK} }}"
elif _tt == "light":
    VARS = f":root {{ {LIGHT} }}"
else:
    VARS = f":root {{ {LIGHT} }} @media (prefers-color-scheme: dark) {{ :root {{ {DARK} }} }}"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
{VARS}
.stApp {{ background: var(--bg); }}
[data-testid="stSidebar"] {{ background: var(--surface); border-right: 1px solid var(--border); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.4rem; max-width: 1380px; }}
h1, h2, h3, .brand, .kpi .val, .card-t, .panel-title {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
h2 {{ color: var(--ink) !important; font-weight: 750 !important; letter-spacing: -0.02em; }}
.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
          padding: 4px 0 18px 0; margin-bottom: 14px; border-bottom: 1px solid var(--border); }}
.brand {{ display:flex; align-items:center; gap:14px; }}
.badge {{ background: var(--accent); border-radius: 14px; width: 46px; height: 46px; display:flex;
         align-items:center; justify-content:center; }}
.brand-name {{ font-weight: 800; font-size: 1.35rem; color: var(--ink); line-height: 1.1; }}
.brand-sub {{ color: var(--muted); font-size: 0.92rem; font-family: inherit; font-weight: 500; }}
.chip {{ display:inline-flex; align-items:center; gap:8px; background: var(--accent-soft); color: var(--accent);
        border-radius: 999px; padding: 6px 14px; font-size: 0.86rem; font-weight: 600; }}
.chip .dot {{ width:8px; height:8px; border-radius:50%; background: var(--accent); }}
.lead {{ color: var(--muted); font-size: 1.02rem; margin-top: -8px; margin-bottom: 18px; max-width: 880px; }}
.crumb {{ color: var(--faint); font-size: 0.9rem; margin-bottom: -6px; }}
.crumb b {{ color: var(--accent); font-weight: 600; }}
.kpi {{ border-radius: 14px; padding: 16px 18px 14px 18px; border: 1px solid var(--border); background: var(--surface);
       border-top: 4px solid var(--accent); height: 132px; }}
.kpi .lab {{ font-size: 0.8rem; font-weight: 600; color: var(--muted); }}
.kpi .val {{ font-size: 2.15rem; font-weight: 800; color: var(--ink); margin-top: 2px; line-height: 1.2; }}
.kpi .sub {{ font-size: 0.84rem; color: var(--faint); }}
.kpi.warn {{ border-top-color: var(--warn); }} .kpi.warn .val {{ color: var(--warn); }}
.kpi.bad {{ border-top-color: var(--bad); }} .kpi.bad .val {{ color: var(--bad); }}
.kpi.info {{ border-top-color: var(--info); }}
.kpi.plain {{ border-top: 1px solid var(--border); background: var(--surface2); }}
.panel-title {{ font-weight: 700; font-size: 1.05rem; color: var(--ink); margin-bottom: 8px; }}
.section {{ font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; font-size: 1.1rem; color: var(--ink);
           margin: 6px 0 10px 0; }}
.sp-row {{ display:flex; justify-content:space-between; align-items:center; padding: 9px 0;
          border-bottom: 1px dashed var(--border); color: var(--ink); }}
.sp-row:last-child {{ border-bottom: none; }}
.sp-name {{ font-weight: 600; color: var(--ink); }}
.sp-sci {{ font-style: italic; color: var(--faint); font-size: 0.86rem; }}
.sp-n {{ font-weight: 800; color: var(--accent); text-align: right; font-size: 1.05rem; }}
.sp-g {{ color: var(--faint); font-size: 0.82rem; text-align: right; }}
.pill {{ display:inline-block; border-radius: 999px; padding: 2px 10px; font-size: 0.8rem; font-weight: 700; white-space: nowrap; }}
.pill.ok {{ background: var(--ok-soft); color: var(--ok); }}
.pill.mid {{ background: var(--warn-soft); color: var(--warn); }}
.pill.low {{ background: var(--bad-soft); color: var(--bad); }}
.pill.grey {{ background: var(--info-soft); color: var(--info); }}
.pill.rej {{ background: var(--bad-soft); color: var(--bad); }}
.card-h {{ display:flex; justify-content:space-between; align-items:flex-start; gap:8px; margin-top: 2px; }}
.card-t {{ font-weight: 700; font-size: 1.05rem; color: var(--ink); }}
.card-s {{ font-style: italic; color: var(--faint); font-size: 0.88rem; margin-top: -2px; }}
.card-m {{ display:flex; justify-content:space-between; align-items:center; margin-top: 10px; color: var(--muted); }}
.card-m b {{ color: var(--ink); }}
.card-d {{ color: var(--faint); font-size: 0.86rem; margin-top: 4px; }}
.card-truth {{ color: var(--faint); font-size: 0.78rem; margin-top: 2px; opacity: 0.9; }}
.goat-h {{ display:flex; justify-content:space-between; align-items:center; }}
.note {{ color: var(--faint); font-size: 0.85rem; }}
.side-brand {{ color: var(--muted); font-size: 0.82rem; line-height: 1.55; }}
.side-brand b {{ color: var(--ink); }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background: var(--surface); border-radius: 14px !important;
    border-color: var(--border) !important; }}
div[data-testid="stImage"] img {{ border-radius: 10px; }}
.stAppDeployButton, [data-testid="stAppDeployButton"] {{ display: none; }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background: var(--accent-soft) !important; }}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--accent) !important; font-weight: 700; }}
[data-testid="stSidebarNavLink"] span {{ font-size: 0.98rem; }}
div[data-baseweb="slider"] div[role="slider"] {{ background-color: var(--accent) !important; box-shadow: none !important; }}
[data-testid="stSliderThumbValue"] {{ color: var(--accent) !important; }}
.stButton button:hover {{ border-color: var(--accent) !important; color: var(--accent) !important; }}
.bar {{ height: 7px; background: var(--border); border-radius: 99px; margin-top: 10px; overflow: hidden; }}
.bar > div {{ height: 100%; border-radius: 99px; }}
.legend-dot {{ display:inline-block; width:11px; height:11px; border-radius:50%; margin-right:8px; }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# M1: FEATURES AND THE COLLAR MODEL (feature code identical to the notebooks)
# =============================================================================
def extract_features(Xb, fs=200):
    Xb = Xb.astype(np.float64)
    n, T, _ = Xb.shape
    acc_mag = np.linalg.norm(Xb[:, :, 0:3], axis=2, keepdims=True)
    gyr_mag = np.linalg.norm(Xb[:, :, 3:6], axis=2, keepdims=True)
    S = np.concatenate([Xb, acc_mag, gyr_mag], axis=2)
    mean = S.mean(1); std = S.std(1); cen = S - mean[:, None, :]
    q25, q50, q75, q90 = np.percentile(S, [25, 50, 75, 90], axis=1)
    mn, mx = S.min(1), S.max(1)
    rms = np.sqrt((S ** 2).mean(1)); eps = 1e-9
    skew = (cen ** 3).mean(1) / (std ** 3 + eps)
    kurt = (cen ** 4).mean(1) / (std ** 4 + eps) - 3.0
    jerk = np.abs(np.diff(S, axis=1)).mean(1)
    zcr = (np.diff(np.signbit(cen), axis=1)).mean(1)
    power = np.abs(np.fft.rfft(cen, axis=1)) ** 2
    freqs = np.fft.rfftfreq(T, 1.0 / fs)
    total = power.sum(1) + eps
    dom = freqs[power[:, 1:, :].argmax(1) + 1]
    bands = [power[:, (freqs >= a) & (freqs < b), :].sum(1) / total
             for a, b in [(0, 2), (2, 5), (5, 10), (10, 20), (20, fs / 2)]]
    p = power / total[:, None, :]
    ent = -(p * np.log(p + eps)).sum(1) / np.log(power.shape[1])
    per_signal = [mean, q50, mn, mx, mx - mn, std, q75 - q25, rms,
                  skew, kurt, jerk, zcr, dom] + bands + [ent, q90]
    block = np.stack(per_signal, axis=2).reshape(n, -1)
    pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
    corrs = [(cen[:, :, i] * cen[:, :, j]).mean(1) / (std[:, i] * std[:, j] + eps) for i, j in pairs]
    return np.concatenate([block, np.stack(corrs, axis=1)], axis=1).astype(np.float32)


@st.cache_resource
def load_m1():
    return joblib.load(M1_BUNDLE)


def m1_probability(bundle, windows):
    """P(eating) for each 2-second window, whatever kind of model M1_7 chose."""
    F = extract_features(windows, bundle["sample_rate_hz"])[:, bundle["feature_idx"]]
    m, kind = bundle["model"], bundle["kind"]
    if kind == "int_forest":
        q = np.round((np.clip(F, bundle["f_min"], bundle["f_min"] + bundle["f_scale"] * 65535)
                      - bundle["f_min"]) / bundle["f_scale"]) - 32768
        leaves = m.apply(q.astype(np.float32))
        votes = np.stack([bundle["leaf_q"][t][leaves[:, t]] for t in range(leaves.shape[1])], axis=1)
        return votes.astype(np.float64).mean(1) / 255.0
    if kind == "tree_regressor":
        return np.clip(m.predict(F), 0, 1)
    return m.predict_proba(F)[:, 1]


@st.cache_data(show_spinner="Scoring the held-out collar recordings with M1…")
def m1_test_scores():
    """M1 once over every held-out test window; the herd simulation reuses these scores."""
    d = np.load(M1_WINDOWS, allow_pickle=True)
    te = d["splits"] == "test"
    X, labels, segs = d["X"][te], d["labels"][te], d["segments"][te]
    t0 = time.time()
    p = m1_probability(load_m1(), X)
    ms = 1000 * (time.time() - t0) / max(1, len(X))
    return p, labels.astype(str), segs, ms


# =============================================================================
# M2: THE PLANT MODEL (TensorFlow Lite, integer)
# =============================================================================
@st.cache_resource
def load_m2():
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    it = Interpreter(model_path=M2_MODEL)
    it.allocate_tensors()
    return it, json.load(open(M2_META))


def m2_species_probs(img_uint8):
    """img_uint8: (size, size, 3). Returns probabilities per species."""
    it, meta = load_m2()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    x = img_uint8[None].astype(np.float32)
    if i["dtype"] != np.float32:
        s, z = i["quantization"]
        x = np.clip(np.round(x / s + z), np.iinfo(i["dtype"]).min, np.iinfo(i["dtype"]).max).astype(i["dtype"])
    it.set_tensor(i["index"], x)
    it.invoke()
    y = it.get_tensor(o["index"]).astype(np.float32)[0]
    if o["dtype"] != np.float32:
        s, z = o["quantization"]
        y = (y - z) * s
    to_sp = np.array(meta["folder_to_species"])
    return np.array([y[to_sp == j].sum() for j in range(len(meta["species"]))])


def prepare_photo(pil_img, size):
    return np.asarray(pil_img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _resolve(path, rel):
    if isinstance(path, str) and os.path.exists(path):
        return path
    for root in PHOTO_ROOTS:
        cand = os.path.join(root, str(rel))
        if os.path.exists(cand):
            return cand
    return None


@st.cache_data(show_spinner="Scoring the held-out leaf photos with M2…")
def m2_test_scores():
    """M2 once over (up to MAX_PHOTOS) held-out test photos."""
    _, meta = load_m2()
    files = sorted(glob.glob(os.path.join(M2_SPLITS, "*_split.csv")))
    if not files:
        return pd.DataFrame()
    d = pd.read_csv(files[0])
    d = d[d["split"] == "test"].copy()
    d["file"] = [_resolve(p, r) for p, r in zip(d["path"], d.get("rel", d["path"]))]
    d = d[d["file"].notna()]
    if len(d) > MAX_PHOTOS:
        d = d.sample(MAX_PHOTOS, random_state=0)
    folders = list(meta["folders"])
    rows = []
    for _, r in d.iterrows():
        try:
            img = Image.open(r["file"])
        except Exception:
            continue
        probs = m2_species_probs(prepare_photo(img, meta["image_size"]))
        true_sp = meta["folder_to_species"][folders.index(r["class"])] if r["class"] in folders else -1
        rows.append({"file": r["file"], "true_sp": int(true_sp), "pred_sp": int(probs.argmax()),
                     "conf": float(probs.max())})
    return pd.DataFrame(rows)


@st.cache_data
def thumbnail(path, w=420, h=260):
    return ImageOps.fit(Image.open(path).convert("RGB"), (w, h), Image.BILINEAR)


# =============================================================================
# THE SIMULATED HERD (real model outputs, simulated who / when / where)
# =============================================================================
@st.cache_data(show_spinner="Replaying the herd…")
def simulate_herd(tau, seed):
    p, labels, segs, _ = m1_test_scores()
    photos = m2_test_scores()
    rng = np.random.default_rng(seed)
    uniq = np.unique(segs)
    first_label = {s: labels[segs == s][0] for s in uniq}
    graze = [s for s in uniq if first_label[s] == "grazing"]
    other = [s for s in uniq if first_label[s] != "grazing"]
    seg_idx = {s: np.where(segs == s)[0][::2] for s in uniq}   # non-overlapping 2 s blocks
    today = dt.date.today()
    days = [today - dt.timedelta(days=N_DAYS - 1 - k) for k in range(N_DAYS)]
    dets, daylogs = [], {}
    for gi, (goat, collar, _, off_from) in enumerate(GOATS):
        g_lat = SITE_LAT + rng.normal(0, 0.004)
        g_lon = SITE_LON + rng.normal(0, 0.004)
        for di, day in enumerate(days):
            if off_from is not None and di >= off_from:
                continue
            n_g = max(1, BOUTS_PER_DAY // 3)
            chosen = list(rng.choice(graze, size=min(n_g, len(graze)), replace=False)) + \
                list(rng.choice(other, size=min(BOUTS_PER_DAY - n_g, len(other)), replace=False))
            rng.shuffle(chosen)
            idx = np.concatenate([seg_idx[s] for s in chosen])
            pp, truth = p[idx], labels[idx]
            eat = pp >= tau
            daylogs[(goat, str(day))] = (pp, truth)
            # one detection (photo) per feeding bout = run of consecutive "eating" windows
            n = len(idx)
            k = 0
            c_lat, c_lon = g_lat + rng.normal(0, 0.0015), g_lon + rng.normal(0, 0.0015)
            while k < n:
                if not eat[k]:
                    k += 1
                    continue
                a = k
                while k < n and eat[k]:
                    k += 1
                minute = int(7 * 60 + (a / n) * 11 * 60 + rng.integers(0, 20))
                ts = dt.datetime.combine(day, dt.time(min(minute // 60, 23), minute % 60))
                if len(photos):
                    ph = photos.iloc[int(rng.integers(len(photos)))]
                    file, pred, conf, tsp = ph["file"], int(ph["pred_sp"]), float(ph["conf"]), int(ph["true_sp"])
                else:
                    file, pred, conf, tsp = None, -1, 0.0, -1
                dets.append({"id": f"{collar}-{day:%m%d}-{a:04d}", "goat": goat, "collar": collar,
                             "time": ts, "day": day, "pred_sp": pred, "conf": conf, "true_sp": tsp,
                             "file": file, "bout_s": 2 * (k - a),
                             "was_grazing": bool((truth[a:k] == "grazing").mean() >= 0.5),
                             "p_eat": float(pp[a:k].mean()),
                             "lat": c_lat + rng.normal(0, 0.0008), "lon": c_lon + rng.normal(0, 0.0008)})
    return pd.DataFrame(dets), daylogs, [str(d) for d in days]


# =============================================================================
# HELPERS
# =============================================================================
def nice(name):
    return str(name).replace("_", " ").strip().title()


def sci(name):
    key = str(name).replace("_", " ").lower().replace(" bg", "").strip()
    return SCIENTIFIC.get(key, "Scientific name to be confirmed")


def status_of(row, theta, review_thr):
    manual = st.session_state.reviews.get(row["id"])
    if manual:
        return manual
    if row["conf"] < theta:
        return "Needs review"
    return "Auto-confirmed" if row["conf"] >= review_thr else "Needs review"


def conf_pill(c, theta, review_thr):
    cls = "ok" if c >= review_thr else ("mid" if c >= theta else "low")
    return f'<span class="pill {cls}">{c:.0%} confidence</span>'


def status_pill(s):
    cls = {"Auto-confirmed": "ok", "Confirmed": "ok", "Needs review": "grey", "Rejected": "rej"}[s]
    return f'<span class="pill {cls}">{s}</span>'


def kpi(label, value, sub="", tone=""):
    return (f'<div class="kpi {tone}"><div class="lab">{label}</div><div class="val">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def page_head(title, lead=None, crumb=True):
    if crumb:
        st.markdown(f'<div class="crumb">Overview / <b>{title}</b></div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if lead:
        st.markdown(f'<div class="lead">{lead}</div>', unsafe_allow_html=True)


def detection_card(r, key_prefix, allow_review=True):
    with st.container(border=True):
        if r["file"]:
            st.image(thumbnail(r["file"]), width="stretch")
        st.markdown(
            f'<div class="card-h"><div><div class="card-t">{r["species"]}</div>'
            f'<div class="card-s">{r["sci"]}</div></div>{conf_pill(r["conf"], THETA, REVIEW)}</div>'
            f'<div class="card-m"><span>{r["goat"]} · {r["collar"]}</span>{status_pill(r["status"])}</div>'
            f'<div class="card-d">{r["time"]:%d %b, %H:%M} · feeding bout {r["bout_s"]} s</div>'
            f'<div class="card-truth">demo check — photo is really {r["truth"]}'
            f'{" ✓" if r["correct"] else " ✗"}</div>', unsafe_allow_html=True)
        if allow_review and r["status"] == "Needs review":
            a, b = st.columns(2)
            if a.button("Confirm", key=f"{key_prefix}c{r['id']}", width="stretch"):
                st.session_state.reviews[r["id"]] = "Confirmed"; st.rerun()
            if b.button("Reject", key=f"{key_prefix}r{r['id']}", width="stretch"):
                st.session_state.reviews[r["id"]] = "Rejected"; st.rerun()


def area_chart(df, x, y, title_y, height=300):
    grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                        stops=[alt.GradientStop(color="rgba(13,148,136,0.0)", offset=0),
                               alt.GradientStop(color="rgba(13,148,136,0.35)", offset=1)])
    base = alt.Chart(df).encode(x=alt.X(f"{x}:N", title=None, axis=alt.Axis(labelAngle=0)),
                                y=alt.Y(f"{y}:Q", title=title_y, axis=alt.Axis(tickMinStep=1)),
                                tooltip=[x, y])
    return (base.mark_area(interpolate="monotone", color=grad) +
            base.mark_line(interpolate="monotone", color=ACCENT, strokeWidth=2.5) +
            base.mark_circle(color=ACCENT, size=46)).properties(height=height).configure_view(strokeWidth=0) \
        .configure_axis(gridColor=GRID, domain=False)


# =============================================================================
# LOAD EVERYTHING
# =============================================================================
missing = [p for p in [M1_BUNDLE, M1_WINDOWS, M2_MODEL, M2_META] if not os.path.exists(p)]
if missing:
    st.error("Model files not found:\n\n" + "\n".join(f"- `{p}`" for p in missing) +
             "\n\nOpen a terminal in the folder that contains `outputs_kamminga` and `plant_results` "
             "(for example `cd Downloads`) and run the app from there.")
    st.stop()

if "reviews" not in st.session_state:
    st.session_state.reviews = {}

M1B = load_m1()
_, META = load_m2()
NAMES = [nice(n) for n in META["names"]]

with st.sidebar:
    st.markdown("&nbsp;")
    with st.expander("⚙️ Model settings", expanded=False):
        TAU = st.slider("M1 eating threshold τ", 0.05, 0.95, float(M1B["threshold"]), 0.01,
                        help="Chosen on validation data in M1_7. Lower = more photos, fewer missed bites.")
        THETA = st.slider("M2 'unidentified' threshold θ", 0.20, 0.99, float(max(0.2, META["theta"])), 0.01,
                          help="Below this the plant is recorded as Unidentified species.")
        REVIEW = st.slider("Auto-confirm above", 0.50, 1.00, 0.90, 0.01,
                           help="Detections at or above this confidence skip the review queue.")
        SEED = st.number_input("Simulation seed", 0, 999, 7)
        if st.button("Clear my reviews", width="stretch"):
            st.session_state.reviews = {}
    st.markdown(f'<div class="side-brand"><b>On the collar</b><br>M1 motion · {M1B["name"]}<br>'
                f'M2 plant · {META["model"]} ({META["file_mb"]:.2f} MB)</div>', unsafe_allow_html=True)

DETS, DAYLOGS, DAYS = simulate_herd(TAU, int(SEED))
if len(DETS):
    DETS["named"] = DETS["conf"] >= THETA
    DETS["species"] = [NAMES[p] if ok else "Unidentified species" for p, ok in zip(DETS["pred_sp"], DETS["named"])]
    DETS["sci"] = [sci(META["names"][p]) if ok else "Pending expert review"
                   for p, ok in zip(DETS["pred_sp"], DETS["named"])]
    DETS["truth"] = [NAMES[t] if t >= 0 else "unknown" for t in DETS["true_sp"]]
    DETS["correct"] = DETS["pred_sp"] == DETS["true_sp"]
    DETS["status"] = [status_of(r, THETA, REVIEW) for _, r in DETS.iterrows()]
    DETS = DETS.sort_values("time", ascending=False).reset_index(drop=True)

st.markdown(
    '<div class="topbar"><div class="brand"><div class="badge">'
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none"><ellipse cx="14" cy="11" rx="10" ry="7" '
    'stroke="#fff" stroke-width="2.6"/><rect x="10" y="16" width="8" height="8" rx="2" fill="#fff"/>'
    '<circle cx="14" cy="20" r="1.6" fill="#0f766e"/></svg></div>'
    f'<div><div class="brand-name">Goat Collar</div><div class="brand-sub">Prototype field console · '
    f'pilot herd, {SITE}</div></div></div>'
    '<div class="chip"><span class="dot"></span>Real models · simulated field data</div></div>',
    unsafe_allow_html=True)


# =============================================================================
# PAGES
# =============================================================================
def page_overview():
    st.markdown("## Overview")
    st.markdown('<div class="lead">Detections from accelerometer-triggered camera captures across the pilot '
                'herd, identified on the collar and queued for plant-species review.</div>',
                unsafe_allow_html=True)
    active = sum(1 for g in GOATS if g[3] is None)
    confirmed = DETS[DETS["status"].isin(["Auto-confirmed", "Confirmed"]) & DETS["named"]]
    c = st.columns(4)
    c[0].markdown(kpi("Total detections", len(DETS), f"last {N_DAYS} days", ), unsafe_allow_html=True)
    c[1].markdown(kpi("Active collars", f"{active} / {len(GOATS)}", "reporting today", "warn"),
                  unsafe_allow_html=True)
    c[2].markdown(kpi("Species identified", confirmed["species"].nunique(), f"of {len(NAMES)} the model knows", "info"),
                  unsafe_allow_html=True)
    c[3].markdown(kpi("Awaiting review", int((DETS["status"] == "Needs review").sum()),
                      "open the Detections page", "bad"), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.75, 1])
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Detections per day</div>', unsafe_allow_html=True)
            per_day = DETS.groupby("day").size().reindex(pd.to_datetime(DAYS).date, fill_value=0)
            dfc = pd.DataFrame({"day": [d.strftime("%m-%d") for d in per_day.index],
                                "detections": per_day.values})
            st.altair_chart(area_chart(dfc, "day", "detections", None), width="stretch")
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Top candidate species</div>', unsafe_allow_html=True)
            g = DETS.groupby(["species", "sci"]).agg(n=("id", "size"), goats=("goat", "nunique")) \
                .reset_index().sort_values("n", ascending=False).head(6)
            html = "".join(f'<div class="sp-row"><div><div class="sp-name">{r.species}</div>'
                           f'<div class="sp-sci">{r.sci}</div></div><div><div class="sp-n">{r.n}</div>'
                           f'<div class="sp-g">{r.goats} goats</div></div></div>' for r in g.itertuples())
            st.markdown(html, unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="section">How well the collar is doing</div>', unsafe_allow_html=True)
    eat_all = np.concatenate([v[0] >= TAU for v in DAYLOGS.values()])
    truth_all = np.concatenate([v[1] == "grazing" for v in DAYLOGS.values()])
    _, _, _, ms = m1_test_scores()
    c = st.columns(4)
    c[0].markdown(kpi("Camera on", f"{eat_all.mean():.0%}", "of the time (always-on = 100%)", "plain"),
                  unsafe_allow_html=True)
    c[1].markdown(kpi("Grazing caught", f"{(eat_all & truth_all).sum() / max(1, truth_all.sum()):.0%}",
                      "of real grazing time photographed", "plain"), unsafe_allow_html=True)
    c[2].markdown(kpi("Photos while grazing", f"{(eat_all & truth_all).sum() / max(1, eat_all.sum()):.0%}",
                      "camera triggers that were real grazing", "plain"), unsafe_allow_html=True)
    c[3].markdown(kpi("Plant ID correct", f"{DETS['correct'].mean():.0%}" if len(DETS) else "—",
                      "M2 on the photos taken", "plain"), unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="section">Latest detections</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (_, r) in zip(cols, DETS.head(4).iterrows()):
        with col:
            detection_card(r, "ov", allow_review=False)


def page_detections():
    page_head("Detections")
    with st.container(border=True):
        f = st.columns([1, 1.4, 1.1, 1.3])
        goat = f[0].selectbox("Goat", ["All goats"] + [g[0] for g in GOATS])
        sp = f[1].selectbox("Species", ["All species"] + NAMES + ["Unidentified species"])
        stat = f[2].selectbox("Review status", ["All statuses", "Needs review", "Auto-confirmed",
                                                "Confirmed", "Rejected"])
        minc = f[3].slider("Min. confidence", 0, 100, 0, 5, format="%d%%")
    d = DETS
    if goat != "All goats":
        d = d[d["goat"] == goat]
    if sp != "All species":
        d = d[d["species"] == sp]
    if stat != "All statuses":
        d = d[d["status"] == stat]
    d = d[d["conf"] * 100 >= minc]
    per_page = 12
    pages = max(1, int(np.ceil(len(d) / per_page)))
    top = st.columns([3, 1])
    top[0].markdown(f'<div class="lead">{len(d)} of {len(DETS)} events shown</div>', unsafe_allow_html=True)
    pg = top[1].number_input("Page", 1, pages, 1, label_visibility="collapsed") if pages > 1 else 1
    chunk = d.iloc[(pg - 1) * per_page: pg * per_page]
    for i in range(0, len(chunk), 3):
        cols = st.columns(3)
        for col, (_, r) in zip(cols, chunk.iloc[i:i + 3].iterrows()):
            with col:
                detection_card(r, "dt")
    if len(d) == 0:
        st.info("No detections match these filters.")


def page_species():
    page_head("Species", "Every plant the collars have photographed, with how many goats chose it. "
                         "A plant visited by many goats is a stronger candidate for follow-up with herders "
                         "and traditional healers.")
    g = DETS.groupby(["species", "sci"]).agg(
        detections=("id", "size"), goats=("goat", "nunique"), avg_conf=("conf", lambda c: 100 * c.mean()),
        confirmed=("status", lambda s: int(s.isin(["Auto-confirmed", "Confirmed"]).sum())),
        first_seen=("time", "min"), last_seen=("time", "max")).reset_index() \
        .sort_values("detections", ascending=False)
    left, right = st.columns([1, 1.25])
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Detections by species</div>', unsafe_allow_html=True)
            ch = alt.Chart(g).mark_bar(cornerRadiusEnd=6, color=ACCENT, height=22).encode(
                y=alt.Y("species:N", sort="-x", title=None),
                x=alt.X("detections:Q", title="detections"),
                tooltip=["species", "detections", "goats"])
            txt = ch.mark_text(align="left", dx=5, color=MUTE, fontWeight="bold").encode(text="detections:Q")
            st.altair_chart((ch + txt).properties(height=40 * len(g) + 30).configure_view(strokeWidth=0)
                            .configure_axis(gridColor=GRID, domain=False), width="stretch")
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Species register</div>', unsafe_allow_html=True)
            show = g.rename(columns={"species": "Species", "sci": "Scientific name", "detections": "Detections",
                                     "goats": "Goats", "avg_conf": "Avg. confidence", "confirmed": "Confirmed",
                                     "first_seen": "First seen", "last_seen": "Last seen"})
            st.dataframe(show, hide_index=True, width="stretch", column_config={
                "Avg. confidence": st.column_config.NumberColumn(format="%.0f%%"),
                "First seen": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
                "Last seen": st.column_config.DatetimeColumn(format="D MMM, HH:mm")})
    st.markdown('<div class="note">Scientific names are looked up from the class name; confirm them with a '
                'botanist before they go into the heritage record.</div>', unsafe_allow_html=True)


def page_map():
    page_head("Map", "Where each detection happened. Colours show the species. Positions are simulated "
                     "until the collars carry GPS.")
    sp_list = sorted(DETS["species"].unique())
    colors = {s: SPECIES_COLORS[i % len(SPECIES_COLORS)] for i, s in enumerate(sp_list)}
    d = DETS.copy()
    d["color"] = [[int(colors[s][k:k + 2], 16) for k in (1, 3, 5)] + [210] for s in d["species"]]
    d["when"] = d["time"].dt.strftime("%d %b %H:%M")
    d["confp"] = (d["conf"] * 100).round().astype(int)
    left, right = st.columns([3, 1])
    with left:
        layer = pdk.Layer("ScatterplotLayer", d, get_position="[lon, lat]", get_fill_color="color",
                          get_radius=22, radius_min_pixels=5, pickable=True, stroked=True,
                          get_line_color=[255, 255, 255], line_width_min_pixels=1)
        st.pydeck_chart(pdk.Deck(layers=[layer], map_style="dark" if theme_type() == "dark" else "light",
                                 initial_view_state=pdk.ViewState(latitude=SITE_LAT, longitude=SITE_LON,
                                                                  zoom=14.2),
                                 tooltip={"text": "{species}\n{goat} · {when}\n{confp}% confidence"}),
                        height=560)
    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Legend</div>', unsafe_allow_html=True)
            counts = DETS["species"].value_counts()
            st.markdown("".join(
                f'<div class="sp-row"><div><span class="legend-dot" style="background:{colors[s]}"></span>{s}</div>'
                f'<div class="sp-n">{counts[s]}</div></div>' for s in sp_list), unsafe_allow_html=True)


def page_goats():
    page_head("Goats", "Each collar in the pilot herd, and a replay of what its motion model saw on any day.")
    cols = st.columns(3)
    for i, (goat, collar, batt, off) in enumerate(GOATS):
        gd = DETS[DETS["goat"] == goat]
        logs = [v for (g, _), v in DAYLOGS.items() if g == goat]
        eat = np.concatenate([v[0] >= TAU for v in logs]) if logs else np.array([])
        tr = np.concatenate([v[1] == "grazing" for v in logs]) if logs else np.array([])
        active = off is None
        with cols[i % 3]:
            with st.container(border=True):
                pill = '<span class="pill ok">Active</span>' if active else '<span class="pill low">Offline</span>'
                last = gd["time"].max().strftime("%d %b, %H:%M") if len(gd) else "—"
                st.markdown(
                    f'<div class="goat-h"><div><div class="card-t">{goat}</div>'
                    f'<div class="card-s" style="font-style:normal">collar {collar} · battery {batt}%</div></div>'
                    f'{pill}</div>'
                    f'<div class="card-m"><span>Detections</span><b>{len(gd)}</b></div>'
                    f'<div class="card-m"><span>Species visited</span><b>{gd.loc[gd.named, "species"].nunique()}</b></div>'
                    f'<div class="card-m"><span>Camera on</span><b>{eat.mean():.0%}</b></div>'
                    f'<div class="card-m"><span>Grazing caught</span>'
                    f'<b>{(eat & tr).sum() / max(1, tr.sum()):.0%}</b></div>'
                    f'<div class="card-d">last detection {last}</div>'
                    f'<div class="bar"><div style="width:{batt}%;background:{"var(--accent)" if batt >= 30 else "var(--bad)"}">'
                    f'</div></div>', unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        st.markdown('<div class="panel-title">Collar replay — what M1 saw</div>', unsafe_allow_html=True)
        a, b = st.columns(2)
        goat = a.selectbox("Goat", [g[0] for g in GOATS], key="rg")
        days = [d for (g, d) in DAYLOGS if g == goat]
        day = b.selectbox("Day", days[::-1], key="rd")
        pp, truth = DAYLOGS[(goat, day)]
        t = np.arange(len(pp)) * 2
        df = pd.DataFrame({"seconds": t, "P(eating)": pp, "grazing": (truth == "grazing").astype(float)})
        shade = alt.Chart(df).mark_area(color=WARM, opacity=0.22, interpolate="step-after").encode(
            x=alt.X("seconds:Q", title="seconds of sampled activity"), y=alt.Y("grazing:Q", title="P(eating)"))
        line = alt.Chart(df).mark_line(color=ACCENT, strokeWidth=1.8).encode(x="seconds:Q", y="P(eating):Q")
        rule = alt.Chart(pd.DataFrame({"y": [TAU]})).mark_rule(strokeDash=[5, 4], color=MUTE).encode(y="y:Q")
        st.altair_chart((shade + line + rule).properties(height=240).configure_view(strokeWidth=0)
                        .configure_axis(gridColor=GRID, domain=False), width="stretch")
        st.markdown(f'<div class="note">Green line: M1\'s P(eating) for each 2-second window. Orange shading: '
                    f'when the goat was really grazing. Dashed line: τ = {TAU:.2f}; above it the camera fires. '
                    f'The motion is real held-out test data from other goats (Kamminga et al.).</div>',
                    unsafe_allow_html=True)


def page_identify():
    page_head("Identify a leaf", "Run the plant model on any photo, the same way the collar does.")
    left, right = st.columns([1, 1.3])
    with left:
        with st.container(border=True):
            up = st.file_uploader("Upload a leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"])
            if st.button("…or use a random test photo", width="stretch"):
                ph = m2_test_scores()
                if len(ph):
                    st.session_state.sample_photo = ph.sample(1).iloc[0]["file"]
            img = Image.open(up) if up is not None else (
                Image.open(st.session_state.sample_photo) if st.session_state.get("sample_photo") else None)
            if img is not None:
                st.image(img.convert("RGB"), width="stretch")
    with right:
        if img is None:
            st.info("Upload a photo or pick a test photo to see M2's answer.")
            return
        t0 = time.time()
        probs = m2_species_probs(prepare_photo(img, META["image_size"]))
        ms = 1000 * (time.time() - t0)
        j = int(probs.argmax())
        with st.container(border=True):
            if probs[j] >= THETA:
                st.markdown(f'<div class="card-t" style="font-size:1.5rem">{NAMES[j]}</div>'
                            f'<div class="card-s">{sci(META["names"][j])}</div><br>'
                            f'{conf_pill(probs[j], THETA, REVIEW)}', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="card-t" style="font-size:1.5rem">Unidentified species</div>'
                            f'<div class="card-s">best guess {NAMES[j]}, below θ = {THETA:.2f}</div><br>'
                            f'{conf_pill(probs[j], THETA, REVIEW)}', unsafe_allow_html=True)
            df = pd.DataFrame({"species": NAMES, "probability": probs, "top": [k == j for k in range(len(NAMES))]})
            ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=20).encode(
                y=alt.Y("species:N", sort="-x", title=None),
                x=alt.X("probability:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
                color=alt.condition("datum.top", alt.value(ACCENT), alt.value("rgba(128,128,128,0.35)")),
                tooltip=["species", alt.Tooltip("probability:Q", format=".1%")])
            rule = alt.Chart(pd.DataFrame({"x": [THETA]})).mark_rule(strokeDash=[5, 4], color=MUTE).encode(x="x:Q")
            st.altair_chart((ch + rule).properties(height=36 * len(NAMES) + 20).configure_view(strokeWidth=0)
                            .configure_axis(gridColor=GRID, domain=False), width="stretch")
            st.markdown(f'<div class="note">Decided in {ms:.1f} ms by the {META["file_mb"]:.2f} MB integer model.'
                        f'</div>', unsafe_allow_html=True)


def page_models():
    page_head("Models", "How the two models on the collar were chosen and compressed.")
    with st.container(border=True):
        st.markdown(f"""
| Stage | Four models trained | Best | Four optimisations | Deployed |
|---|---|---|---|---|
| **M1: is the goat eating?** (collar motion) | Random Forest, 1D CNN, LSTM, CNN-LSTM | Random Forest | compaction, feature selection, integer quantisation, distillation | **{M1B['name']}**: {M1B['setting']} · test PR-AUC {M1B['test_pr_auc']:.3f} |
| **M2: which plant?** (camera photo) | Simple CNN, ResNet-18, MobileNetV2, EfficientNet-B0 | {META['base_model']} | dynamic-range INT8, full INT8, structured pruning, distillation | **{META['model']}**: {META['setting']} · {META['file_mb']:.2f} MB · test accuracy {META['test_accuracy']:.1%} |
""")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Why gate the camera with motion?</div>', unsafe_allow_html=True)
            st.markdown("A camera running all day would drain a collar battery quickly. The motion model runs "
                        "continuously at very low cost and switches the camera on only while the goat is eating "
                        "(the *Camera on* figure on the Overview page).")
    with b:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Why a review queue?</div>', unsafe_allow_html=True)
            st.markdown("A wrong plant name puts a false record into the heritage database. Low-confidence photos "
                        "are recorded as *Unidentified species*, and anything below the auto-confirm level waits "
                        "for a person to confirm or reject it.")
    with st.container(border=True):
        st.markdown('<div class="panel-title">What is real in this console</div>', unsafe_allow_html=True)
        st.markdown("**Real:** both models, every motion window (held-out Kamminga test recordings) and every photo "
                    "(held-out leaf test photos). **Simulated:** the herd, collar IDs, battery levels, dates, times "
                    "and GPS positions. **Limitation:** both models were trained on public data (other goats, "
                    "hand-held leaf photos); accuracy on photos from a goat's collar is expected to be lower, and "
                    "measuring that gap is the next phase of the project.")


pg = st.navigation([
    st.Page(page_overview, title="Overview", icon=":material/grid_view:", default=True),
    st.Page(page_detections, title="Detections", icon=":material/list:", url_path="detections"),
    st.Page(page_species, title="Species", icon=":material/eco:", url_path="species"),
    st.Page(page_map, title="Map", icon=":material/map:", url_path="map"),
    st.Page(page_goats, title="Goats", icon=":material/pets:", url_path="goats"),
    st.Page(page_identify, title="Identify a leaf", icon=":material/photo_camera:", url_path="identify"),
    st.Page(page_models, title="Models", icon=":material/memory:", url_path="models"),
])
pg.run()
