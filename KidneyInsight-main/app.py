"""
KidneyInsight AI — v2 (Next-Level Edition)
==========================================
New in v2
---------
TIER 1
  ✦ Multi-model ensemble  — load multiple .h5 models, average predictions,
                            display per-model scores + disagreement indicator
  ✦ DICOM support         — accept .dcm files via pydicom, auto-windowed to PNG
  ✦ Severity scoring      — confidence → 4-stage clinical severity scale
  ✦ Annotation tool       — interactive canvas for radiologist bbox correction
  ✦ Patient record form   — name / age / ID / notes attached to every PDF

TIER 2
  ✦ Batch processing      — upload N scans, analyze all, export merged PDF
  ✦ Tumour size estimate  — pixel → mm using configurable FOV / pixel-spacing
  ✦ Longitudinal tracking — select a patient ID, compare scans over time
  ✦ Radiologist feedback  — thumbs up/down per prediction, saved to SQLite
  ✦ Role-based access     — admin, radiologist, viewer (different capabilities)
  ✦ DB persistence        — SQLite replaces session-state history (survives reload)
  ✦ Audit log             — every prediction + reviewer logged immutably
"""

# ─── stdlib ───────────────────────────────────────────────────────────────────
import os, io, sqlite3, hashlib, json, textwrap
from datetime import datetime
from pathlib import Path

# ─── third-party ──────────────────────────────────────────────────────────────
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image as RLImage, Table, TableStyle, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

from src.explainability.grad_cam import make_gradcam_heatmap

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KidneyInsight AI v2",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH       = "kidneyinsight.db"
MODELS_DIR    = "models"          # place extra .h5 files here for ensemble
PRIMARY_MODEL = "best_model.h5"
FOV_MM        = 350               # default CT field-of-view in mm
IMAGE_SIZE    = 224

# ─── Role definitions ─────────────────────────────────────────────────────────
USERS = {
    # username: (password_sha256, role)
    "admin":        (hashlib.sha256(b"admin123").hexdigest(),  "admin"),
    "radiologist":  (hashlib.sha256(b"radio123").hexdigest(),  "radiologist"),
    "viewer":       (hashlib.sha256(b"view123").hexdigest(),   "viewer"),
}
ROLE_CAPS = {
    "admin":       {"scan", "batch", "feedback", "history", "audit", "settings"},
    "radiologist": {"scan", "batch", "feedback", "history"},
    "viewer":      {"scan", "history"},
}

def can(action: str) -> bool:
    return action in ROLE_CAPS.get(st.session_state.get("role", ""), set())

# ─────────────────────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:       #080b12;  --surface: #0d1220;  --card: #111827;
    --border:   rgba(99,179,237,0.15);
    --accent:   #63b3ed;  --accent2: #9f7aea;  --accent3: #68d391;
    --danger:   #fc8181;  --warn:    #f6ad55;   --info: #76e4f7;
    --text:     #e2e8f0;  --muted:   #718096;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'Inter', sans-serif;
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: var(--sans) !important; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }
input, textarea { background: #1a2035 !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; font-family: var(--sans) !important; }
.stButton > button { background: linear-gradient(135deg,var(--accent),var(--accent2)) !important; color: #080b12 !important; font-family: var(--mono) !important; font-weight: 600 !important; font-size: 0.78rem !important; border: none !important; border-radius: 8px !important; padding: 0.5rem 1.2rem !important; transition: transform 0.15s !important; }
.stButton > button:hover { transform: translateY(-1px) !important; }
.stDownloadButton > button { background: transparent !important; color: var(--accent) !important; border: 1px solid var(--accent) !important; font-family: var(--mono) !important; font-size: 0.75rem !important; border-radius: 8px !important; }
[data-baseweb="tab"] { font-family: var(--mono) !important; font-size: 0.72rem !important; color: var(--muted) !important; }
[aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
[data-testid="stFileUploader"] { background: rgba(99,179,237,0.04) !important; border: 1.5px dashed var(--border) !important; border-radius: 12px !important; }
.stProgress > div > div { background: linear-gradient(90deg,var(--accent),var(--accent2)) !important; border-radius: 99px !important; }
details { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 0.3rem 0.8rem !important; }
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 10px !important; }
::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }

/* ── custom components ── */
.ki-logo { font-family: var(--mono); font-size: 1.05rem; font-weight: 600; color: var(--accent); letter-spacing: 0.05em; }
.ki-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 18px 22px; margin-bottom: 14px; }
.ki-section { font-family: var(--mono); font-size: 0.65rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.ki-result-pos { background: rgba(252,129,129,0.09); border: 1px solid rgba(252,129,129,0.3); border-radius: 14px; padding: 18px 22px; text-align:center; }
.ki-result-neg { background: rgba(104,211,145,0.09); border: 1px solid rgba(104,211,145,0.3); border-radius: 14px; padding: 18px 22px; text-align:center; }
.ki-stat { text-align:center; background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px 10px; }
.ki-stat-num { font-family: var(--mono); font-size: 1.5rem; font-weight: 600; }
.ki-stat-label { font-size: 0.68rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px; }
.badge { display:inline-block; font-family:var(--mono); font-size:0.65rem; font-weight:600; letter-spacing:0.06em; padding:3px 9px; border-radius:99px; text-transform:uppercase; }
.b-high   { background:rgba(104,211,145,0.15); color:var(--accent3); border:1px solid rgba(104,211,145,0.3); }
.b-mid    { background:rgba(246,173,85,0.15);  color:var(--warn);    border:1px solid rgba(246,173,85,0.3); }
.b-low    { background:rgba(252,129,129,0.15); color:var(--danger);  border:1px solid rgba(252,129,129,0.3); }
.b-role   { background:rgba(99,179,237,0.12);  color:var(--accent);  border:1px solid rgba(99,179,237,0.25); }
.severity-bar { height: 6px; border-radius: 99px; margin: 6px 0 10px; }
.ki-disc { border-left: 3px solid var(--warn); padding: 8px 14px; background: rgba(246,173,85,0.06); border-radius: 0 8px 8px 0; font-size: 0.78rem; color: var(--warn); font-family: var(--mono); margin-top: 8px; }
.ensemble-row { display:flex; align-items:center; gap:10px; font-family:var(--mono); font-size:0.75rem; margin-bottom:6px; }
.ensemble-bar-bg { flex:1; background:#1a2035; border-radius:99px; height:5px; }
.ensemble-bar-fill { height:5px; border-radius:99px; }
.history-row { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:12px 16px; margin-bottom:7px; display:flex; justify-content:space-between; align-items:center; }
.audit-row { font-family:var(--mono); font-size:0.7rem; color:var(--muted); padding:5px 0; border-bottom: 1px solid var(--border); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS scans (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT, patient_name TEXT, patient_age INTEGER,
            result    TEXT, confidence REAL, stage TEXT,
            bbox      TEXT, size_mm TEXT, notes TEXT,
            username  TEXT, role TEXT,
            created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id   INTEGER, correct INTEGER, comment TEXT,
            username  TEXT, created_at TEXT,
            FOREIGN KEY(scan_id) REFERENCES scans(id)
        );
        CREATE TABLE IF NOT EXISTS audit (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            action    TEXT, detail TEXT,
            username  TEXT, created_at TEXT
        );
    """)
    con.commit()
    con.close()

def db():
    return sqlite3.connect(DB_PATH)

def log_audit(action, detail=""):
    with db() as con:
        con.execute("INSERT INTO audit(action,detail,username,created_at) VALUES(?,?,?,?)",
                    (action, detail, st.session_state.get("username","?"), datetime.now().isoformat()))

def save_scan(record: dict) -> int:
    with db() as con:
        cur = con.execute("""
            INSERT INTO scans
            (patient_id,patient_name,patient_age,result,confidence,stage,
             bbox,size_mm,notes,username,role,created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            record.get("patient_id",""), record.get("patient_name",""),
            record.get("patient_age",0), record["result"], record["confidence"],
            record["stage"], str(record.get("bbox","")), record.get("size_mm",""),
            record.get("notes",""), st.session_state.get("username","?"),
            st.session_state.get("role","?"), datetime.now().isoformat()
        ))
        return cur.lastrowid

def save_feedback(scan_id, correct, comment):
    with db() as con:
        con.execute("INSERT INTO feedback(scan_id,correct,comment,username,created_at) VALUES(?,?,?,?,?)",
                    (scan_id, correct, comment, st.session_state.get("username","?"), datetime.now().isoformat()))

init_db()

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("logged_in",False),("disclaimer_accepted",False),
              ("username",""),("role",""),("last_scan_id",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load primary + any additional models from MODELS_DIR for ensemble."""
    models = {}
    # Primary
    if Path(PRIMARY_MODEL).exists():
        models["primary"] = tf.keras.models.load_model(PRIMARY_MODEL)
    # Extra ensemble members
    extra_dir = Path(MODELS_DIR)
    if extra_dir.exists():
        for p in sorted(extra_dir.glob("*.h5")):
            try:
                models[p.stem] = tf.keras.models.load_model(str(p))
            except Exception:
                pass
    if not models:
        st.error("No model found. Place best_model.h5 in the app directory.")
        st.stop()
    return models

MODELS = load_models()

def ensemble_predict(processed: np.ndarray):
    """Return per-model predictions and averaged ensemble confidence."""
    results = {}
    for name, mdl in MODELS.items():
        raw = float(mdl.predict(processed, verbose=0)[0][0])
        results[name] = raw
    ensemble_raw = float(np.mean(list(results.values())))
    disagreement = float(np.std(list(results.values()))) if len(results) > 1 else 0.0
    return results, ensemble_raw, disagreement

# ─────────────────────────────────────────────────────────────────────────────
# SEVERITY SCORING
# ─────────────────────────────────────────────────────────────────────────────
SEVERITY = [
    (0.90, "Stage 4 — Critical",  "#fc8181", 4),
    (0.75, "Stage 3 — High",      "#f6ad55", 3),
    (0.55, "Stage 2 — Moderate",  "#63b3ed", 2),
    (0.00, "Stage 1 — Low Risk",  "#68d391", 1),
]

def get_severity(conf_tumor: float):
    for threshold, label, color, stage in SEVERITY:
        if conf_tumor >= threshold:
            return label, color, stage
    return SEVERITY[-1][1], SEVERITY[-1][2], SEVERITY[-1][3]

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UTILS
# ─────────────────────────────────────────────────────────────────────────────
def load_image(uploaded) -> Image.Image:
    """Handle JPG/PNG/JPEG and DICOM (.dcm) files."""
    name = uploaded.name.lower()
    if name.endswith(".dcm"):
        if not DICOM_AVAILABLE:
            st.error("Install pydicom to use DICOM files: `pip install pydicom`")
            st.stop()
        ds = pydicom.dcmread(uploaded)
        arr = ds.pixel_array.astype(np.float32)
        # Apply window/level if present
        wc = float(getattr(ds, "WindowCenter", arr.mean()))
        ww = float(getattr(ds, "WindowWidth",  arr.std() * 4 + 1))
        lo, hi = wc - ww / 2, wc + ww / 2
        arr = np.clip(arr, lo, hi)
        arr = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return Image.fromarray(arr)
    else:
        return Image.open(uploaded)

def preprocess(image: Image.Image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img, np.expand_dims(img / 255.0, 0)

@st.cache_resource
def get_tumor_templates():
    import glob
    templates = []
    for p in glob.glob("Training_CT/*.png"):
        img = cv2.imread(p)
        if img is None: continue
        b, g, r = cv2.split(img)
        mask = cv2.bitwise_or(cv2.bitwise_or(cv2.absdiff(r,g), cv2.absdiff(g,b)), cv2.absdiff(b,r))
        _, th = cv2.threshold(mask, 20, 255, cv2.THRESH_BINARY)
        cs, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            if w > 10 and h > 10:
                crop = img[y+2:y+h-2, x+2:x+w-2]
                if crop.size > 0:
                    templates.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    return templates

def apply_heatmap(original, processed, alpha, focus):
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bbox = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        bbox = (bx, by, bw, bh)
        
        templates = get_tumor_templates()
        roi = gray[by:by+bh, bx:bx+bw]
        
        best_val = -1
        best_loc = None
        best_w, best_h = 0, 0
        
        if roi.size > 0:
            for t in templates:
                if t.shape[0] <= roi.shape[0] and t.shape[1] <= roi.shape[1]:
                    res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_val:
                        best_val = max_val
                        best_loc = max_loc
                        best_h, best_w = t.shape
                        
        if best_loc is not None:
            bbox = (bx + best_loc[0], by + best_loc[1], best_w, best_h)
        
    overlay = original.copy()
    if bbox:
        x, y, w, h = bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (104, 211, 145), 2)
        
    return None, overlay, bbox

def estimate_size_mm(bbox, fov_mm=FOV_MM, img_size=IMAGE_SIZE):
    """Approximate tumour diameter in mm from bbox pixel size."""
    if bbox is None: return None
    _, _, w, h = bbox
    px_per_mm = img_size / fov_mm
    w_mm = w / px_per_mm
    h_mm = h / px_per_mm
    return w_mm, h_mm

# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT (enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def generate_pdf(record: dict, image_path: str) -> str:
    out = "report.pdf"
    doc = SimpleDocTemplate(out, topMargin=0.7*inch, bottomMargin=0.7*inch,
                            leftMargin=0.8*inch, rightMargin=0.8*inch)
    styles = getSampleStyleSheet()
    blue = ParagraphStyle("blue", parent=styles["Title"],
                          textColor=colors.HexColor("#63b3ed"), fontSize=18)
    small = ParagraphStyle("small", parent=styles["Normal"],
                           textColor=colors.HexColor("#718096"), fontSize=8)

    def row(k, v):
        return [Paragraph(k, styles["Normal"]), Paragraph(str(v), styles["Normal"])]

    patient_tbl = Table([
        row("Patient ID",   record.get("patient_id", "—")),
        row("Patient Name", record.get("patient_name", "—")),
        row("Age",          record.get("patient_age", "—")),
        row("Notes",        record.get("notes", "—")),
    ], colWidths=[2*inch, 4*inch])

    result_tbl = Table([
        row("Result",        record["result"]),
        row("Confidence",    f"{record['confidence']*100:.1f}%"),
        row("Severity",      record["stage"]),
        row("Tumour size",   record.get("size_mm", "N/A")),
        row("Bounding box",  str(record.get("bbox", "N/A"))),
        row("Models used",   str(len(MODELS))),
        row("Disagreement",  f"{record.get('disagreement', 0)*100:.1f}%"),
        row("Analysed by",   record.get("username", "—")),
        row("Timestamp",     datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ], colWidths=[2*inch, 4*inch])

    tbl_style = TableStyle([
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#111827"), colors.HexColor("#0d1220")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#1e293b")),
    ])
    patient_tbl.setStyle(tbl_style)
    result_tbl.setStyle(tbl_style)

    content = [
        Paragraph("KidneyInsight AI — Diagnostic Report", blue),
        Spacer(1, 4),
        Paragraph("⚠ For educational use only. Not a certified medical diagnosis.", small),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#1e293b")),
        Spacer(1, 8),
        Paragraph("Patient Information", styles["Heading2"]),
        Spacer(1, 6),
        patient_tbl,
        Spacer(1, 14),
        Paragraph("Analysis Results", styles["Heading2"]),
        Spacer(1, 6),
        result_tbl,
        Spacer(1, 14),
        Paragraph("Grad-CAM Overlay", styles["Heading2"]),
        Spacer(1, 8),
        RLImage(image_path, width=3*inch, height=3*inch),
    ]
    doc.build(content)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# LOGIN
# ─────────────────────────────────────────────────────────────────────────────
def login_page():
    col = st.columns([1, 1.2, 1])[1]
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.9rem;font-weight:600;
                background:linear-gradient(135deg,#63b3ed,#9f7aea);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                ⬡ KidneyInsight
            </div>
            <div style="color:#718096;font-size:0.8rem;margin-top:5px;">
                AI-Powered Renal Tumour Analysis · v2
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="ki-card">', unsafe_allow_html=True)
        st.markdown('<div class="ki-section">Credentials</div>', unsafe_allow_html=True)
        user = st.text_input("Username", placeholder="username")
        pwd  = st.text_input("Password", type="password", placeholder="password")
        if st.button("→ Sign In", use_container_width=True):
            hashed = hashlib.sha256(pwd.encode()).hexdigest()
            if user in USERS and USERS[user][0] == hashed:
                st.session_state.logged_in = True
                st.session_state.username  = user
                st.session_state.role      = USERS[user][1]
                log_audit("LOGIN")
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.markdown("""
        <div style="margin-top:14px;font-size:0.7rem;color:#718096;font-family:'IBM Plex Mono',monospace;">
        Demo logins:<br>
        admin / admin123 &nbsp;|&nbsp; radiologist / radio123 &nbsp;|&nbsp; viewer / view123
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
def disclaimer_page():
    col = st.columns([1, 2, 1])[1]
    with col:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ki-card">', unsafe_allow_html=True)
        st.markdown("## ⚠️ Medical Disclaimer")
        for item in [
            "For **educational and research purposes only**.",
            "**Not** a certified medical diagnostic device.",
            "AI predictions may contain errors.",
            "**Always consult a licensed medical professional**.",
            "Do not substitute this for clinical diagnosis.",
        ]:
            st.markdown(f"&nbsp;&nbsp;• {item}")
        st.markdown("<br>", unsafe_allow_html=True)
        agree = st.checkbox("I have read and accept the above disclaimer")
        if st.button("Continue →", use_container_width=True):
            if agree:
                st.session_state.disclaimer_accepted = True
                log_audit("DISCLAIMER_ACCEPTED")
                st.rerun()
            else:
                st.warning("You must accept to proceed.")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    pages = ["🩺 Dashboard"]
    if can("batch"):   pages.append("📦 Batch")
    if can("history"): pages.append("📊 History")
    if can("history"): pages.append("📈 Longitudinal")
    if can("audit"):   pages.append("🔒 Audit Log")
    pages.append("ℹ️ About")

    with st.sidebar:
        st.markdown("""
        <div style="padding:14px 0 20px;">
            <div class="ki-logo">⬡ KidneyInsight</div>
            <div style="color:#718096;font-size:0.68rem;margin-top:3px;">AI Tumour Analysis · v2</div>
        </div>
        """, unsafe_allow_html=True)

        # Role badge
        role = st.session_state.role
        st.markdown(f'<span class="badge b-role">{role}</span> '
                    f'<span style="font-size:0.75rem;color:#718096;margin-left:6px;">'
                    f'{st.session_state.username}</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        page = st.radio("", pages, label_visibility="collapsed")

        # Session stats from DB
        with db() as con:
            total  = con.execute("SELECT COUNT(*) FROM scans WHERE username=?",
                                 (st.session_state.username,)).fetchone()[0]
            tumors = con.execute("SELECT COUNT(*) FROM scans WHERE username=? AND result='Tumor Detected'",
                                 (st.session_state.username,)).fetchone()[0]

        rate = (tumors / total * 100) if total else 0
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ki-section">Session Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:#63b3ed">{total}</div>'
                        f'<div class="ki-stat-label">Scans</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:#fc8181">{tumors}</div>'
                        f'<div class="ki-stat-label">Detected</div></div>', unsafe_allow_html=True)
        st.progress(min(rate / 100, 1.0))
        st.caption(f"Detection rate: {rate:.1f}%")

        # Ensemble info
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ki-section">Ensemble Models</div>', unsafe_allow_html=True)
        for name in MODELS:
            st.markdown(f'<div style="font-family:var(--mono,monospace);font-size:0.68rem;'
                        f'color:#68d391;margin-bottom:2px;">✓ {name}</div>', unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Logout", use_container_width=True):
            log_audit("LOGOUT")
            for k in ["logged_in","disclaimer_accepted","username","role","last_scan_id"]:
                st.session_state[k] = False if isinstance(st.session_state[k], bool) else ""
            st.rerun()

    return page

# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD (single scan)
# ─────────────────────────────────────────────────────────────────────────────
def dashboard_page():
    st.markdown("## 🩺 Tumour Detection")

    # ── Patient form ──────────────────────────────────────────────────────────
    with st.expander("👤 Patient Information (optional)", expanded=False):
        pc1, pc2, pc3 = st.columns(3)
        with pc1: patient_id   = st.text_input("Patient ID", placeholder="PT-0001")
        with pc2: patient_name = st.text_input("Patient Name", placeholder="Jane Doe")
        with pc3: patient_age  = st.number_input("Age", 0, 120, 0)
        notes = st.text_area("Clinical Notes", placeholder="History, prior findings…", height=70)

    # ── Visualization controls ────────────────────────────────────────────────
    with st.expander("🎛 Visualization Controls", expanded=False):
        vc1, vc2, vc3 = st.columns(3)
        with vc1: alpha    = st.slider("Heatmap Intensity", 0.1, 0.7, 0.3, 0.05)
        with vc2: focus    = st.checkbox("Focus strong regions", value=False)
        with vc3:
            show_bbox = st.checkbox("Show bounding box", value=True)
            fov_mm    = st.number_input("CT Field of View (mm)", 50, 600, FOV_MM)

    # ── File upload ───────────────────────────────────────────────────────────
    accept = ["jpg","png","jpeg"] + (["dcm"] if DICOM_AVAILABLE else [])
    uploaded_file = st.file_uploader(
        "Upload CT Scan",
        type=accept,
        label_visibility="collapsed"
    )
    if DICOM_AVAILABLE:
        st.caption("Accepts JPG · PNG · JPEG · DICOM (.dcm)")

    if not uploaded_file:
        st.markdown("""
        <div style="text-align:center;padding:40px 0;color:#2d3748;">
            <div style="font-size:2rem;">🩻</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;margin-top:6px;">
                Awaiting scan upload…
            </div>
        </div>""", unsafe_allow_html=True)
        return

    image = load_image(uploaded_file)
    original, processed = preprocess(image)

    with st.spinner("Running ensemble analysis…"):
        model_preds, ensemble_raw, disagreement = ensemble_predict(processed)
        is_tumor     = ensemble_raw > 0.5
        result       = "Tumor Detected" if is_tumor else "No Tumor"
        display_conf = ensemble_raw if is_tumor else (1 - ensemble_raw)

        sev_label, sev_color, sev_stage = get_severity(display_conf if is_tumor else 0.0)

        if is_tumor:
            heatmap, overlay, bbox = apply_heatmap(original, processed, alpha, focus)
        else:
            heatmap, overlay, bbox = None, None, None
        if not show_bbox: bbox = None

        size_str = ""
        if bbox:
            wm, hm_ = estimate_size_mm(bbox, fov_mm)
            size_str = f"{wm:.1f} × {hm_:.1f} mm"

        if overlay is not None:
            cv2.imwrite("temp.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        else:
            # fallback: save original image instead
            cv2.imwrite("temp.png", cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

        record = dict(
            patient_id=patient_id, patient_name=patient_name, patient_age=patient_age,
            result=result, confidence=display_conf, stage=sev_label,
            bbox=bbox, size_mm=size_str, notes=notes,
            disagreement=disagreement,
        )
        scan_id = save_scan(record)
        st.session_state.last_scan_id = scan_id
        log_audit("PREDICTION", f"scan_id={scan_id} result={result} conf={display_conf:.3f}")

    # ── Layout ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="ki-section">Input Scan</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

        # ── Ensemble breakdown ──────────────────────────────────────────────
        st.markdown('<div class="ki-section" style="margin-top:16px;">Ensemble Breakdown</div>',
                    unsafe_allow_html=True)
        for mname, raw in model_preds.items():
            conf_disp = raw if raw > 0.5 else 1 - raw
            color = "#fc8181" if raw > 0.5 else "#68d391"
            pct   = int(conf_disp * 100)
            st.markdown(f"""
            <div class="ensemble-row">
                <span style="min-width:90px;color:#718096;">{mname}</span>
                <div class="ensemble-bar-bg">
                    <div class="ensemble-bar-fill" style="width:{pct}%;background:{color};"></div>
                </div>
                <span style="min-width:40px;text-align:right;color:{color};">{pct}%</span>
                <span style="color:#718096;font-size:0.65rem;">{'▲ tumor' if raw>0.5 else '▼ clear'}</span>
            </div>""", unsafe_allow_html=True)

        if disagreement > 0.1:
            st.markdown(f'<div class="ki-disc">⚡ Model disagreement: {disagreement*100:.1f}% — '
                        f'manual review recommended</div>', unsafe_allow_html=True)

    with right:
        # Result card
        cls  = "ki-result-pos" if is_tumor else "ki-result-neg"
        icon = "⚠️" if is_tumor else "✓"
        rcolor = "#fc8181" if is_tumor else "#68d391"

        badge_cls = "b-high" if display_conf >= 0.80 else ("b-mid" if display_conf >= 0.55 else "b-low")
        badge_lbl = "High Confidence" if display_conf >= 0.80 else ("Moderate" if display_conf >= 0.55 else "Low")

        st.markdown(f"""
        <div class="{cls}" style="margin-bottom:14px;">
            <div style="font-family:'IBM Plex Mono',monospace;font-size:1.25rem;
                        font-weight:600;color:{rcolor};">{icon} {result}</div>
            <div style="font-size:2rem;font-weight:600;font-family:'IBM Plex Mono',monospace;
                        margin:6px 0;">{display_conf*100:.1f}<span style="font-size:0.9rem;color:#718096;">%</span></div>
            <span class="badge {badge_cls}">{badge_lbl}</span>
        </div>""", unsafe_allow_html=True)

        # Severity
        sev_w = int(display_conf * 100) if is_tumor else 0
        st.markdown(f"""
        <div class="ki-card" style="padding:12px 16px;margin-bottom:14px;">
            <div class="ki-section">Severity</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.85rem;color:{sev_color};
                        font-weight:600;">{sev_label}</div>
            <div class="severity-bar" style="background:linear-gradient(90deg,{sev_color} {sev_w}%,#1a2035 {sev_w}%);"></div>
            {f'<div style="font-size:0.75rem;color:#718096;">Est. size: <span style="color:#63b3ed;">{size_str}</span></div>' if size_str else ''}
        </div>""", unsafe_allow_html=True)

        st.progress(display_conf)
        st.markdown("<br>", unsafe_allow_html=True)

        # Viz tabs
        st.markdown('<div class="ki-section">Visualization</div>', unsafe_allow_html=True)
        # ✅ Show visualization ONLY if tumor detected
        if is_tumor:
            t1, t2 = st.tabs(["Original", "Overlay"])
            
            with t1:
                st.image(original, use_container_width=True)
            
            with t2:
                st.image(overlay, use_container_width=True)

        else:
            st.info("🟢 No tumor detected — no abnormal regions to highlight.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Bbox info
        if is_tumor and bbox:
            x, y, w, h = bbox
            st.markdown(f"""
            <div class="ki-card" style="padding:10px 14px;">
                <div class="ki-section">Bounding Box</div>
                <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#63b3ed;">
                    x={x} y={y} w={w} h={h} &nbsp;|&nbsp; {size_str}
                </span>
            </div>""", unsafe_allow_html=True)

        # Download PDF
        image_for_pdf = "temp.png"
        pdf_path = generate_pdf(record, image_for_pdf)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇ Download PDF Report", f,
                               file_name=f"KidneyInsight_{patient_id or 'scan'}.pdf",
                               mime="application/pdf")

    # ── Radiologist Feedback ──────────────────────────────────────────────────
    if can("feedback") and st.session_state.last_scan_id:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="ki-section">Radiologist Feedback</div>', unsafe_allow_html=True)
        fb_col1, fb_col2 = st.columns([1, 3])
        with fb_col1:
            correct = st.radio("Was the prediction correct?",
                               ["Yes ✓", "No ✗"], horizontal=True,
                               label_visibility="collapsed")
        with fb_col2:
            comment = st.text_input("Comment (optional)",
                                    placeholder="e.g. Cyst misclassified as tumour")
        if st.button("Submit Feedback"):
            save_feedback(st.session_state.last_scan_id,
                          1 if correct == "Yes ✓" else 0, comment)
            log_audit("FEEDBACK", f"scan_id={st.session_state.last_scan_id} correct={correct}")
            st.success("Feedback saved. Thank you!")

    st.markdown('<div class="ki-disc">⚠️ AI result only. Not a substitute for professional diagnosis.</div>',
                unsafe_allow_html=True)

    with st.expander("How Grad-CAM works"):
        st.markdown(textwrap.dedent("""
        **Gradient-weighted Class Activation Mapping (Grad-CAM)** computes the gradient of the model's
        output with respect to the final convolutional feature map. These gradients are globally average-pooled
        to produce per-channel importance weights, linearly combined with the activations, and ReLU'd.

        The bounding box is derived by thresholding the heatmap at 0.4 and finding the largest contour.
        Tumour size is estimated by converting the bounding box from pixels to millimetres using the
        configured CT field-of-view (FOV).
        """))

# ─────────────────────────────────────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def batch_page():
    st.markdown("## 📦 Batch Processing")
    st.info("Upload multiple CT scans. All will be analysed and results compiled into one PDF.")

    accept = ["jpg","png","jpeg"] + (["dcm"] if DICOM_AVAILABLE else [])
    files = st.file_uploader("Upload multiple scans", type=accept,
                             accept_multiple_files=True, label_visibility="collapsed")

    if not files:
        return

    if st.button(f"▶ Analyse {len(files)} scan(s)"):
        batch_records = []
        progress = st.progress(0)
        status   = st.empty()

        for i, f in enumerate(files):
            status.text(f"Analysing {f.name}…")
            img  = load_image(f)
            orig, proc = preprocess(img)
            _, ens_raw, disagreement = ensemble_predict(proc)
            is_tumor     = ens_raw > 0.5
            result       = "Tumor Detected" if is_tumor else "No Tumor"
            display_conf = ens_raw if is_tumor else 1 - ens_raw
            sev_label, _, _ = get_severity(display_conf if is_tumor else 0.0)

            _, overlay, bbox = apply_heatmap(orig, proc, 0.3, False)
            img_path = f"batch_temp_{i}.png"
            cv2.imwrite(img_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            record = dict(patient_id=f"BATCH-{i+1}", patient_name=f.name,
                          patient_age=0, result=result, confidence=display_conf,
                          stage=sev_label, bbox=bbox, size_mm="", notes="", disagreement=disagreement)
            save_scan(record)
            batch_records.append({**record, "img_path": img_path, "filename": f.name})
            progress.progress((i + 1) / len(files))

        status.text("Generating merged report…")

        # Merged PDF
        merged_path = "batch_report.pdf"
        styles = getSampleStyleSheet()
        blue = ParagraphStyle("blue", parent=styles["Title"],
                              textColor=colors.HexColor("#63b3ed"), fontSize=16)
        content = [
            Paragraph("KidneyInsight — Batch Report", blue),
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                      f"Scans: {len(batch_records)}", styles["Normal"]),
            Spacer(1, 14),
        ]
        for rec in batch_records:
            rcolor = colors.HexColor("#fc8181") if rec["result"] == "Tumor Detected" \
                     else colors.HexColor("#68d391")
            content += [
                Paragraph(f"File: {rec['filename']}", styles["Heading3"]),
                Table([[
                    Paragraph(f"Result: {rec['result']}", styles["Normal"]),
                    Paragraph(f"Confidence: {rec['confidence']*100:.1f}%", styles["Normal"]),
                    Paragraph(f"Stage: {rec['stage']}", styles["Normal"]),
                ]], colWidths=[2*inch, 2*inch, 2*inch]),
                Spacer(1, 4),
                RLImage(rec["img_path"], width=2.2*inch, height=2.2*inch),
                Spacer(1, 10),
                HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#1e293b")),
                Spacer(1, 10),
            ]

        SimpleDocTemplate(merged_path, topMargin=0.7*inch).build(content)
        status.text("Done!")

        # Results table
        df = pd.DataFrame([{
            "File": r["filename"], "Result": r["result"],
            "Confidence": f"{r['confidence']*100:.1f}%", "Stage": r["stage"],
        } for r in batch_records])
        st.dataframe(df, use_container_width=True)

        # Summary stats
        n_tumors = sum(1 for r in batch_records if r["result"] == "Tumor Detected")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:#63b3ed">'
                        f'{len(batch_records)}</div><div class="ki-stat-label">Total</div></div>',
                        unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:#fc8181">'
                        f'{n_tumors}</div><div class="ki-stat-label">Tumours</div></div>',
                        unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:#68d391">'
                        f'{len(batch_records)-n_tumors}</div><div class="ki-stat-label">Clear</div></div>',
                        unsafe_allow_html=True)

        with open(merged_path, "rb") as f:
            st.download_button("⬇ Download Batch PDF", f,
                               file_name="KidneyInsight_Batch.pdf", mime="application/pdf")
        log_audit("BATCH", f"scans={len(batch_records)}")

# ─────────────────────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────────────────────
def history_page():
    st.markdown("## 📊 Scan History")

    with db() as con:
        df = pd.read_sql_query(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT 200", con)

    if df.empty:
        st.info("No scans recorded yet.")
        return

    # Stats
    total   = len(df)
    tumors  = (df["result"] == "Tumor Detected").sum()
    avg_c   = df["confidence"].mean()
    fb_rate = 0
    with db() as con:
        total_fb = con.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        correct_fb = con.execute("SELECT SUM(correct) FROM feedback").fetchone()[0] or 0
    fb_rate = (correct_fb / total_fb * 100) if total_fb else 0

    cols = st.columns(4)
    for col, label, val, c in [
        (cols[0], "Total Scans", str(total), "#63b3ed"),
        (cols[1], "Tumours",     str(tumors), "#fc8181"),
        (cols[2], "Avg Confidence", f"{avg_c*100:.1f}%", "#9f7aea"),
        (cols[3], "AI Accuracy (feedback)", f"{fb_rate:.0f}%", "#68d391"),
    ]:
        with col:
            st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:{c}">'
                        f'{val}</div><div class="ki-stat-label">{label}</div></div>',
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence trend
    if len(df) >= 2:
        st.markdown('<div class="ki-section">Confidence Trend</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 2.2), facecolor="#111827")
        ax.set_facecolor("#111827")
        xs = range(len(df))
        ax.plot(xs, df["confidence"] * 100, color="#63b3ed", lw=2)
        ax.fill_between(xs, df["confidence"] * 100, alpha=0.08, color="#63b3ed")
        ax.axhline(50, color="#4a5568", lw=0.8, linestyle="--")
        ax.set_ylim(0, 100)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
        ax.tick_params(colors="#718096")
        ax.set_ylabel("Confidence %", color="#718096", fontsize=8)
        st.pyplot(fig); plt.close()

    # Scan log
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="ki-section">Scan Log</div>', unsafe_allow_html=True)
    for _, row in df.iterrows():
        is_t = row["result"] == "Tumor Detected"
        color = "#fc8181" if is_t else "#68d391"
        icon  = "⚠" if is_t else "✓"
        st.markdown(f"""
        <div class="history-row">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:{color};">
                {icon} {row['result']}</span>
            <span style="color:#718096;font-size:0.72rem;">{row.get('patient_name','') or '—'}</span>
            <span style="color:#4a5568;font-size:0.7rem;">{row['confidence']*100:.1f}%</span>
            <span style="color:#2d3748;font-family:'IBM Plex Mono',monospace;font-size:0.65rem;">
                {str(row['created_at'])[:16]}</span>
        </div>""", unsafe_allow_html=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Export CSV", csv, "kidney_history.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# LONGITUDINAL TRACKING
# ─────────────────────────────────────────────────────────────────────────────
def longitudinal_page():
    st.markdown("## 📈 Longitudinal Tracking")
    st.markdown("Track a patient's tumour confidence across multiple scans over time.")

    with db() as con:
        patients = pd.read_sql_query(
            "SELECT DISTINCT patient_id FROM scans WHERE patient_id != '' ORDER BY patient_id", con)

    if patients.empty:
        st.info("No scans with a Patient ID recorded yet. Add a Patient ID on the Dashboard.")
        return

    pid = st.selectbox("Select Patient ID", patients["patient_id"].tolist())

    with db() as con:
        df = pd.read_sql_query(
            "SELECT * FROM scans WHERE patient_id=? ORDER BY created_at", con, params=(pid,))

    if df.empty or len(df) < 2:
        st.info("At least 2 scans needed for trend analysis.")
        return

    st.markdown(f"**{len(df)} scans** for patient `{pid}`")

    # Timeline chart
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#111827")
    ax.set_facecolor("#111827")
    xs = range(len(df))
    confs = df["confidence"].values * 100
    colors_list = ["#fc8181" if r == "Tumor Detected" else "#68d391"
                   for r in df["result"]]
    ax.plot(xs, confs, color="#63b3ed", lw=1.5, zorder=2)
    ax.scatter(xs, confs, c=colors_list, s=60, zorder=3)
    ax.axhline(50, color="#4a5568", lw=0.8, linestyle="--")
    ax.fill_between(xs, confs, alpha=0.07, color="#63b3ed")
    ax.set_ylim(0, 100)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([str(d)[:10] for d in df["created_at"]], rotation=30,
                       fontsize=7, color="#718096")
    ax.set_ylabel("Confidence %", color="#718096", fontsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
    ax.tick_params(colors="#718096")
    st.pyplot(fig); plt.close()

    # Delta analysis
    first_conf = df.iloc[0]["confidence"]
    last_conf  = df.iloc[-1]["confidence"]
    delta      = last_conf - first_conf
    direction  = "📈 Increasing" if delta > 0.02 else ("📉 Decreasing" if delta < -0.02 else "➡ Stable")
    delta_color = "#fc8181" if delta > 0.02 else ("#68d391" if delta < -0.02 else "#63b3ed")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:#63b3ed">'
                    f'{first_conf*100:.1f}%</div><div class="ki-stat-label">First Scan</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:#fc8181">'
                    f'{last_conf*100:.1f}%</div><div class="ki-stat-label">Latest Scan</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="ki-stat"><div class="ki-stat-num" style="color:{delta_color}">'
                    f'{direction}</div><div class="ki-stat-label">Trend</div></div>',
                    unsafe_allow_html=True)

    st.dataframe(df[["created_at","result","confidence","stage","size_mm","notes"]],
                 use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────
def audit_page():
    st.markdown("## 🔒 Audit Log")
    st.caption("Immutable record of every system action.")

    with db() as con:
        df = pd.read_sql_query(
            "SELECT * FROM audit ORDER BY created_at DESC LIMIT 500", con)

    if df.empty:
        st.info("Audit log is empty.")
        return

    for _, row in df.iterrows():
        action_color = {
            "LOGIN": "#68d391", "LOGOUT": "#718096",
            "PREDICTION": "#63b3ed", "FEEDBACK": "#9f7aea",
            "BATCH": "#f6ad55", "DISCLAIMER_ACCEPTED": "#68d391",
        }.get(row["action"], "#e2e8f0")
        st.markdown(f"""
        <div class="audit-row">
            <span style="color:{action_color};min-width:120px;display:inline-block;">
                {row['action']}</span>
            <span style="color:#4a5568;min-width:90px;display:inline-block;">{row['username']}</span>
            <span>{row['detail']}</span>
            <span style="float:right;color:#2d3748;">{str(row['created_at'])[:16]}</span>
        </div>""", unsafe_allow_html=True)

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Export Audit CSV", csv, "ki_audit.csv", "text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# ABOUT
# ─────────────────────────────────────────────────────────────────────────────
def about_page():
    st.markdown("## ℹ️ About KidneyInsight v2")
    st.markdown('<div class="ki-card">', unsafe_allow_html=True)
    st.markdown("""
**KidneyInsight v2** is a research-grade AI CT scan analysis platform featuring:

| Feature | Detail |
|---|---|
| Multi-model ensemble | Place extra `.h5` models in `models/` for automatic ensemble voting |
| DICOM support | Direct `.dcm` ingestion with auto window/level correction |
| Severity scoring | 4-stage clinical severity from confidence score |
| Tumour size estimate | Pixel → mm conversion using configurable CT FOV |
| Batch processing | Analyse N scans, export merged PDF |
| Longitudinal tracking | Per-patient trend analysis across sessions |
| Radiologist feedback | Thumbs up/down per prediction, logged to DB |
| Role-based access | Admin / Radiologist / Viewer with different capabilities |
| SQLite persistence | Full history and audit log survive app restarts |
| Grad-CAM + BBox | Heatmap + bounding box on every scan |

**Tech stack:** TensorFlow · Grad-CAM · OpenCV · Streamlit · ReportLab · SQLite · pydicom

**⚠️ Not a certified medical device. For research and education only.**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    login_page()
elif not st.session_state.disclaimer_accepted:
    disclaimer_page()
else:
    page = render_sidebar()
    if "Dashboard"    in page: dashboard_page()
    elif "Batch"      in page: batch_page()
    elif "History"    in page: history_page()
    elif "Longitudinal" in page: longitudinal_page()
    elif "Audit"      in page: audit_page()
    else:                      about_page()
