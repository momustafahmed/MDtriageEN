
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load
import json

# ---------------- Basic setup ----------------
st.set_page_config(page_title="Health Triage Assistant", layout="centered")

# Subtle top spacing
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

# Load fitted pipeline and (optional) label encoder
pipe = load("models/best_pipe.joblib")
try:
    le = load("models/label_encoder.joblib")
except Exception:
    le = None

# Load feature schema if available (for correct column order/types)
CAT_FALLBACK = [
    "Has_Fever","Fever_Level","Fever_Duration_Level","Chills",
    "Has_Cough","Cough_Type","Cough_Duration_Level","Blood_Cough","Breath_Difficulty",
    "Has_Headache","Headache_Severity","Headache_Duration_Level","Photophobia","Neck_Stiffness",
    "Has_Abdominal_Pain","Pain_Location","Pain_Duration_Level","Nausea","Diarrhea",
    "Has_Fatigue","Fatigue_Severity","Fatigue_Duration_Level","Weight_Loss","Fever_With_Fatigue",
    "Has_Vomiting","Vomiting_Severity","Vomiting_Duration_Level","Blood_Vomit","Unable_To_Keep_Fluids",
    "Age_Group"
]
NUM_FALLBACK = ["Red_Flag_Count"]

try:
    with open("ui_assets/feature_schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    CAT_COLS = schema.get("cat_cols", CAT_FALLBACK)
    NUM_COLS = schema.get("num_cols", NUM_FALLBACK)
except Exception:
    CAT_COLS, NUM_COLS = CAT_FALLBACK, NUM_FALLBACK

EXPECTED_COLS = CAT_COLS + NUM_COLS

# --------------- Choices (English UI, mapped to Somali tokens) ---------------
# IMPORTANT: model expects Somali category tokens; we display English and map back.

YN_DISPLAY = ["Yes", "No"]
YN_MAP = {"Yes": "haa", "No": "maya"}

SEV_DISPLAY = ["mild", "moderate", "very severe"]
SEV_MAP = {"mild": "fudud", "moderate": "dhexdhexaad", "very severe": "aad u daran"}

COUGH_DISPLAY = ["dry", "wet/productive"]
COUGH_MAP = {"dry": "qalalan", "wet/productive": "qoyan"}

PAIN_DISPLAY = ["upper abdomen", "lower abdomen", "whole abdomen"]
PAIN_MAP = {"upper abdomen": "caloosha sare", "lower abdomen": "caloosha hoose", "whole abdomen": "caloosha oo dhan"}

AGE_DISPLAY = ["child", "adult", "elderly"]
AGE_MAP = {"child": "caruur", "adult": "qof weyn", "elderly": "waayeel"}

# Duration mapping: show English phrases, map to model tokens (Somali)
DUR_TOKEN_TO_DISPLAY = {
    "fudud": "≤ 1 day",
    "dhexdhexaad": "2–3 days",
    "dhexdhexaad ah": "2–3 days",
    "aad u daran": "≥ 3 days",
}
# When user picks a phrase, convert back to token for model input
DUR_DISPLAY_TO_TOKEN = {
    v: ("dhexdhexaad" if k.startswith("dhexdhexaad") else k)
    for k, v in DUR_TOKEN_TO_DISPLAY.items()
}
DUR_DISPLAY = list(dict.fromkeys(DUR_TOKEN_TO_DISPLAY.values()))

# --------------- Default one-sentence tips (English, keyed by Somali labels) ---------------
TRIAGE_TIPS = {
    "Xaalad fudud (Daryeel guri)":
        "Rest at home, drink plenty of fluids, eat light meals, consider simple pain/fever relievers if needed, and monitor symptoms for 24 hours. Seek care if symptoms worsen.",
    "Xaalad dhax dhaxaad eh (Bukaan socod)":
        "Visit a clinic within 24 hours for evaluation. Bring any prior prescriptions or records and keep hydrated.",
    "Xaalad dhax dhaxaad ah (Bukaan socod)":
        "Visit a clinic within 24 hours for evaluation. Bring any prior prescriptions or records and keep hydrated.",
    "Xaalad deg deg ah":
        "Go to the hospital immediately. Do not try home treatment. If possible, go with someone and bring prior prescriptions/records."
}
EXTRA_NOTICE = (
    "Special note: This is a general assessment to help you understand your condition and next steps. "
    "If you are worried about your health, contact a clinician."
)

# English display for model labels (Somali -> English)
LABEL_SO_TO_EN = {
    "Xaalad fudud (Daryeel guri)": "Mild condition (Home care)",
    "Xaalad dhax dhaxaad eh (Bukaan socod)": "Moderate condition (Outpatient)",
    "Xaalad dhax dhaxaad ah (Bukaan socod)": "Moderate condition (Outpatient)",
    "Xaalad deg deg ah": "Emergency",
}

# --------------- Helpers ---------------
def make_input_df(payload: dict) -> pd.DataFrame:
    """Ensure types are model-friendly (avoid isnan/type errors)."""
    row = {c: np.nan for c in EXPECTED_COLS}
    row.update(payload or {})

    # Categorical as object, numeric coerced
    for c in CAT_COLS:
        v = row.get(c, np.nan)
        if v is None:
            row[c] = np.nan
        else:
            s = str(v).strip()
            row[c] = np.nan if s == "" else s

    for c in NUM_COLS:
        try:
            row[c] = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        except Exception:
            row[c] = np.nan

    df_one = pd.DataFrame([row])
    for c in CAT_COLS:
        df_one[c] = df_one[c].astype("object")
    return df_one

def decode_label(y):
    """Return model label (Somali token) from output."""
    try:
        if le is not None and isinstance(y, (int, np.integer)):
            return le.inverse_transform([y])[0]
    except Exception:
        pass
    return str(y)

def triage_style_from_token(label_token: str):
    """
    Return (bg, text, border) for a light, readable card.
    Colors decided from Somali token (compatible with model).
    Green (home care), Amber (outpatient), Red (emergency).
    """
    t = (label_token or "").lower()
    if "deg deg" in t:
        return ("#FFEBEE", "#B71C1C", "#EF9A9A")
    if "dhax dhaxaad" in t:
        return ("#FFF8E1", "#8D6E00", "#FFD54F")
    return ("#E8F5E9", "#1B5E20", "#A5D6A7")

def render_select(label, wtype, key):
    placeholder = "Select"
    if wtype == "yn":
        disp = st.selectbox(label, YN_DISPLAY, index=None, placeholder=placeholder, key=key)
        if disp is None:
            return None
        return YN_MAP.get(disp)
    if wtype == "sev":
        disp = st.selectbox(label, SEV_DISPLAY, index=None, placeholder=placeholder, key=key)
        if disp is None:
            return None
        return SEV_MAP.get(disp)
    if wtype == "cough":
        disp = st.selectbox(label, COUGH_DISPLAY, index=None, placeholder=placeholder, key=key)
        if disp is None:
            return None
        return COUGH_MAP.get(disp)
    if wtype == "painloc":
        disp = st.selectbox(label, PAIN_DISPLAY, index=None, placeholder=placeholder, key=key)
        if disp is None:
            return None
        return PAIN_MAP.get(disp)
    if wtype == "dur":
        disp = st.selectbox(label, DUR_DISPLAY, index=None, placeholder=placeholder, key=key)
        if disp is None:
            return None
        return DUR_DISPLAY_TO_TOKEN.get(disp, disp)
    return None

# --------------- Symptom groups (English UI; internal flags unchanged) ---------------
SYMPTOMS = {
    "Fever": {
        "flag": "Has_Fever",
        "fields": [
            ("Fever_Level", "Fever severity", "sev"),
            ("Fever_Duration_Level", "Fever duration", "dur"),
            ("Chills", "Chills", "yn"),
        ],
    },
    "Cough": {
        "flag": "Has_Cough",
        "fields": [
            ("Cough_Type", "Type of cough", "cough"),
            ("Cough_Duration_Level", "Cough duration", "dur"),
            ("Blood_Cough", "Coughing blood", "yn"),
            ("Breath_Difficulty", "Shortness of breath", "yn"),
        ],
    },
    "Headache": {
        "flag": "Has_Headache",
        "fields": [
            ("Headache_Severity", "Headache severity", "sev"),
            ("Headache_Duration_Level", "Headache duration", "dur"),
            ("Photophobia", "Light sensitivity", "yn"),
            ("Neck_Stiffness", "Stiff neck", "yn"),
        ],
    },
    "Abdominal pain": {
        "flag": "Has_Abdominal_Pain",
        "fields": [
            ("Pain_Location", "Abdominal pain location", "painloc"),
            ("Pain_Duration_Level", "Abdominal pain duration", "dur"),
            ("Nausea", "Nausea", "yn"),
            ("Diarrhea", "Diarrhea", "yn"),
        ],
    },
    "Fatigue": {
        "flag": "Has_Fatigue",
        "fields": [
            ("Fatigue_Severity", "Fatigue severity", "sev"),
            ("Fatigue_Duration_Level", "Fatigue duration", "dur"),
            ("Weight_Loss", "Unintentional weight loss", "yn"),
        ],
    },
    "Vomiting": {
        "flag": "Has_Vomiting",
        "fields": [
            ("Vomiting_Severity", "Vomiting severity", "sev"),
            ("Vomiting_Duration_Level", "Vomiting duration", "dur"),
            ("Blood_Vomit", "Vomiting blood", "yn"),
            ("Unable_To_Keep_Fluids", "Unable to keep fluids down", "yn"),
        ],
    },
}
ALL_FLAGS = [v["flag"] for v in SYMPTOMS.values()]

# ---------------- UI ----------------
st.title("Health Triage Assistant")
st.markdown("Select one or more symptoms, then follow the questions about the symptoms you selected.")

colA, colB = st.columns(2)
with colA:
    age_disp = st.selectbox("Age group", AGE_DISPLAY, index=None, placeholder="Select")
with colB:
    st.caption("Skip any questions that don't apply to you.")

selected = st.multiselect("Your symptoms", list(SYMPTOMS.keys()), placeholder="Select symptom(s)")

# Build payload; default all Has_* to 'maya' (No)
payload = {}
if age_disp:
    payload["Age_Group"] = AGE_MAP[age_disp]
for flag in ALL_FLAGS:
    payload.setdefault(flag, "maya")

# Render follow-ups only for chosen symptoms; set their Has_* to 'haa'
for group in selected:
    cfg = SYMPTOMS[group]
    payload[cfg["flag"]] = "haa"  # user selected this symptom
    with st.expander(group, expanded=True):
        for (col, label, wtype) in cfg["fields"]:
            val = render_select(label, wtype, key=f"{group}:{col}")
            if val is not None:
                payload[col] = val

# Derived feature (fever + fatigue)
if (payload.get("Has_Fever") == "haa") and (payload.get("Has_Fatigue") == "haa"):
    payload["Fever_With_Fatigue"] = "haa"

# Red flags if model expects it
if "Red_Flag_Count" in NUM_COLS:
    def compute_red_flag_count(pl: dict) -> int:
        score = 0
        for k in ["Breath_Difficulty","Blood_Cough","Neck_Stiffness","Blood_Vomit","Unable_To_Keep_Fluids"]:
            if pl.get(k) == "haa":
                score += 1
        for sevk in ["Fever_Severity","Headache_Severity","Fatigue_Severity","Vomiting_Severity"]:
            v = pl.get(sevk) or pl.get(sevk.replace("_Severity","_Level"))
            if v == "aad u daran":
                score += 1
        return score
    payload["Red_Flag_Count"] = compute_red_flag_count(payload)

# ---------------- Predict ----------------
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
if st.button("Assess"):
    if not age_disp:
        st.warning("Please select your age group.")
    elif len(selected) == 0:
        st.warning("Please select at least one symptom.")
    else:
        x = make_input_df(payload)
        y_pred = pipe.predict(x)[0]
        label_token = decode_label(y_pred)  # Somali token from model
        label_en = LABEL_SO_TO_EN.get(label_token, label_token)

        # Light, modern result card with dynamic colors
        bg, fg, br = triage_style_from_token(label_token)

        st.markdown(
            f"""
            <div style="
                padding:18px;
                border-radius:14px;
                background:{bg};
                color:{fg};
                border:1px solid {br};
                box-shadow:0 2px 8px rgba(0,0,0,0.04);
                font-size:1.15rem;
                font-weight:700;
                margin-top:6px;
                margin-bottom:14px;">
                Result: {label_en}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tips card (light blue)
        tip_text = TRIAGE_TIPS.get(label_token) or "General advice: if you are concerned about your condition, contact a healthcare provider."
        st.markdown(
            f"""
            <div style="
                padding:16px;
                border-radius:12px;
                background:#E3F2FD;
                color:#0D47A1;
                border:1px solid #90CAF9;
                box-shadow:0 2px 8px rgba(0,0,0,0.03);
                font-size:1.02rem;">
                <strong>Advice:</strong> {tip_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:12px; color:#374151;'>" + EXTRA_NOTICE + "</div>", unsafe_allow_html=True)
