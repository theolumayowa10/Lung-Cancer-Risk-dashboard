# Power-BI style Streamlit dashboard for Lung Cancer Risk Classification
# File: streamlit_lung_cancer_app.py
# Data path used (exact): C:\Users\DELL\Desktop\ccc\lung_cancer_examples.csv

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score
)

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = r"C:\Users\DELL\Desktop\ccc\lung_cancer_examples.csv"
LOG_MODEL = "model_logistic.pkl"
RF_MODEL = "model_random_forest.pkl"
SCALER_FILE = "scaler.pkl"

st.set_page_config(page_title="Lung Cancer Prediction — Dashboard", layout="wide")

# Simple CSS to make a cleaner card/metric look
st.markdown(
    """
    <style>
    .metric {padding:18px; border-radius:8px; background-color:#ffffff; box-shadow: 0 1px 6px rgba(0,0,0,0.06);}
    .kpi {background:#0f172a; color:#fff; padding:14px; border-radius:8px;}
    .section-title {font-size:26px; font-weight:700; margin-bottom:8px;}
    .small-muted {color:#6b7280; font-size:13px;}
    .centered {display:flex; align-items:center; justify-content:center;}
    .big-plot {height:520px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# UTILITIES
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df

def to_binary(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {'1','yes','y','true','t'}:
        return 1
    if s in {'0','no','n','false','f'}:
        return 0
    try:
        val = float(s)
        return 1 if val != 0 else 0
    except:
        if 'yes' in s:
            return 1
        if 'no' in s:
            return 0
    return np.nan

def preprocess(df):
    # ensure required columns exist
    expected = ['Name','Surname','Age','Smokes','AreaQ','Alkhol','Result']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    df_proc = df[['Age','Smokes','AreaQ','Alkhol','Result']].copy()
    df_proc['Age'] = pd.to_numeric(df_proc['Age'], errors='coerce')
    df_proc['Smokes'] = df_proc['Smokes'].apply(to_binary)
    df_proc['AreaQ'] = df_proc['AreaQ'].apply(to_binary)
    df_proc['Alkhol'] = df_proc['Alkhol'].apply(to_binary)
    df_proc['Result'] = df_proc['Result'].apply(to_binary)
    df_proc = df_proc.dropna().reset_index(drop=True)
    df_proc[['Smokes','AreaQ','Alkhol','Result']] = df_proc[['Smokes','AreaQ','Alkhol','Result']].astype(int)
    return df_proc

def train_and_save(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_scaled, y_train)

    rf_model = RandomForestClassifier(n_estimators=250, random_state=42)
    rf_model.fit(X_train, y_train)

    joblib.dump(log_model, LOG_MODEL)
    joblib.dump(rf_model, RF_MODEL)
    joblib.dump(scaler, SCALER_FILE)

    return log_model, rf_model, scaler

def load_models_if_exist():
    if os.path.exists(LOG_MODEL) and os.path.exists(RF_MODEL) and os.path.exists(SCALER_FILE):
        return joblib.load(LOG_MODEL), joblib.load(RF_MODEL), joblib.load(SCALER_FILE)
    return None, None, None

def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler is not None:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test
    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': prec,
        'recall': rec,
        'cm': cm,
        'y_prob': y_prob,
        'y_pred': y_pred
    }

# -------------------------
# Load & preprocess data
# -------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Data file not found at: {DATA_PATH}. Please check the path.")
    st.stop()

raw = load_data(DATA_PATH)
try:
    df = preprocess(raw)
except Exception as e:
    st.error(f"Preprocessing error: {e}")
    st.stop()

# Sidebar - Filters + model controls
st.sidebar.title("Controls")
st.sidebar.markdown("Upload a new CSV or adjust filters below")

# Data filters (for interactivity)
min_age, max_age = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age range", min_age, max_age, (min_age, max_age))
smoke_filter = st.sidebar.multiselect("Smokes", options=[0,1], default=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
areaq_filter = st.sidebar.multiselect("AreaQ", options=[0,1], default=[0,1], format_func=lambda x: "Yes" if x==1 else "No")
alkhol_filter = st.sidebar.multiselect("Alkhol", options=[0,1], default=[0,1], format_func=lambda x: "Yes" if x==1 else "No")

download_group = st.sidebar.container()
with download_group:
    if os.path.exists(LOG_MODEL):
        with open(LOG_MODEL,'rb') as f:
            st.sidebar.download_button("Download logistic model", f, file_name=LOG_MODEL)
    if os.path.exists(RF_MODEL):
        with open(RF_MODEL,'rb') as f:
            st.sidebar.download_button("Download random forest", f, file_name=RF_MODEL)
    if os.path.exists(SCALER_FILE):
        with open(SCALER_FILE,'rb') as f:
            st.sidebar.download_button("Download scaler", f, file_name=SCALER_FILE)

st.sidebar.markdown("---")
st.sidebar.markdown("Built by theolumayowa — Portfolio project")

# Apply filters
df_f = df.query("Age >= @age_range[0] and Age <= @age_range[1]")
df_f = df_f[df_f['Smokes'].isin(smoke_filter)]
df_f = df_f[df_f['AreaQ'].isin(areaq_filter)]
df_f = df_f[df_f['Alkhol'].isin(alkhol_filter)]

# -------------------------
# Train / Load models
# -------------------------
X = df[['Age','Smokes','AreaQ','Alkhol']]
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_model, rf_model, scaler = load_models_if_exist()
if log_model is None:
    with st.spinner("Training models..."):
        log_model, rf_model, scaler = train_and_save(X_train, y_train)
    st.success("Models trained and saved.")

# Evaluate both models
eval_lr = evaluate_model(log_model, X_test, y_test, scaler=scaler)
eval_rf = evaluate_model(rf_model, X_test, y_test, scaler=None)

# -------------------------
# TOP KPIs
# -------------------------
st.markdown("<div class='section-title'>Lung Cancer Risk — Dashboard</div>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns([1,1,1,1])
kpi_style = "class='metric'"

with k1:
    st.markdown(f"<div {kpi_style}><div style='font-size:15px;color:#6b7280'>Logistic AUC</div><div style='font-size:24px;font-weight:700;color:#0b5cff'>{eval_lr['auc']:.3f}</div></div>", unsafe_allow_html=True)

with k2:
    st.markdown(f"<div {kpi_style}><div style='font-size:15px;color:#6b7280'>RandomForest AUC</div><div style='font-size:24px;font-weight:700;color:#0b5cff'>{eval_rf['auc']:.3f}</div></div>", unsafe_allow_html=True)

with k3:
    st.markdown(f"<div {kpi_style}><div style='font-size:15px;color:#6b7280'>Logistic Accuracy</div><div style='font-size:24px;font-weight:700;color:#0b5cff'>{eval_lr['accuracy']:.3f}</div></div>", unsafe_allow_html=True)

with k4:
    st.markdown(f"<div {kpi_style}><div style='font-size:15px;color:#6b7280'>RandomForest Accuracy</div><div style='font-size:24px;font-weight:700;color:#0b5cff'>{eval_rf['accuracy']:.3f}</div></div>", unsafe_allow_html=True)

st.markdown("")

# -------------------------
# LAYOUT: Left column (charts) | Right column (details & prediction)
# -------------------------
left_col, right_col = st.columns([2.2,1])

with left_col:
    # Target distribution donut
    st.subheader("Target Distribution")
    target_counts = df_f['Result'].value_counts().rename({0:'Negative',1:'Positive'})
    fig_d = px.pie(values=target_counts.values, names=target_counts.index, hole=0.55,
                   title="Result distribution (filtered)", color_discrete_sequence=px.colors.qualitative.Prism)
    fig_d.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_d, use_container_width=True)

    # Age distribution
    st.subheader("Age Distribution (Filtered)")
    fig_age = px.histogram(df_f, x='Age', nbins=15, title="Age distribution", marginal='box')
    st.plotly_chart(fig_age, use_container_width=True)

    # ROC comparison (Plotly)
    st.subheader("ROC Curve Comparison")
    fpr_lr, tpr_lr, _ = roc_curve(y_test, eval_lr['y_prob'])
    fpr_rf, tpr_rf, _ = roc_curve(y_test, eval_rf['y_prob'])
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode='lines', name=f'Logistic (AUC={eval_lr["auc"]:.3f})'))
    fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode='lines', name=f'RF (AUC={eval_rf["auc"]:.3f})'))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
    fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=450)
    st.plotly_chart(fig_roc, use_container_width=True)

    # Feature importance (RF)
    st.subheader("Feature Importance (Random Forest)")
    fi = rf_model.feature_importances_
    fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi}).sort_values('importance', ascending=False)
    fig_fi = px.bar(fi_df, x='feature', y='importance', title='Feature Importances', text='importance')
    fig_fi.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_fi.update_layout(yaxis=dict(range=[0, max(fi_df['importance']) * 1.2]), height=420)
    st.plotly_chart(fig_fi, use_container_width=True)

with right_col:
    st.subheader("Model Summary & Confusion Matrices")
    # Confusion matrix heatmaps side-by-side
    cm_lr = eval_lr['cm']
    cm_rf = eval_rf['cm']

    # Create heatmap for logistic
    fig_cm_lr = go.Figure(data=go.Heatmap(
        z=cm_lr,
        x=['Pred 0','Pred 1'],
        y=['True 0','True 1'],
        colorscale='Blues',
        showscale=False,
        hoverongaps=False))
    fig_cm_lr.update_layout(title=f'Logistic CM (acc={eval_lr["accuracy"]:.2f})', height=240, margin=dict(t=40,b=10))
    st.plotly_chart(fig_cm_lr, use_container_width=True)

    fig_cm_rf = go.Figure(data=go.Heatmap(
        z=cm_rf,
        x=['Pred 0','Pred 1'],
        y=['True 0','True 1'],
        colorscale='Greens',
        showscale=False,
        hoverongaps=False))
    fig_cm_rf.update_layout(title=f'RF CM (acc={eval_rf["accuracy"]:.2f})', height=240, margin=dict(t=40,b=10))
    st.plotly_chart(fig_cm_rf, use_container_width=True)

    # Compact classification report tables
    st.markdown("**Logistic Regression (classification metrics)**")
    rep_lr = classification_report(y_test, eval_lr['y_pred'], output_dict=True)
    rep_lr_df = pd.DataFrame(rep_lr).T.round(3)
    st.dataframe(rep_lr_df)

    st.markdown("**Random Forest (classification metrics)**")
    rep_rf = classification_report(y_test, eval_rf['y_pred'], output_dict=True)
    rep_rf_df = pd.DataFrame(rep_rf).T.round(3)
    st.dataframe(rep_rf_df)

    # Prediction card
    st.markdown("---")
    st.markdown("### Predict for a new patient")
    with st.form("predict_form"):
        p_age = st.number_input("Age", min_value=1, max_value=120, value=40)
        p_smokes = st.selectbox("Smokes", ["No","Yes"])
        p_areaq = st.selectbox("AreaQ", ["No","Yes"])
        p_alkhol = st.selectbox("Alkhol", ["No","Yes"])
        submitted = st.form_submit_button("Get prediction")

    if submitted:
        def yn(x): return 1 if x == "Yes" else 0
        X_new = pd.DataFrame({
            'Age':[p_age],
            'Smokes':[yn(p_smokes)],
            'AreaQ':[yn(p_areaq)],
            'Alkhol':[yn(p_alkhol)]
        })

        # logistic (scaled)
        X_new_scaled = scaler.transform(X_new)
        prob_log = float(log_model.predict_proba(X_new_scaled)[0,1])
        pred_log = int(log_model.predict(X_new_scaled)[0])

        # rf (raw)
        prob_rf_new = float(rf_model.predict_proba(X_new)[0,1])
        pred_rf_new = int(rf_model.predict(X_new)[0])

        # Show card
        st.markdown("<div class='metric'>", unsafe_allow_html=True)
        st.markdown(f"**Logistic probability:** <span style='color:#0b5cff; font-weight:700'>{prob_log:.3f}</span>", unsafe_allow_html=True)
        st.markdown(f"**RandomForest probability:** <span style='color:#0b5cff; font-weight:700'>{prob_rf_new:.3f}</span>", unsafe_allow_html=True)

        # colored badge for RF (preferred)
        badge_color = "#16a34a" if prob_rf_new < 0.5 else "#dc2626"
        badge_text = "Low risk" if prob_rf_new < 0.5 else "High risk"
        st.markdown(f"<div style='display:inline-block;padding:10px;border-radius:8px;background:{badge_color};color:#fff;font-weight:700'>{badge_text}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Bottom: Data preview & notes
# -------------------------
st.markdown("---")
st.subheader("Data preview (filtered)")
st.dataframe(df_f.head(25))

st.markdown("**Notes**")
st.markdown("""
- Models: Logistic Regression (scaled features) and RandomForest (raw features).  
- Use the sidebar controls to filter the dataset and see the KPIs & charts update.  
- Retrain models by deleting existing model files and reloading the app, or use the 'Retrain Models' button below.
""")

# Retrain button
if st.button("Retrain Models (rebuild & save)"):
    with st.spinner("Retraining..."):
        log_model, rf_model, scaler = train_and_save(X_train, y_train)
        st.success("Retrained and saved models. Refreshing page...")
        st.experimental_rerun()
