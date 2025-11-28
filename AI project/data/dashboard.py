import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import time
from pathlib import Path
from PIL import Image
import database as db
import run_full_pipeline as pipeline
import plotly.graph_objects as go
# --- 1. App Setup & Structure ---
st.set_page_config(
    page_title="Stroke Risk Prediction Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
OUTDIR = Path(r"c:\Users\User\Desktop\AI project\data\stroke_outputs")
FIGDIR = OUTDIR / "figures"
DEFAULT_DATA_PATH = r"c:\Users\User\Desktop\AI project\data\healthcare-dataset-stroke-data.csv"

# Ensure directories exist
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

# --- Database Init ---
db.init_db()

# --- Custom CSS ---
def local_css():
    st.markdown("""
    <style>
        /* Global Dark Theme */
        .stApp {
            background-color: #020b1c;
            color: white;
        }
        
        /* Text Colors */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stText {
            color: white !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #0b1426;
            border-right: 1px solid #1e2a45;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #1f77b4;
            color: white !important;
            border: none;
            border-radius: 4px;
        }
        .stButton>button:hover {
            background-color: #1c669c;
            color: white !important;
        }
        .stButton>button:focus {
            background-color: #1f77b4;
            color: white !important;
            box-shadow: none;
        }
        .stButton>button:active {
            background-color: #1a5a8a;
            color: white !important;
        }
        
        /* Form Submit Button (Analyze Risk) - Light Green */
        div[data-testid="stForm"] .stButton > button {
            background-color: #2ecc71 !important;
            color: white !important;
            border: none;
        }
        div[data-testid="stForm"] .stButton > button:hover {
            background-color: #27ae60 !important;
            color: white !important;
        }
        div[data-testid="stForm"] .stButton > button:active {
            background-color: #219150 !important;
            color: white !important;
        }
        
        /* KPI Card Style */
        .kpi-card {
            background-color: #111c30;
            border-radius: 8px;
            padding: 15px;
            border: 1px solid #1e2a45;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            height: 150px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .kpi-header {
            background-color: #1c2b4a;
            color: #64b5f6;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            width: fit-content;
            margin-bottom: 8px;
            text-transform: uppercase;
        }
        
        .kpi-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            margin: 0;
        }
        
        .kpi-sub {
            font-size: 0.75rem;
            color: #8b9bb4;
            margin: 0;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 10px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #111c30;
            color: white;
            border-radius: 4px;
            border: 1px solid #1e2a45;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Helper Functions ---
def load_dataset(path):
    return pd.read_csv(path)

def login_page():
    # Modern Split Layout
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        # Hero Image Section
        if os.path.exists("login_hero.png"):
            st.image("login_hero.png", use_container_width=True)
        else:
            st.markdown("### 🧠 AI-Powered Healthcare")
            
        st.markdown("""
        <div style='padding: 20px; background-color: #111c30; border-radius: 10px; margin-top: 10px;'>
            <h3 style='color: #3498db; margin-bottom: 10px;'>Advanced Stroke Prediction</h3>
            <p style='color: #b0bec5; font-size: 1.1em;'>
                Leveraging state-of-the-art Artificial Intelligence to analyze health metrics and predict stroke risks with high accuracy.
                Our platform empowers patients and doctors with actionable insights.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Login Form Section
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True) # Spacer
        
        with st.container():
            st.markdown("""
            <div style='background-color: #0b1426; padding: 40px; border-radius: 15px; border: 1px solid #1e2a45; box-shadow: 0 10px 25px rgba(0,0,0,0.5);'>
                <h2 style='text-align: center; margin-bottom: 30px; color: white;'>Welcome Back</h2>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type='password', placeholder="Enter your password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🚀 Secure Login", use_container_width=True):
                role, gender = db.login_user(username, password)
                if role:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['role'] = role
                    st.session_state['gender'] = gender
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            
            st.markdown("""
                <div style='text-align: center; margin-top: 20px; color: #666;'>
                    Don't have an account?
                </div>
            """, unsafe_allow_html=True)
            
            if st.button("Create New Account", use_container_width=True):
                st.session_state['auth_mode'] = 'register'
                st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)

def home_dashboard_page():
    st.title("🏠 Dashboard")
    
    # Check for data and load if missing
    if 'data' not in st.session_state:
        if os.path.exists(DEFAULT_DATA_PATH):
            st.session_state['data'] = load_dataset(DEFAULT_DATA_PATH)
            
    has_data = 'data' in st.session_state
    df = st.session_state['data'] if has_data else None
    
    # --- Patient Specific View ---
    if st.session_state.get('role') == 'Patient' and has_data and 'patient_name' in df.columns:
        username = st.session_state.get('username')
        patient_data = df[df['patient_name'] == username]
        
        if not patient_data.empty:
            p_row = patient_data.iloc[0]
            
            st.markdown(f"""
            <div style='background-color: #111c30; padding: 20px; border-radius: 10px; border-left: 5px solid #3498db; margin-bottom: 20px;'>
                <h3 style='margin:0; color:white;'>👋 Welcome back, {username}</h3>
                <p style='color:#8b9bb4; margin:0;'>Here is your personal health summary based on your latest records.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Patient Metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-header">YOUR AGE</div>
                    <div class="kpi-value">{p_row['age']:.0f}</div>
                    <div class="kpi-sub">Years</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                bmi_val = p_row['bmi'] if pd.notnull(p_row['bmi']) else "N/A"
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-header">YOUR BMI</div>
                    <div class="kpi-value">{bmi_val}</div>
                    <div class="kpi-sub">Body Mass Index</div>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-header">AVG GLUCOSE</div>
                    <div class="kpi-value">{p_row['avg_glucose_level']:.1f}</div>
                    <div class="kpi-sub">mg/dL</div>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                risk_status = "High" if p_row['stroke'] == 1 else "Low"
                risk_color = "#e74c3c" if p_row['stroke'] == 1 else "#2ecc71"
                st.markdown(f"""
                <div class="kpi-card" style='border-color: {risk_color};'>
                    <div class="kpi-header">STROKE HISTORY</div>
                    <div class="kpi-value" style='color: {risk_color};'>{risk_status}</div>
                    <div class="kpi-sub">Based on historical data</div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📄 View Full Health Profile", expanded=False):
                st.markdown("#### Additional Health & Lifestyle Information")
                
                # Row 1: Medical History
                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1:
                    h_icon = "❤️" if p_row['hypertension']==1 else "💚"
                    st.markdown(f"**Hypertension:** {h_icon} {'Yes' if p_row['hypertension']==1 else 'No'}")
                with r1c2:
                    hd_icon = "❤️" if p_row['heart_disease']==1 else "💚"
                    st.markdown(f"**Heart Disease:** {hd_icon} {'Yes' if p_row['heart_disease']==1 else 'No'}")
                with r1c3:
                    smk_map = {'formerly smoked': '🚬 (Former)', 'never smoked': '🚭 (Never)', 'smokes': '🚬 (Current)', 'Unknown': '❓'}
                    smk_val = p_row['smoking_status']
                    st.markdown(f"**Smoking Status:** {smk_map.get(smk_val, smk_val)}")

                st.markdown("---")
                
                # Row 2: Lifestyle
                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1:
                    st.markdown(f"**Work Type:** 💼 {p_row['work_type'].replace('_', ' ').title()}")
                with r2c2:
                    st.markdown(f"**Residence:** 🏠 {p_row['Residence_type']}")
                with r2c3:
                    st.markdown(f"**Marital Status:** 💍 {'Married' if p_row['ever_married']=='Yes' else 'Single'}")
            
            st.markdown("---")
            st.markdown("### 🌍 Global Dataset Statistics")

    st.markdown("### 🔔 Insurance Descriptive Analytics")
    st.markdown("**Page: Dashboard**")
    
    if has_data:
        # Single Row for all 5 Metrics
        k1, k2, k3, k4, k5 = st.columns(5)
        
        with k1:
            val = f"{df.shape[0]:,}"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">TOTAL RECORDS</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">Patients in dataset</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k2:
            stroke_count = df['stroke'].sum()
            stroke_prev = (stroke_count / len(df)) * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">STROKE CASES</div>
                <div class="kpi-value">{stroke_count}</div>
                <div class="kpi-sub">≈ {stroke_prev:.2f}% prevalence</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            missing_bmi = df['bmi'].isnull().sum() if 'bmi' in df.columns else 0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">MISSING VALUES</div>
                <div class="kpi-value">{missing_bmi}</div>
                <div class="kpi-sub">Missing BMI values</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k4:
            if 'avg_glucose_level' in df.columns:
                median_gl = df['avg_glucose_level'].median()
                max_gl = df['avg_glucose_level'].max()
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-header">AVG GLUCOSE</div>
                    <div class="kpi-value">{median_gl:.1f}</div>
                    <div class="kpi-sub">Median (Max: {max_gl:.0f})</div>
                </div>
                """, unsafe_allow_html=True)
            
        with k5:
            hyp_rate = df['hypertension'].mean() * 100 if 'hypertension' in df.columns else 0
            heart_rate = df['heart_disease'].mean() * 100 if 'heart_disease' in df.columns else 0
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">CONDITIONS</div>
                <div style="display:flex; justify-content:space-between; align-items:end;">
                    <div>
                        <div style="font-size:1.2rem; font-weight:bold; color:white;">{hyp_rate:.1f}%</div>
                        <div style="font-size:0.6rem; color:#8b9bb4;">Hyper</div>
                    </div>
                    <div style="border-right:1px solid #1e2a45; height:30px; margin:0 5px;"></div>
                    <div>
                        <div style="font-size:1.2rem; font-weight:bold; color:white;">{heart_rate:.1f}%</div>
                        <div style="font-size:0.6rem; color:#8b9bb4;">Heart</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("⚠️ No dataset loaded. Please go to **Dataset Overview** to start.")
        
    st.markdown("---")
    st.markdown("### Welcome to the Stroke Risk Prediction Platform")
    st.markdown("""
    This advanced analytics platform allows medical professionals to:
    - **Explore** patient demographics and health factors.
    - **Train** machine learning models to predict stroke risk.
    - **Evaluate** model performance with detailed metrics.
    - **Predict** stroke risk for individual patients in real-time.
    """)

    if has_data:
        st.markdown("### 📉 Patient Health Conditions Overview")
        
        # Prepare data for plotting
        conditions = ['hypertension', 'heart_disease', 'stroke']
        # Filter only existing columns
        existing_conditions = [c for c in conditions if c in df.columns]
        
        if existing_conditions:
            counts = [df[c].sum() for c in existing_conditions]
            labels = [c.replace('_', ' ').title() for c in existing_conditions]
            
            # Use columns to control size (Medium size)
            c1, c2, c3 = st.columns([1, 2, 1])
            
            with c2:
                fig, ax = plt.subplots(figsize=(5, 5))
                # Set dark background style for plot
                plt.style.use('dark_background')
                
                # Pie Chart
                wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, 
                                                  colors=sns.color_palette("viridis", len(counts)),
                                                  textprops={'color':"white"})
                
                ax.set_title("Distribution of Health Conditions", color='white', fontsize=14)
                
                # Ensure transparency matches app theme
                fig.patch.set_facecolor('#020b1c')
                ax.set_facecolor('#020b1c')
                
                st.pyplot(fig, use_container_width=True)

def dataset_overview_page():
    st.title("📁 Dataset Overview")
    
    # Data Loading & User Input
    # Data Loading
    df = None
    if os.path.exists(DEFAULT_DATA_PATH):
        df = load_dataset(DEFAULT_DATA_PATH)
    else:
        st.error("Default dataset not found.")
            
    if df is not None:
        # Check if we need to update session state (e.g. if patient_name was added)
        if 'data' in st.session_state:
            if 'patient_name' in df.columns and 'patient_name' not in st.session_state['data'].columns:
                st.session_state['data'] = df
                st.rerun()
        
        st.session_state['data'] = df # Store in session
        
        # Reorder columns to show patient_name first if it exists
        if 'patient_name' in df.columns:
            cols = ['patient_name'] + [c for c in df.columns if c != 'patient_name']
            df = df[cols]
        
        # Data Quality Summary
        st.markdown("### 📊 Data Quality Summary")
        
        # Custom KPI Cards
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">TOTAL RECORDS</div>
                <div class="kpi-value">{df.shape[0]:,}</div>
                <div class="kpi-sub">Rows</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">FEATURES</div>
                <div class="kpi-value">{df.shape[1]}</div>
                <div class="kpi-sub">Columns</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi3:
            missing_count = df.isnull().sum().sum()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">MISSING VALUES</div>
                <div class="kpi-value">{missing_count}</div>
                <div class="kpi-sub">Total cells</div>
            </div>
            """, unsafe_allow_html=True)
            
        with kpi4:
            dup_count = df.duplicated().sum()
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-header">DUPLICATES</div>
                <div class="kpi-value">{dup_count}</div>
                <div class="kpi-sub">Duplicate rows</div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("🔍 View Raw Data & Statistics", expanded=False):
            st.dataframe(df.head())
            st.write("Descriptive Statistics:")
            st.write(df.describe(include='all'))

def eda_page():
    st.title("📊 Exploratory Analysis & Visualization")
    
    if st.session_state.get('data') is None:
        st.warning("Please load a dataset in the 'Dataset Overview' page first.")
        return
        
    df = st.session_state['data']
    
    tab1, tab2, tab3 = st.tabs(["Distributions", "Categorical Analysis", "Correlations"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='age', hue='stroke', kde=True, ax=ax)
            st.pyplot(fig)
            st.info("💡 Older age groups show higher stroke prevalence.")
        with col2:
            st.markdown("#### Glucose Level Distribution")
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='avg_glucose_level', hue='stroke', kde=True, ax=ax)
            st.pyplot(fig)
            st.info("💡 Higher glucose levels correlate with stroke risk.")
            
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Stroke by Gender")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='gender', hue='stroke', ax=ax)
            st.pyplot(fig)
        with col2:
            st.markdown("#### Stroke by Smoking Status")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='smoking_status', hue='stroke', ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
    with tab3:
        st.markdown("#### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

def preprocessing_page():
    st.title("🧹 Preprocessing & Feature Engineering")
    
    if st.session_state.get('data') is None:
        st.warning("Please load a dataset in the Home tab first.")
        return
        
    df = st.session_state['data']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        st.markdown("**Feature Engineering**")
        use_age_bin = st.checkbox("Create Age Bins", value=True)
        use_bmi_cat = st.checkbox("Create BMI Categories", value=True)
        use_risk_score = st.checkbox("Create Risk Score", value=True)
        
        st.markdown("**Balancing Strategy**")
        balance_method = st.selectbox("Class Imbalance Handling", ["smote", "random", "none"])
        
        if st.button("Run Preprocessing"):
            with st.spinner("Processing data..."):
                # Use pipeline function
                processed_df = pipeline.clean_and_fe(df, use_age_bin, use_bmi_cat, use_risk_score)
                st.session_state['processed_df'] = processed_df
                st.session_state['fe_config'] = {
                    'use_age_bin': use_age_bin,
                    'use_bmi_cat': use_bmi_cat,
                    'use_risk_score': use_risk_score,
                    'balance_method': balance_method
                }
                st.success("Preprocessing Complete!")
                
    with col2:
        if 'processed_df' in st.session_state:
            st.subheader("✅ Processed Data Preview")
            st.dataframe(st.session_state['processed_df'].head())
            
            st.markdown("#### New Features Distribution")
            if use_risk_score:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=st.session_state['processed_df'], x='risk_score', hue='stroke', ax=ax)
                st.pyplot(fig)

def training_page():
    st.title("⚙️ Model Training")
    
    if 'processed_df' not in st.session_state:
        st.warning("Please run preprocessing first.")
        return
        
    df = st.session_state['processed_df']
    config = st.session_state.get('fe_config', {'balance_method': 'smote'})
    
    # Premium Header Card
    st.markdown("""
    <div style='background-color: #111c30; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); margin-bottom: 25px; border-left: 5px solid #3498db;'>
        <h3 style='margin-top: 0; color: white;'>🚀 Ready to Train</h3>
        <p style='color: #8b9bb4; margin-bottom: 0;'>Initialize the XGBoost training pipeline with your current configuration. This process includes hyperparameter tuning and cross-validation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Info Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">BALANCING STRATEGY</div>
            <div class="kpi-value">{config['balance_method'].upper()}</div>
            <div class="kpi-sub">Class Imbalance Handling</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-header">DATASET DIMENSIONS</div>
            <div class="kpi-value">{df.shape}</div>
            <div class="kpi-sub">Rows × Columns</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("###") # Spacer
    
    # Action Button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        start_btn = st.button("🚀 Start Training Pipeline", type="primary", use_container_width=True)
        
    if start_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.markdown("**Initializing training environment...**")
            progress_bar.progress(10)
            time.sleep(0.5) # Visual feedback
            
            # Run training
            with st.spinner("Training & Tuning XGBoost Model..."):
                results = pipeline.train_and_tune(df, balancing_method=config['balance_method'])
                
            progress_bar.progress(100)
            status_text.markdown("✅ **Training Complete!**")
            
            st.balloons()
            st.success("Model successfully trained and saved to disk.")
            
            with st.expander("View Training Results JSON", expanded=True):
                st.json(results)
            
        except Exception as e:
            st.error(f"Training failed: {e}")

def model_evaluation_page():
    st.title("🧪 Model Evaluation")
    
    if not os.path.exists(OUTDIR / "metrics_summary.csv"):
        st.warning("No model results found. Please train the model first.")
        return
        
    c1, c2 = st.columns(2)
    if (FIGDIR / "roc_curve.png").exists():
        c1.image(str(FIGDIR / "roc_curve.png"), caption="ROC Curve", use_container_width=True)
    if (FIGDIR / "pr_curve.png").exists():
        c2.image(str(FIGDIR / "pr_curve.png"), caption="Precision-Recall Curve", use_container_width=True)
        
    st.markdown("---")
    st.subheader("📑 Classification Report")
    if (OUTDIR / "classification_report.csv").exists():
        report = pd.read_csv(OUTDIR / "classification_report.csv")
        st.dataframe(report)
    if (OUTDIR / "confusion_matrix.csv").exists():
        cm = pd.read_csv(OUTDIR / "confusion_matrix.csv")
        st.write("Confusion Matrix:")
        st.dataframe(cm)

def explainability_page():
    st.title("🔍 Explainability (XAI)")
    
    if not (FIGDIR / "shap_summary.png").exists():
        st.warning("SHAP analysis not found. Please train the model first.")
        return
        
    st.write("**SHAP (SHapley Additive exPlanations)** helps us understand which features contributed most to the model's predictions.")
    c1, c2 = st.columns(2)
    if (FIGDIR / "shap_summary.png").exists():
        c1.image(str(FIGDIR / "shap_summary.png"), caption="SHAP Summary Plot", use_container_width=True)
    if (FIGDIR / "shap_bar.png").exists():
        c2.image(str(FIGDIR / "shap_bar.png"), caption="Feature Importance Bar Plot", use_container_width=True)

def export_reports_page():
    st.title("📤 Export & Reports")
    st.info("This feature is coming soon. You will be able to export PDF reports of model performance and patient predictions.")

def settings_page():
    st.title("⚙️ Settings")
    st.write("Configure application settings here.")
    st.checkbox("Enable Dark Mode Support", value=True)
    st.checkbox("Show Advanced Metrics", value=True)

def prediction_page():
    st.title("🔮 Real-Time Stroke Risk Prediction")
    
    model_path = OUTDIR / "best_model.joblib"
    if not model_path.exists():
        st.error("Model not found. Please ask an administrator to train the model.")
        return
        
    model = joblib.load(model_path)
    
    with st.form("prediction_form"):
        st.subheader("Patient Vitals")
        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female", "Other"])
        age = c2.number_input("Age", 0, 120, 50)
        bmi = c3.number_input("BMI", 10.0, 60.0, 25.0)
        
        c4, c5, c6 = st.columns(3)
        hypertension = c4.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        heart_disease = c5.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        avg_glucose = c6.number_input("Avg Glucose Level", 50.0, 300.0, 100.0)
        
        st.subheader("Lifestyle & History")
        c7, c8, c9 = st.columns(3)
        married = c7.selectbox("Ever Married", ["Yes", "No"])
        work = c8.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence = c9.selectbox("Residence Type", ["Urban", "Rural"])
        smoking = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
        
        submitted = st.form_submit_button("🔍 Analyze Risk", type="primary", use_container_width=True)
        
    if submitted:
        # Create input dataframe
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [married],
            'work_type': [work],
            'Residence_type': [residence],
            'avg_glucose_level': [avg_glucose],
            'bmi': [bmi],
            'smoking_status': [smoking]
        })
        
        try:
            # Apply FE
            input_fe = pipeline.clean_and_fe(input_data) 
            
            # Predict
            prob = model.predict_proba(input_fe)[0][1]
            pred = model.predict(input_fe)[0]
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col_res, col_adv = st.columns([1, 2])
            
            with col_res:
                # Gauge Chart (Speedometer)
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Stroke Risk Probability", 'font': {'size': 18, 'color': "white"}},
                    number = {'suffix': "%", 'font': {'size': 24, 'color': "white"}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#e74c3c" if prob > 0.5 else "#2ecc71"},
                        'bgcolor': "#0b1426",
                        'borderwidth': 2,
                        'bordercolor': "#1e2a45",
                        'steps': [
                            {'range': [0, 50], 'color': '#1e2a45'},
                            {'range': [50, 100], 'color': '#111c30'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': prob * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor="#111c30", 
                    font={'color': "white", 'family': "Arial"},
                    height=250,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with col_adv:
                if prob > 0.5:
                    st.error("⚠️ **High Risk Detected**")
                    st.write("The model indicates a significant probability of stroke. Immediate medical consultation is recommended.")
                else:
                    st.success("✅ **Low Risk Detected**")
                    st.write("The model indicates a low probability of stroke. Maintain a healthy lifestyle.")
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        
    if not st.session_state['logged_in']:
        # Handle Login/Register
        if 'auth_mode' not in st.session_state:
            st.session_state['auth_mode'] = 'login'
            
        if st.session_state['auth_mode'] == 'login':
            login_page()
        else:
            # Register Page (Simplified inline)
            st.markdown("<h2 style='text-align: center;'>📝 Register</h2>", unsafe_allow_html=True)
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    new_user = st.text_input("New Username")
                    new_pass = st.text_input("New Password", type='password')
                    role = st.selectbox("Role", ["Patient", "Doctor"])
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    if st.button("Register"):
                        if db.add_user(new_user, new_pass, role, gender):
                            st.success("Account created! Go to login.")
                            st.session_state['auth_mode'] = 'login'
                            st.rerun()
                        else:
                            st.error("Username exists.")
                    if st.button("Back to Login"):
                        st.session_state['auth_mode'] = 'login'
                        st.rerun()
        return

    # Logged In View
    user_role = st.session_state.get('role', 'Patient')
    
    # Sidebar
    with st.sidebar:
        st.title("🧠 StrokeAI")
        st.write(f"Welcome, **{st.session_state['username']}**")
        
        from streamlit_option_menu import option_menu
        
        if user_role == 'Doctor':
            options = [
                "Dashboard", "Dataset Overview", "Data Cleaning", 
                "Exploratory Analysis", "Visualization", "Preprocessing", 
                "Model Training", "Model Evaluation", "Explainability (XAI)", 
                "Stroke Risk Prediction", "Export & Reports", "Settings"
            ]
            icons = [
                "house", "folder", "brush", 
                "bar-chart", "graph-up", "gear", 
                "robot", "clipboard-data", "search", 
                "heart-pulse", "file-earmark-arrow-down", "sliders"
            ]
        else:
            options = ["Dashboard", "Stroke Risk Prediction"]
            icons = ["house", "heart-pulse"]
            
        nav = option_menu(
            menu_title=None,
            options=options,
            icons=icons,
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#0b1426"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#1c2b4a", "color": "white"},
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )
        
        st.markdown("---")
        if st.button("Logout", type="primary"):
            st.session_state['logged_in'] = False
            st.rerun()
        
    # Routing
    if nav == "Dashboard":
        home_dashboard_page()
    elif nav == "Dataset Overview":
        dataset_overview_page()
    elif nav == "Data Cleaning":
        preprocessing_page()
    elif nav == "Exploratory Analysis":
        eda_page()
    elif nav == "Visualization":
        eda_page()
    elif nav == "Preprocessing":
        preprocessing_page()
    elif nav == "Model Training":
        training_page()
    elif nav == "Model Evaluation":
        model_evaluation_page()
    elif nav == "Explainability (XAI)":
        explainability_page()
    elif nav == "Stroke Risk Prediction":
        prediction_page()
    elif nav == "Export & Reports":
        export_reports_page()
    elif nav == "Settings":
        settings_page()
        
    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center; color: grey; font-size: 0.8em;'>Stroke Risk Prediction Platform v2.0 | © 2025</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
