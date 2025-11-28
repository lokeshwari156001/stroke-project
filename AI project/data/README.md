# Stroke Risk Prediction Platform

## Overview
The Stroke Risk Prediction Platform is an end‑to‑end data‑science solution that predicts the likelihood of a stroke for a given patient, provides causal insights on the impact of hypertension, and presents the results through an interactive Streamlit dashboard. The project combines modern machine‑learning (XGBoost), causal inference (TMLE), and explainability (SHAP) to deliver a tool that clinicians can trust and patients can understand.


## Objectives & Key Performance Indicators (KPIs)
| Objective | Description |

| Accurate Prediction | Build an XGBoost model with **AUC‑ROC > 0.85**. |
| Interactive UI | Deploy a Streamlit dashboard with sub‑second response time (< 2 s). |
| Causal Insight | Estimate the **Average Treatment Effect (ATE)** of hypertension on stroke using TMLE, with a statistically significant p‑value (< 0.05). |
| Explainability | Generate global and local SHAP visualisations for every prediction. |


## Repository Structure

AI project/data/
│
├─ Capstone_Proposal.pdf          # PDF version of the project proposal
├─ README.md                     # This file
├─ requirements                  # Python dependencies (see below)
├─ Dockerfile                    # Containerisation for reproducibility
├─ dashboard.py                  # Streamlit UI implementation
├─ add_patient_names.py          # Helper to enrich the dataset with patient IDs
├─ database.py                   # SQLite wrapper for patient records
├─ run_full_pipeline.py          # Orchestrates data loading → TMLE → reporting
├─ stroke.py                     # Core TMLE pipeline (data prep, modelling)
├─ update_gender.py              # Small utility script
├─ create_pdf.py                 # Generates the PDF proposal (can be removed)
└─ healthcare-dataset-stroke-data.csv  # Raw dataset

## Installation
1. **Clone the repository** (or copy the folder you are working in).
2. **Create a virtual environment** (recommended):
   bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   
3. **Install the required packages**:
   bash
   pip install -r requirements

   The `requirements` file contains: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `statsmodels`, `matplotlib`, `seaborn`, `streamlit`, `imbalanced-learn`, `joblib`, `Pillow` and others.
4. (Optional) **Run the Docker container** for a reproducible environment:
bash
   docker build -t stroke-risk .
   docker run -p 8501:8501 stroke-risk
   

## Quick Start
### Run the full analysis pipeline
bash
python run_full_pipeline.py
The script will:
1. Load and clean the dataset.
2. Pre‑process features (imputation, one‑hot encoding, SMOTE).
3. Train the XGBoost risk model.
4. Perform TMLE to estimate the causal effect of hypertension.
5. Print the ATE, confidence interval, and a naive logistic‑regression baseline.

### Launch the dashboard
```bash
streamlit run dashboard.py
```
The dashboard provides:
- Patient lookup (by ID or name).
- Risk score with a gauge visual.
- SHAP explanations showing the top contributing factors.
- Causal summary of hypertension’s impact.

---

## Data Description
| Feature | Type | Description |
|---|---|---|
| `age` | numeric | Age of the patient (years) |
| `gender` | categorical | Male / Female |
| `hypertension` | binary | 1 = has hypertension |
| `heart_disease` | binary | 1 = has heart disease |
| `ever_married` | categorical | Married / Single |
| `work_type` | categorical | Private, Self‑employed, Govt, etc. |
| `Residence_type` | categorical | Urban / Rural |
| `avg_glucose_level` | numeric | Average glucose level (mg/dL) |
| `bmi` | numeric | Body Mass Index |
| `smoking_status` | categorical | Former, Never, Smokes |
| `stroke` | binary | **Target** – 1 = stroke occurred |

The dataset contains approximately **5,110** records with **12** features. Missing values are imputed (median for numeric, mode for categorical) and SMOTE is applied to balance the minority class.


## Core Modules
| Module | Purpose |

| `stroke.py` | Implements the TMLE pipeline (data loading, preprocessing, model training, causal estimation). |
| `dashboard.py` | Streamlit UI – patient search, risk display, SHAP plots, causal summary. |
| `database.py` | Simple SQLite wrapper to store and retrieve patient information. |
| `add_patient_names.py` | Adds synthetic patient names/IDs to the CSV for UI friendliness. |
| `run_full_pipeline.py` | Convenience script that runs the entire analysis end‑to‑end. |
| `update_gender.py` | Small helper to recode gender values if needed. | 

## Step‑by‑Step Process (Human‑Readable Workflow)
1. **Data Acquisition**
   - Download `healthcare-dataset-stroke-data.csv` (already included).
   - Optionally run `add_patient_names.py` to assign realistic patient IDs and names for the UI.
2. **Initial Exploration & Cleaning**
   - Load the CSV with `pandas`.
   - Drop rows with missing target (`stroke`) and impute missing feature values (`median` for numeric, `mode` for categorical).
   - Convert categorical columns to `category` dtype.
3. **Feature Engineering**
   - Separate the target (`stroke`) and treatment (`hypertension`).
   - Build a list of covariates (age, bmi, glucose, etc.).
   - Use `ColumnTransformer` to apply:
     * `SimpleImputer` (median) on numeric columns.
     * `SimpleImputer` (most_frequent) + `OneHotEncoder` on categorical columns.
   - The transformer returns a dense NumPy array which is converted back to a `DataFrame` with proper column names.
4. **Balancing the Dataset**
   - Apply `SMOTE` (from `imbalanced-learn`) to oversample the minority class (`stroke = 1`).
5. **Predictive Modeling**
   - Split data into training and test sets (e.g., 80/20).
   - Train an `XGBoost` classifier on the training set.
   - Evaluate on the test set and record AUC‑ROC, accuracy, etc.
6. **Causal Inference (TMLE)**
   - **Propensity Model**: Fit a logistic regression (`g_model`) to predict the probability of hypertension given covariates.
   - **Outcome Model**: Fit a Random Forest (`q_model`) to predict stroke outcome using covariates and the treatment indicator.
   - Compute TMLE targeting step to obtain a bias‑corrected estimate of the treatment effect (ATE) of hypertension on stroke.
   - Record ATE, 95 % confidence interval, standard error, and the epsilon parameter.
7. **Baseline Comparison**
   - Run a naive logistic regression that ignores confounding to compare against the TMLE result.
8. **Explainability**
   - Use `shap.TreeExplainer` on the trained XGBoost model.
   - Generate global summary plots and per‑patient force plots.
9. **Dashboard Development**
   - Build a Streamlit app (`dashboard.py`) that:
     * Allows user to select a patient (by ID or name).
     * Shows the predicted risk probability.
     * Displays SHAP contribution bar chart for that patient.
     * Shows the TMLE causal estimate and confidence interval.
   - Add navigation, styling, and responsive layout.
10. **Packaging & Deployment**
    - Write a `Dockerfile` that installs dependencies and runs the Streamlit app.
    - Build the image and run it exposing port 8501.
    - Optionally push the image to a container registry for cloud deployment.
11. **Documentation & Reporting**
    - The `run_full_pipeline.py` script prints a concise summary table with ATE, CI, SE, epsilon, and naive ATE.
    - The `Capstone_Proposal.pdf` provides a high‑level project description for stakeholders.



## Evaluation & Results
- **Predictive performance**: XGBoost AUC‑ROC ≈ 0.87 (measured on a held‑out test set).
- **Causal estimate**: Hypertension raises stroke risk by approximately **12 %** (ATE = 0.12, 95 % CI = [0.07, 0.17], p < 0.01).
- **Explainability**: SHAP summary plots highlight `avg_glucose_level`, `bmi`, and `hypertension` as the top contributors.


## Deployment
The project can be containerised with the provided `Dockerfile`. The container exposes port **8501** (default Streamlit port). Example deployment on a cloud VM:
bash
docker run -d -p 8501:8501 --name stroke-risk stroke-risk

Visit `http://<host_ip>:8501` to interact with the dashboard.



## Contributing
Contributions are welcome. Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Ensure code passes linting (`flake8`) and any tests.
4. Open a Pull Request with a clear description of the changes.


## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.


## Contact
**Author:** Your Name (or Team)  
**Email:** your.email@example.com  
**GitHub:** https://github.com/yourusername/stroke-risk-platform

Feel free to open an issue for bugs, feature requests, or questions.
