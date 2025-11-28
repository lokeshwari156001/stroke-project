
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap

def create_proposal_pdf(filename):
    # Data for the proposal
    data = {
        "Project Title": "Stroke Risk Prediction Platform",
        "Problem Statement": "Stroke is a leading cause of death and disability globally. Traditional risk assessments often fail to capture complex non-linear interactions between health factors. There is a critical need for an accessible, AI-driven tool that predicts stroke risk with high accuracy and provides interpretable insights.",
        "Objectives / KPIs": "Objectives:\n1. Develop a high-accuracy ML model (XGBoost).\n2. Build an interactive Streamlit dashboard.\n3. Implement Causal Inference (TMLE).\n4. Provide model explainability (SHAP).\n\nKPIs:\n- Model AUC-ROC > 0.85\n- Dashboard Response Time < 2s\n- Causal Estimate p-value < 0.05",
        "Dataset": "Source: Healthcare Dataset Stroke Data (Kaggle)\nSize: ~5,110 records, 12 features\nType: Structured Tabular Data (Demographics, Vitals, Medical History).",
        "Methodology": "1. Preprocessing: Imputation, One-Hot Encoding, SMOTE.\n2. Modeling: XGBoost Classifier for risk prediction.\n3. Causal Inference: TMLE using Logistic Regression & Random Forest.",
        "Evaluation Metrics": "Predictive: AUC-ROC, F1-Score, Accuracy.\nCausal: ATE, Standard Error, 95% CI.\nExplainability: SHAP values.",
        "Tools / Libraries": "Python, Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels, Matplotlib, Seaborn, Streamlit.",
        "Expected Outcome": "A deployed web app where Doctors get real-time risk predictions and Patients view health profiles. Includes visual explanations of risk factors and statistical reports on hypertension impact.",
        "Timeline / Milestones": "Week 1: Data Prep & EDA\nWeek 2: Model Training\nWeek 3: Causal Inference\nWeek 4: Dashboard Dev\nWeek 5: Explainability\nWeek 6: Final Polish & Docker"
    }

    # Create a PDF object
    with PdfPages(filename) as pdf:
        # A4 size in inches
        fig = plt.figure(figsize=(8.27, 11.69))
        
        # Title
        plt.text(0.5, 0.95, "Capstone Proposal", fontsize=20, fontweight='bold', ha='center')
        
        y_pos = 0.90
        left_margin = 0.1
        line_height = 0.025
        section_gap = 0.02
        
        # Iterate through data
        for section, content in data.items():
            # Section Title
            plt.text(left_margin, y_pos, section, fontsize=12, fontweight='bold')
            y_pos -= line_height
            
            # Content (Wrapped)
            # Wrap text to approx 90 characters
            wrapped_text = textwrap.fill(content, width=95)
            
            plt.text(left_margin, y_pos, wrapped_text, fontsize=10, va='top', fontfamily='sans-serif')
            
            # Calculate how much space the text took
            num_lines = wrapped_text.count('\n') + 1
            y_pos -= (num_lines * line_height) + section_gap
            
            # Check if we need a new page (simple check)
            if y_pos < 0.1:
                plt.axis('off')
                pdf.savefig(fig)
                plt.close()
                fig = plt.figure(figsize=(8.27, 11.69))
                y_pos = 0.95

        plt.axis('off')
        pdf.savefig(fig)
        plt.close()

if __name__ == "__main__":
    output_file = r"C:\Users\User\Desktop\AI project\data\Capstone_Proposal.pdf"
    create_proposal_pdf(output_file)
    print(f"PDF generated: {output_file}")
