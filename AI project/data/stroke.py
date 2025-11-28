# ======================================================================
# TMLE Project – Hypertension → Stroke
# Full working code with updated OneHotEncoder(sparse_output=False)
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score

# Set plot style
sns.set_theme(style="whitegrid")

def load_and_prep_data(filepath):
    """Loads dataset and selects relevant variables."""
    print("Loading data...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None

    covariates = [
        'age', 'avg_glucose_level', 'bmi', 'gender', 'ever_married',
        'work_type', 'Residence_type', 'smoking_status', 'heart_disease'
    ]
    
    # Basic cleaning
    df = df[['hypertension', 'stroke'] + covariates].copy()
    df = df.dropna(subset=['hypertension', 'stroke'])
    df['bmi'] = df['bmi'].fillna(df['bmi'].median())
    
    df['hypertension'] = df['hypertension'].astype(int)
    df['stroke'] = df['stroke'].astype(int)
    df['heart_disease'] = df['heart_disease'].astype(int)
    
    A = df['hypertension'].values
    Y = df['stroke'].values
    X = df[covariates].copy()
    
    return df, X, A, Y

def preprocess_data(X):
    """Preprocesses covariates using ColumnTransformer."""
    print("Preprocessing data...")
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline([
            ('imp', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
        ]), cat_cols)
    ])

    X_trans = preprocessor.fit_transform(X)
    
    # Get feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
    ohe_cols = list(ohe.get_feature_names_out(cat_cols))
    proc_cols = num_cols + ohe_cols
    
    X_proc = pd.DataFrame(X_trans, columns=proc_cols)
    return X_proc

def calculate_smd(X_proc, A):
    """Calculates Standardized Mean Differences."""
    print("Calculating SMD...")
    def std_mean_diff(x1, x0):
        m1, m0 = np.nanmean(x1), np.nanmean(x0)
        s1, s0 = np.nanvar(x1, ddof=1), np.nanvar(x0, ddof=1)
        denom = np.sqrt((s1 + s0) / 2)
        return 0 if denom == 0 else (m1 - m0) / denom

    smd_list = []
    for col in X_proc.columns:
        smd_val = std_mean_diff(X_proc.loc[A==1, col], X_proc.loc[A==0, col])
        smd_list.append((col, smd_val))

    smd_df = pd.DataFrame(smd_list, columns=['covariate', 'smd']) \
                .sort_values('smd', ascending=False)
    return smd_df

def plot_covariate_balance(smd_df):
    """Plots the Standardized Mean Differences."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='smd', y='covariate', data=smd_df.head(15), palette='viridis')
    plt.title('Top 15 Covariate Imbalances (SMD)')
    plt.xlabel('Standardized Mean Difference')
    plt.tight_layout()
    plt.show()

def train_propensity_model(X_proc, A):
    """Trains the propensity score model (g-model)."""
    print("Training propensity model...")
    g_model = LogisticRegression(max_iter=2000)
    g_model.fit(X_proc, A)
    g_pred = g_model.predict_proba(X_proc)[:, 1]
    print("Propensity model AUC:", roc_auc_score(A, g_pred))
    return g_model, g_pred

def plot_propensity_scores(g_pred, A):
    """Plots the distribution of propensity scores."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(g_pred[A==0], label='Control (A=0)', fill=True, alpha=0.3)
    sns.kdeplot(g_pred[A==1], label='Treated (A=1)', fill=True, alpha=0.3)
    plt.title('Propensity Score Distribution')
    plt.xlabel('Propensity Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_outcome_model(X_proc, A, Y):
    """Trains the outcome regression model (Q-model)."""
    print("Training outcome model...")
    X_withA = X_proc.copy()
    X_withA['A'] = A

    q_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=0)
    q_model.fit(X_withA, Y)

    q_init_obs = q_model.predict_proba(X_withA)[:, 1]
    
    X1 = X_proc.copy(); X1['A'] = 1
    X0 = X_proc.copy(); X0['A'] = 0
    
    q_A1 = q_model.predict_proba(X1)[:, 1]
    q_A0 = q_model.predict_proba(X0)[:, 1]
    
    print("Outcome model AUC:", roc_auc_score(Y, q_init_obs))
    return q_init_obs, q_A1, q_A0

def run_tmle(Y, A, g_pred, q_init_obs, q_A1, q_A0):
    """Performs the TMLE targeting step."""
    print("Running TMLE...")
    eps_clip = 1e-6
    q_init = np.clip(q_init_obs, eps_clip, 1 - eps_clip)
    g = np.clip(g_pred, eps_clip, 1 - eps_clip)

    H = A/g - (1-A)/(1-g)
    offset = logit(q_init)

    fluc_model = sm.GLM(
        Y,
        H.reshape(-1,1),
        family=sm.families.Binomial(),
        offset=offset
    )
    fluc_res = fluc_model.fit()
    epsilon = float(fluc_res.params[0])

    print("Estimated epsilon:", epsilon)

    def update_q(q_vals, eps, H_vals):
        return expit(logit(np.clip(q_vals, eps_clip, 1-eps_clip)) + eps * H_vals)

    qstar_A1 = update_q(q_A1, epsilon, 1/g)
    qstar_A0 = update_q(q_A0, epsilon, -1/(1-g))
    qstar_obs = update_q(q_init, epsilon, H)

    ate = np.mean(qstar_A1 - qstar_A0)
    IC = H*(Y - qstar_obs) + (qstar_A1 - qstar_A0) - ate
    se = np.sqrt(np.var(IC, ddof=1) / len(Y))

    ci_low = ate - 1.96*se
    ci_high = ate + 1.96*se

    return ate, ci_low, ci_high, se, epsilon

def run_naive_analysis(X_proc, A, Y):
    """Runs a naive logistic regression for comparison."""
    print("Running naive analysis...")
    X_lr = pd.concat([pd.Series(A, name='A'), X_proc.reset_index(drop=True)], axis=1)
    lr = LogisticRegression(max_iter=2000).fit(X_lr, Y)

    X1_lr = X_lr.copy(); X1_lr['A'] = 1
    X0_lr = X_lr.copy(); X0_lr['A'] = 0

    naive_ate = lr.predict_proba(X1_lr)[:,1].mean() - lr.predict_proba(X0_lr)[:,1].mean()
    return naive_ate

def main():
    filepath = r"C:\Users\User\Desktop\AI project\data\healthcare-dataset-stroke-data.csv"
    df, X, A, Y = load_and_prep_data(filepath)
    
    if df is None:
        return

    X_proc = preprocess_data(X)
    
    smd_df = calculate_smd(X_proc, A)
    print("\nTop covariate imbalance before TMLE:")
    print(smd_df.head(10).to_string(index=False))
    plot_covariate_balance(smd_df)
    
    g_model, g_pred = train_propensity_model(X_proc, A)
    plot_propensity_scores(g_pred, A)
    
    q_init_obs, q_A1, q_A0 = train_outcome_model(X_proc, A, Y)
    
    ate, ci_low, ci_high, se, epsilon = run_tmle(Y, A, g_pred, q_init_obs, q_A1, q_A0)
    
    print("\nTMLE ATE =", ate)
    print("95% CI =", (ci_low, ci_high))
    print("Standard Error =", se)
    
    naive_ate = run_naive_analysis(X_proc, A, Y)
    print("\nNaive Marginal ATE =", naive_ate)
    
    # Save results
    output = pd.DataFrame({
        "ATE_TMLE":[ate],
        "CI_lower":[ci_low],
        "CI_upper":[ci_high],
        "SE":[se],
        "epsilon":[epsilon],
        "Naive_ATE":[naive_ate]
    })
    print("\nResults saved to output DataFrame (not written to disk in this script).")
    print(output)

if __name__ == "__main__":
    main()