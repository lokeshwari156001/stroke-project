"""
run_full_pipeline.py

Full end-to-end stroke pipeline (EDA -> Cleaning -> FE -> Preproc -> Balance -> Train (XGBoost) -> Tune -> Eval -> SHAP)
Dataset path used: c:\\Users\\User\\Desktop\\AI project\\data\\healthcare-dataset-stroke-data.csv

Behavior:
 - If `imblearn` installed -> use SMOTE on training data (recommended).
 - Otherwise -> fallback to random oversampling of minority class (sampling with replacement).
 - XGBoost configured to use CPU-friendly options to avoid GPU/CUDA import issues.
 - Saves outputs to c:\\Users\\User\\Desktop\\AI project\\data\\stroke_outputs\\
"""

import os, sys, joblib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

# Adjusted paths for Windows environment
OUTDIR = Path(r"c:\Users\User\Desktop\AI project\data\stroke_outputs")
FIGDIR = OUTDIR / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = r"c:\Users\User\Desktop\AI project\data\healthcare-dataset-stroke-data.csv"
RANDOM_STATE = 42

# Try to import imblearn.SMOTE
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False

# Try to import xgboost (CPU-friendly config)
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise RuntimeError(
        "xgboost import failed. Install a CPU-compatible xgboost (e.g. pip install xgboost) "
        "or ensure system libraries are compatible. Error: " + str(e)
    )

# Try shap (optional, we'll catch failures)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def load_and_describe(path=DATA_PATH):
    df = pd.read_csv(path)
    info = {
        "shape": df.shape,
        "missing": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.apply(lambda x: x.name).to_dict(),
        "target_counts": df['stroke'].value_counts().to_dict()
    }
    return df, info

def clean_and_fe(df, use_age_bin=True, use_bmi_cat=True, use_risk_score=True):
    df = df.copy()
    if 'id' in df.columns: df = df.drop(columns=['id'])
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()
    # outlier caps
    if 'avg_glucose_level' in df.columns:
        q99 = df['avg_glucose_level'].quantile(0.99)
        df['avg_glucose_level'] = np.where(df['avg_glucose_level']>q99, q99, df['avg_glucose_level'])
    if 'bmi' in df.columns:
        q1 = df['bmi'].quantile(0.25); q3 = df['bmi'].quantile(0.75); iqr = q3-q1
        df['bmi'] = df['bmi'].clip(q1-1.5*iqr, q3+1.5*iqr)
    df = df[(df['age']>0)&(df['age']<120)]
    
    # FE
    if use_age_bin:
        df['age_bin'] = pd.cut(df['age'], bins=[0,18,35,50,65,120], labels=['child','young','mid','senior','elder'])
    if use_bmi_cat:
        df['bmi_cat'] = pd.cut(df['bmi'].fillna(df['bmi'].median()), bins=[0,18.5,25,30,100], labels=['underweight','normal','overweight','obese'])
    if use_risk_score:
        df['risk_score'] = (df['hypertension'].fillna(0).astype(int) + df['heart_disease'].fillna(0).astype(int) + (df['avg_glucose_level'].fillna(0)>140).astype(int))
    return df

def eda_plots(df):
    # target count
    plt.figure(figsize=(4,3)); vals=df['stroke'].value_counts().sort_index(); plt.bar(['no','yes'], vals.values); plt.title('Stroke count'); plt.tight_layout(); plt.savefig(FIGDIR/'stroke_count.png'); plt.close()
    # age hist
    plt.figure(figsize=(6,3)); plt.hist(df['age'].dropna(), bins=30); plt.title('Age distribution'); plt.tight_layout(); plt.savefig(FIGDIR/'age_dist.png'); plt.close()
    # glucose box
    plt.figure(figsize=(6,3))
    data0=df[df['stroke']==0]['avg_glucose_level'].dropna(); data1=df[df['stroke']==1]['avg_glucose_level'].dropna()
    plt.boxplot([data0, data1], labels=['no','yes']); plt.title('Glucose by stroke'); plt.tight_layout(); plt.savefig(FIGDIR/'glucose_box.png'); plt.close()
    # correlation heatmap numeric
    num_cols=[c for c in ['age','avg_glucose_level','bmi','hypertension','heart_disease','risk_score'] if c in df.columns]
    corr=df[num_cols].corr()
    plt.figure(figsize=(6,5)); im=plt.imshow(corr, cmap='viridis', aspect='auto'); plt.colorbar(im)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45); plt.yticks(range(len(num_cols)), num_cols)
    for i in range(len(num_cols)):
        for j in range(len(num_cols)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', color='white', fontsize=8)
    plt.title('Numeric correlation'); plt.tight_layout(); plt.savefig(FIGDIR/'corr_heatmap.png'); plt.close()

def build_preprocessor(df):
    numeric_features=[c for c in ['age','avg_glucose_level','bmi','risk_score'] if c in df.columns]
    categorical_features=[c for c in ['gender','ever_married','work_type','Residence_type','smoking_status','age_bin','bmi_cat'] if c in df.columns]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) 
    preproc = ColumnTransformer([('num', num_pipe, numeric_features), ('cat', cat_pipe, categorical_features)])
    return preproc, numeric_features, categorical_features

def oversample_random(X, y, random_state=RANDOM_STATE):
    # simple random oversampling (fallback if SMOTE unavailable)
    df = X.copy()
    df['_target_'] = y.values
    counts = df['_target_'].value_counts()
    majority = counts.max(); classes = counts.index.tolist()
    frames = [df]
    for cls in classes:
        cnt = counts.loc[cls]
        if cnt < majority:
            to_sample = majority - cnt
            sampled = df[df['_target_']==cls].sample(n=to_sample, replace=True, random_state=random_state)
            frames.append(sampled)
    res = pd.concat(frames, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    y_res = res['_target_']; X_res = res.drop(columns=['_target_'])
    return X_res, y_res

def train_and_tune(df, balancing_method='smote'):
    X = df.drop(columns=['stroke']); y = df['stroke'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    preproc, num_feats, cat_feats = build_preprocessor(df)
    
    # balance training set
    use_imblearn_pipeline = False
    X_train_res, y_train_res = X_train, y_train # default no balancing
    
    if balancing_method == 'smote' and IMBLEARN_AVAILABLE:
        use_imblearn_pipeline = True
    elif balancing_method == 'random':
        X_train_res, y_train_res = oversample_random(X_train, y_train, random_state=RANDOM_STATE)
    elif balancing_method == 'smote' and not IMBLEARN_AVAILABLE:
        print("Warning: SMOTE requested but imblearn not available. Falling back to random oversampling.")
        X_train_res, y_train_res = oversample_random(X_train, y_train, random_state=RANDOM_STATE)


    # Build pipeline for training:
    clf = XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE,
                        n_jobs=1, tree_method='hist')  # CPU-friendly config
    if IMBLEARN_AVAILABLE:
        # build imblearn pipeline so SMOTE is applied inside cross-val (preferred)
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        imb_pipe = ImbPipeline([('preproc', preproc), ('smote', SMOTE(random_state=RANDOM_STATE)), ('clf', clf)])
        search_estimator = imb_pipe
    else:
        # we already oversampled X_train_res (raw dataframe) so build normal pipeline
        pipe = Pipeline([('preproc', preproc), ('clf', clf)])
        search_estimator = pipe

    param_dist = {
        'clf__n_estimators': [50,100,200],
        'clf__max_depth': [3,6,8],
        'clf__learning_rate': [0.01,0.05,0.1],
        'clf__subsample': [0.6,0.8,1.0]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(search_estimator, param_distributions=param_dist, n_iter=6, scoring='roc_auc',
                            n_jobs=1, cv=cv, random_state=RANDOM_STATE, verbose=1)
    if IMBLEARN_AVAILABLE:
        # fit with pipeline that includes SMOTE internally (X_train, y_train)
        rs.fit(X_train, y_train)
    else:
        # fit using our oversampled training dataframe
        rs.fit(X_train_res, y_train_res)
    best = rs.best_estimator_
    # Evaluate
    y_pred = best.predict(X_test); y_prob = best.predict_proba(X_test)[:,1]
    roc = roc_auc_score(y_test, y_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_prob); pr_auc = auc(recall, precision)
    rep_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
    cm = confusion_matrix(y_test, y_pred)
    # save artifacts
    from sklearn.metrics import roc_curve, mean_squared_error, f1_score
    fpr,tpr,_ = roc_curve(y_test, y_prob)
    mse_score = mean_squared_error(y_test, y_prob)
    rmse_score = np.sqrt(mse_score)
    f1 = f1_score(y_test, y_pred)
    
    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC AUC={roc:.4f}'); plt.tight_layout(); plt.savefig(FIGDIR/'roc_curve.png'); plt.close()
    plt.figure(); plt.plot(recall, precision); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR AUC={pr_auc:.4f}'); plt.tight_layout(); plt.savefig(FIGDIR/'pr_curve.png'); plt.close()
    
    # save artifacts
    rep_df.to_csv(OUTDIR/"classification_report.csv")
    pd.DataFrame({
        "roc_auc": [roc],
        "pr_auc": [pr_auc],
        "mse": [mse_score],
        "rmse": [rmse_score],
        "f1_score": [f1]
    }).to_csv(OUTDIR/"metrics_summary.csv")
    # SHAP (if available)
    if SHAP_AVAILABLE:
        try:
            # for shap, transform a small sample via preprocessor and explain classifier
            preproc_part = best.named_steps.get('preproc', None)
            clf_part = best.named_steps.get('clf', None)
            if preproc_part is not None and clf_part is not None:
                X_sample = X_train.sample(n=min(200, X_train.shape[0]), random_state=RANDOM_STATE)
                X_trans = preproc_part.transform(X_sample)
                explainer = shap.TreeExplainer(clf_part)
                shap_values = explainer.shap_values(X_trans)
                plt.figure(figsize=(6,4)); shap.summary_plot(shap_values, X_trans, show=False); plt.tight_layout(); plt.savefig(FIGDIR/'shap_summary.png'); plt.close()
                mean_abs = np.abs(shap_values).mean(axis=0)
                # feature names: numeric + OHE names (best effort)
                try:
                    ohe = preproc_part.named_transformers_['cat'].named_steps['ohe']; cat_names = list(ohe.get_feature_names_out())
                except Exception:
                    cat_names = []
                feat_names = list(num_feats) + cat_names
                inds = np.argsort(mean_abs)[::-1][:30]
                plt.figure(figsize=(6,6)); plt.barh([feat_names[i] for i in inds[::-1]], mean_abs[inds[::-1]]); plt.xlabel('mean(|SHAP|)'); plt.title('SHAP importance'); plt.tight_layout(); plt.savefig(FIGDIR/'shap_bar.png'); plt.close()
        except Exception as e:
            print("SHAP step failed:", e)
    return {"best_params": rs.best_params_, "roc_auc": roc, "pr_auc": pr_auc, "outputs_dir": str(OUTDIR)}

def main():
    print("Loading dataset from:", DATA_PATH)
    df, info = load_and_describe(DATA_PATH)
    print("Initial info:", info)
    df = clean_and_fe(df)
    eda_plots(df)
    results = train_and_tune(df)
    print("Done. Outputs saved to:", results['outputs_dir'])
    print("Best params:", results['best_params'])
    print("ROC AUC:", results['roc_auc'], " PR AUC:", results['pr_auc'])
    print("Model file:", OUTDIR/'best_model.joblib')
    print("Figures:", list(FIGDIR.iterdir()))
    # provide short listing
    for f in (OUTDIR).iterdir():
        print("-", f.name)

if __name__ == "__main__":
    main()
