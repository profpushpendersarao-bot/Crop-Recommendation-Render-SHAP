"""Train and evaluate the Crop Recommendation classification models.

The split and model settings follow crop-recommendation-system.ipynb:
80:20 stratified train/test split, random_state=42; five candidate classifiers;
Random Forest is retained as the final model as specified in the notebook.
Five-fold StratifiedKFold cross-validation is run on the training partition only.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import shap

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "crop_recommendation_model.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

def main():
    df = pd.read_csv(DATA_PATH)
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_cols)}")
    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Dataset contains missing values. Clean them before training.")

    X = df[FEATURES].copy()
    y = df[TARGET].astype(str).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    metrics = {}
    candidates = build_models()

    print(f"Dataset: {len(df)} rows, {y.nunique()} crop classes")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Running {N_SPLITS}-fold stratified CV on training data...")
    for name, estimator in candidates.items():
        scores = cross_val_score(
            estimator, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1
        )
        fitted = estimator.fit(X_train, y_train)
        test_pred = fitted.predict(X_test)
        metrics[name] = {
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std(ddof=1)),
            "test_accuracy": float(accuracy_score(y_test, test_pred)),
        }
        print(
            f"{name}: CV accuracy {scores.mean():.4f} ± {scores.std(ddof=1):.4f}; "
            f"holdout accuracy {metrics[name]['test_accuracy']:.4f}"
        )

    # The source notebook explicitly chooses Random Forest as its final model.
    rf_eval = candidates["Random Forest"].fit(X_train, y_train)
    y_pred = rf_eval.predict(X_test)
    y_proba = rf_eval.predict_proba(X_test)
    classes = sorted(y.unique().tolist())
    y_test_binary = label_binarize(y_test, classes=classes)
    macro_auc = roc_auc_score(y_test_binary, y_proba, average="macro")

    # Refit final Random Forest on all available records for deployment.
    final_model = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    final_model.fit(X, y)

    # Global SHAP importance on a reproducible sample of held-out examples.
    shap_sample = X_test.sample(n=min(100, len(X_test)), random_state=RANDOM_STATE)
    shap_explainer = shap.TreeExplainer(final_model)
    shap_explanation = shap_explainer(shap_sample, check_additivity=False)
    shap_array = np.asarray(shap_explanation.values)
    # Multiclass output is generally (samples, features, classes).
    if shap_array.ndim == 3 and shap_array.shape[1] == len(FEATURES):
        global_shap_values = np.mean(np.abs(shap_array), axis=(0, 2))
    elif shap_array.ndim == 3 and shap_array.shape[2] == len(FEATURES):
        global_shap_values = np.mean(np.abs(shap_array), axis=(0, 1))
    elif shap_array.ndim == 2:
        global_shap_values = np.mean(np.abs(shap_array), axis=0)
    else:
        raise ValueError(f"Unexpected SHAP output shape: {shap_array.shape}")
    shap_global_importance = [
        {"feature": feature, "importance": float(value)}
        for feature, value in sorted(zip(FEATURES, global_shap_values), key=lambda item: item[1], reverse=True)
    ]

    bundle = {
        "model": final_model,
        "features": FEATURES,
        "classes": classes,
        "feature_ranges": {
            col: {"min": float(X[col].min()), "max": float(X[col].max())}
            for col in FEATURES
        },
        "model_metrics": metrics,
        "random_forest_holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "random_forest_macro_roc_auc": float(macro_auc),
        "training_rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "cv_folds": N_SPLITS,
        "shap_global_importance": shap_global_importance,
        "shap_global_sample_size": int(len(shap_sample)),
    }
    joblib.dump(bundle, MODEL_PATH, compress=3)

    results = {
        "dataset_shape": list(df.shape),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "crop_count": int(y.nunique()),
        "model_metrics": metrics,
        "random_forest_holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "random_forest_macro_roc_auc": float(macro_auc),
        "random_forest_classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
        "class_labels": classes,
    }
    (BASE_DIR / "validation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(BASE_DIR / "confusion_matrix.csv")
    print(f"\nRandom Forest holdout accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Random Forest macro ROC-AUC: {macro_auc:.4f}")
    print(f"Saved deployment model: {MODEL_PATH}")

if __name__ == "__main__":
    main()
