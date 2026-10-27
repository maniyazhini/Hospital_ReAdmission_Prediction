"""
Hospital Readmission Prediction — training pipeline

Run from the project root:
    python train_model.py

Expected dataset:
    dataset/diabetic_data.csv
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "diabetic_data.csv"
MODEL_DIR = ROOT / "model"
PLOT_DIR = MODEL_DIR / "plots"
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# A compact, explainable feature set for the academic project.
# These are available in the UCI dataset.
FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
    "insulin",
    "diabetesMed",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "admission_source_id",
    "insulin",
    "diabetesMed",
]

NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]


def load_and_prepare():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download diabetic_data.csv and place it in dataset/"
        )

    df = pd.read_csv(DATA_PATH)
    df = df.replace("?", np.nan)

    missing_columns = [c for c in FEATURES + ["readmitted"] if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Binary target: early readmission within 30 days.
    df["target"] = (df["readmitted"].astype(str).str.strip() == "<30").astype(int)

    # Remove rows where the target is missing (normally none).
    df = df.dropna(subset=["target"])

    X = df[FEATURES].copy()
    y = df["target"].astype(int)

    # Convert known numeric fields to numeric; invalid values become NaN
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return df, X, y


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def make_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", model),
    ])


def evaluate_model(name, pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob) if prob is not None else np.nan,
    }

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    print(f"ROC-AUC: {metrics['ROC-AUC']:.4f}")

    return pipeline, metrics, pred


def make_eda_plots(df):
    # Target distribution
    plt.figure(figsize=(7, 5))
    labels = ["Not readmitted within 30 days", "Readmitted within 30 days"]
    counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
    sns.barplot(x=labels, y=counts.values)
    plt.title("30-Day Readmission Distribution")
    plt.ylabel("Number of Encounters")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "readmission_distribution.png", dpi=160)
    plt.close()

    # Length of stay vs target
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="target", y="time_in_hospital")
    plt.title("Time in Hospital vs 30-Day Readmission")
    plt.xlabel("Target (0 = No early readmission, 1 = Readmitted <30 days)")
    plt.ylabel("Days in Hospital")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "stay_vs_readmission.png", dpi=160)
    plt.close()

    # Inpatient visits vs target
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="target", y="number_inpatient")
    plt.title("Previous Inpatient Visits vs 30-Day Readmission")
    plt.xlabel("Target")
    plt.ylabel("Number of Inpatient Visits")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "previous_inpatient_vs_readmission.png", dpi=160)
    plt.close()


def main():
    print("Loading dataset...")
    df, X, y = load_and_prepare()

    print(f"Dataset shape: {df.shape}")
    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    make_eda_plots(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = []
    fitted = {}

    for name, estimator in models.items():
        pipeline, metrics, pred = evaluate_model(
            name,
            make_pipeline(estimator),
            X_train, X_test, y_train, y_test
        )
        fitted[name] = pipeline
        results.append(metrics)

    results_df = pd.DataFrame(results).sort_values(
        "F1-Score", ascending=False
    ).reset_index(drop=True)

    print("\nMODEL COMPARISON")
    print(results_df.to_string(index=False))

    results_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    best_name = results_df.loc[0, "Model"]
    best_pipeline = fitted[best_name]

    # Confusion matrix for the best model
    best_pred = best_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, best_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "confusion_matrix_best_model.png", dpi=160)
    plt.close()

    # Save everything needed by the Streamlit app.
    bundle = {
        "pipeline": best_pipeline,
        "model_name": best_name,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "category_options": {
            col: sorted(X[col].dropna().astype(str).unique().tolist())
            for col in CATEGORICAL_FEATURES
        },
        "numeric_defaults": {
            col: float(pd.to_numeric(X[col], errors="coerce").median())
            for col in NUMERIC_FEATURES
        },
        "target_definition": {
            1: "Readmitted within 30 days",
            0: "Not readmitted within 30 days",
        },
    }

    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib")

    print(f"\nBest model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'model_bundle.joblib'}")
    print(f"Saved comparison: {MODEL_DIR / 'model_comparison.csv'}")
    print(f"Saved plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
