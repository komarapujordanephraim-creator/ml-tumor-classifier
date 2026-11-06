"""Train and compare two classifiers on the Breast Cancer Wisconsin (Diagnostic) dataset.

The dataset ships with scikit-learn, so no download is needed.
Run:  python train.py
Outputs go to ./outputs (metrics.json, confusion_matrix.png, feature_importance.png, roc_curve.png).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
OUT = Path(__file__).parent / "outputs"


def build_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    }


def main():
    OUT.mkdir(exist_ok=True)
    data = load_breast_cancer()
    X, y = data.data, data.target
    names = list(data.feature_names)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, model in build_models().items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        results[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_te, pred)), 4),
            "test_precision": round(float(precision_score(y_te, pred)), 4),
            "test_recall": round(float(recall_score(y_te, pred)), 4),
            "test_f1": round(float(f1_score(y_te, pred)), 4),
            "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        }
        fitted[name] = model

    meta = {"dataset": "Breast Cancer Wisconsin (Diagnostic), scikit-learn bundled",
            "rows": int(X.shape[0]), "features": int(X.shape[1]),
            "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)), "seed": SEED}
    (OUT / "metrics.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))

    best = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    model = fitted[best]
    plt.rcParams.update({"figure.facecolor": "#0b0d14", "axes.facecolor": "#0b0d14", "text.color": "#e8ecf4",
                         "axes.labelcolor": "#e8ecf4", "xtick.color": "#9aa3b5", "ytick.color": "#9aa3b5",
                         "axes.edgecolor": "#2a3040"})
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay.from_estimator(model, X_te, y_te, display_labels=data.target_names, cmap="cividis", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix: {best}")
    fig.tight_layout(); fig.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for n, m in fitted.items():
        RocCurveDisplay.from_estimator(m, X_te, y_te, name=n, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="#555b6b"); ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(OUT / "roc_curve.png", dpi=160); plt.close(fig)

    rf = fitted["Random Forest"]
    idx = np.argsort(rf.feature_importances_)[-10:]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh([names[i] for i in idx], rf.feature_importances_[idx], color="#e6c878")
    ax.set_title("Top 10 features (Random Forest)")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=160); plt.close(fig)

    for n, r in results.items():
        print(f"{n:20s} CV AUC {r['cv_roc_auc_mean']:.3f}+/-{r['cv_roc_auc_std']:.3f} | test acc {r['test_accuracy']:.3f} "
              f"recall {r['test_recall']:.3f} AUC {r['test_roc_auc']:.3f}")
    print("Best by CV AUC:", best)


if __name__ == "__main__":
    main()
