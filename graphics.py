# -*- coding: utf-8 -*-
"""
Generación de figuras para el reporte LaTeX:
- figures/roc_comparativo.png
- figures/pr_comparativo.png
- figures/confusion_matrices.png
- figures/feature_importance_rf_xgb.png

Usa los artefactos generados por el flujo (preprocessor y modelos) para
reconstruir el conjunto de prueba, calcular probabilidades y trazar curvas.
"""

from pathlib import Path
import io
import zipfile
import requests
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)

from tensorflow.keras.models import load_model


# ----------------------- Configuración de rutas -----------------------
ARTIFACTS_DIR = Path("artifacts")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# Rutas de artefactos esperados
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"
DT_PATH = ARTIFACTS_DIR / "baseline_dt.joblib"
RF_PATH = ARTIFACTS_DIR / "best_rf.joblib"
XGB_PATH = ARTIFACTS_DIR / "best_xgb.joblib"
NN_PATH = ARTIFACTS_DIR / "best_nn.h5"

# Salidas esperadas para LaTeX
ROC_FPATH = FIGURES_DIR / "roc_comparativo.png"
PR_FPATH = FIGURES_DIR / "pr_comparativo.png"
CM_FPATH = FIGURES_DIR / "confusion_matrices.png"
FI_FPATH = FIGURES_DIR / "feature_importance_rf_xgb.png"


# ----------------------- Utilidades -----------------------
def load_uci_bank_additional_full():
    """Descarga y carga bank-additional-full.csv desde UCI (sep=';')."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open("bank-additional/bank-additional-full.csv") as f:
        df = pd.read_csv(f, sep=';')
    return df


def prepare_test_set(df, test_size=0.2, random_state=42, preprocessor=None):
    """Replica el split 80/20 estratificado y transforma el X_test con el preprocessor guardado."""
    df = df.copy()
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    X = df.drop(columns=["y"])
    y = df["y"].values
    # El flujo original ajustó el preprocessor sobre X_train; aquí necesitamos el mismo split
    # para aplicar 'transform' sobre el X_test con ese preprocessor ya ajustado.
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    if preprocessor is None:
        raise FileNotFoundError("No se encontró el preprocessor.joblib ya ajustado.")
    X_test_t = preprocessor.transform(X_test)
    return X_test, X_test_t, y_test


def proba_from_model(model, X):
    """Obtiene probabilidades de clase positiva para distintos tipos de modelo."""
    # Keras (modelo de salida sigmoide)
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.models import Model  # noqa: F401
        # Si es un objeto Keras, predict devuelve probabilidades
        if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
            p = model.predict(X, verbose=0)
            return p.reshape(-1)
    except Exception:
        pass

    # Modelos scikit-learn/xgboost
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    # Fallback: si solo hay predict (clases), convertir a 0/1 float (no ideal para ROC/PR)
    preds = model.predict(X)
    return preds.astype(float)


def plot_roc(models_dict, y_true, out_path, dpi=300):
    plt.figure(figsize=(8, 6))
    colors = {
        "Decision Tree": "tab:gray",
        "Random Forest": "tab:green",
        "XGBoost": "tab:orange",
        "Neural Network": "tab:blue",
    }
    for name, proba in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colors.get(name, None),
                 label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("FPR (Tasa de Falsos Positivos)")
    plt.ylabel("TPR (Tasa de Verdaderos Positivos)")
    plt.title("Curvas ROC comparativas")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_pr(models_dict, y_true, out_path, dpi=300):
    plt.figure(figsize=(8, 6))
    colors = {
        "Decision Tree": "tab:gray",
        "Random Forest": "tab:green",
        "XGBoost": "tab:orange",
        "Neural Network": "tab:blue",
    }
    # Línea base (proporción positiva)
    baseline = (y_true == 1).mean()
    plt.hlines(baseline, 0, 1, linestyles="dashed", colors="k", label=f"Baseline (p+= {baseline:.3f})", lw=1)

    for name, proba in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        plt.step(recall, precision, where="post", lw=2, color=colors.get(name, None),
                 label=f"{name} (AP = {ap:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precisión–Recall comparativas")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def annotate_cm(ax, cm, normalize=False):
    """Dibuja anotaciones sobre una matriz de confusión."""
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)
    else:
        cm_display = cm

    im = ax.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    thresh = cm_display.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm_display[i, j]:.2f}" if normalize else f"{cm_display[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black", fontsize=10)

    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Pred. 0 (No)", "Pred. 1 (Sí)"],
           yticklabels=["Real 0 (No)", "Real 1 (Sí)"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")


def plot_confusion_matrices(y_true, preds_dict, out_path, dpi=300):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Matrices de confusión (umbral 0.5)", fontsize=14, y=1.02)

    names = ["Decision Tree", "Random Forest", "XGBoost", "Neural Network"]
    for ax, name in zip(axes.ravel(), names):
        preds = preds_dict[name]
        cm = confusion_matrix(y_true, preds)
        ax.set_title(name)
        annotate_cm(ax, cm, normalize=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def get_feature_names(preprocessor, input_df):
    """Obtiene nombres de características después del ColumnTransformer."""
    try:
        # sklearn >= 1.0
        return preprocessor.get_feature_names_out()
    except Exception:
        # Fallback mínimo
        return np.array([f"f{i}" for i in range(preprocessor.transform(input_df[:1]).shape[1])])


def plot_feature_importances(preprocessor, X_test_df, rf_model, xgb_model, out_path, top_k=15, dpi=300):
    feat_names = get_feature_names(preprocessor, X_test_df)
    # Asegurar longitudes
    rf_imp = getattr(rf_model, "feature_importances_", None)
    xgb_imp = getattr(xgb_model, "feature_importances_", None)
    if rf_imp is None or xgb_imp is None:
        print("Aviso: no se encontraron importancias en RF/XGB; se omite esta figura.")
        return

    # Top-k por modelo
    def topk(feat_names, importances, k):
        idx = np.argsort(importances)[-k:]
        return feat_names[idx], importances[idx]

    rf_names_top, rf_imp_top = topk(feat_names, rf_imp, top_k)
    xgb_names_top, xgb_imp_top = topk(feat_names, xgb_imp, top_k)

    # Ordenar para barh (ascendente para que el más alto quede arriba)
    rf_order = np.argsort(rf_imp_top)
    xgb_order = np.argsort(xgb_imp_top)

    plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(1, 2, 1)
    ax1.barh(range(top_k), rf_imp_top[rf_order], color="tab:green")
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels([str(n).replace("cat__", "").replace("num__", "") for n in rf_names_top[rf_order]], fontsize=8)
    ax1.set_title("Random Forest: Top {} importancias".format(top_k))
    ax1.set_xlabel("Importancia")

    ax2 = plt.subplot(1, 2, 2)
    ax2.barh(range(top_k), xgb_imp_top[xgb_order], color="tab:orange")
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels([str(n).replace("cat__", "").replace("num__", "") for n in xgb_names_top[xgb_order]], fontsize=8)
    ax2.set_title("XGBoost: Top {} importancias".format(top_k))
    ax2.set_xlabel("Importancia")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    # 1) Cargar artefactos
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    dt = joblib.load(DT_PATH)
    rf = joblib.load(RF_PATH)
    try:
        from xgboost import XGBClassifier  # noqa: F401
        xgb = joblib.load(XGB_PATH)
    except Exception as e:
        raise RuntimeError("Error cargando el modelo XGBoost: {}".format(e))

    # Cargar NN en formato Keras
    nn = load_model(NN_PATH)

    # 2) Cargar datos y preparar X_test e y_test (mismo split del flujo)
    df = load_uci_bank_additional_full()
    X_test_df, X_test, y_test = prepare_test_set(df, preprocessor=preprocessor)

    # 3) Probabilidades y predicciones (umbral 0.5)
    models = {
        "Decision Tree": dt,
        "Random Forest": rf,
        "XGBoost": xgb,
        "Neural Network": nn,
    }
    probas = {name: proba_from_model(model, X_test) for name, model in models.items()}
    preds = {name: (p >= 0.5).astype(int) for name, p in probas.items()}

    # 4) Curvas ROC comparativas
    plot_roc(probas, y_test, ROC_FPATH, dpi=350)

    # 5) Curvas Precisión–Recall comparativas
    plot_pr(probas, y_test, PR_FPATH, dpi=350)

    # 6) Matrices de confusión
    plot_confusion_matrices(y_test, preds, CM_FPATH, dpi=350)

    # 7) Importancias de variables (RF y XGB)
    plot_feature_importances(preprocessor, X_test_df, rf, xgb, FI_FPATH, top_k=15, dpi=350)

    # Impresión rápida de verificación
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
    print("Verificación rápida de métricas en el set de prueba reconstruido:")
    for name in ["Decision Tree", "Random Forest", "XGBoost", "Neural Network"]:
        y_prob = probas[name]
        y_hat = preds[name]
        auc_val = roc_auc_score(y_test, y_prob)
        f1_val = f1_score(y_test, y_hat)
        acc_val = accuracy_score(y_test, y_hat)
        pre_val = precision_score(y_test, y_hat)
        rec_val = recall_score(y_test, y_hat)
        print(f"- {name}: AUC={auc_val:.3f} | F1={f1_val:.3f} | Acc={acc_val:.3f} | P={pre_val:.3f} | R={rec_val:.3f}")

    print(f"\nFiguras guardadas en: {FIGURES_DIR.resolve()}")
    print(f"- ROC: {ROC_FPATH.name}")
    print(f"- PR: {PR_FPATH.name}")
    print(f"- Matrices de confusión: {CM_FPATH.name}")
    print(f"- Importancias RF/XGB: {FI_FPATH.name}")


if __name__ == "__main__":
    main()
