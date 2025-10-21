"""
proyecto: bank marketing con prefect
descripcion: flujo orquestado que carga datos, prepara, busca hiperparametros para redes neuronales, xgboost y random forest,
compara con decision tree, guarda modelos y reporta mejores hiperparametros y numero de neuronas.
"""
from prefect.client import get_client

from pathlib import Path
import time
import json
import joblib
import tempfile
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, ParameterSampler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scikeras.wrappers import KerasClassifier
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# prefect v2
from prefect import flow, task, get_run_logger

# keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# optimizar uso de GPU en tensorflow (si estuviera disponible)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

# ruta para guardar artefactos
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# este task descarga o carga el dataset desde la URL de UCI
@task
def load_data(url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip") -> pd.DataFrame:
    logger = get_run_logger()
    logger.info("cargando datos desde uci")
    import zipfile, requests, io
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open('bank-additional/bank-additional-full.csv') as f:
        df = pd.read_csv(f, sep=';')
    logger.info(f"datos cargados con shape {df.shape}")
    return df

# este task hace la preparacion de datos: codificacion y split
@task
def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    logger = get_run_logger()
    logger.info("iniciando preparacion de datos")
    df = df.copy()
    df['y'] = df['y'].map({'yes':1, 'no':0})

    categorical = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical:
        categorical.remove('y')
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'y' in numeric:
        numeric.remove('y')

    # encoder + scaler
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    preprocessor = ColumnTransformer([
        ('cat', encoder, categorical),
        ('num', scaler, numeric),
    ])

    X = df.drop(columns=['y'])
    y = df['y'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    sm = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = sm.fit_resample(X_train_t, y_train)

    joblib.dump(preprocessor, ARTIFACTS_DIR / 'preprocessor.joblib')

    logger.info(f"preparacion finalizada: train {X_train_bal.shape}, test {X_test_t.shape}")
    return X_train_bal, X_test_t, y_train_bal, y_test, preprocessor

# helper para construir Keras model
def build_keras_model(input_shape, neurons=32, activation='relu',
                      optimizer='adam', learning_rate=1e-3, **kwargs):

    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # salida binaria

    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == "nadam":
        opt = keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# este task realiza el tuning de la red neuronal (búsqueda manual con barra de progreso)
@task
def tune_neural_network(X, y, n_iter=12, random_state=42, cv_folds=3):
    logger = get_run_logger()
    logger.info("tuneando red neuronal (manual search con tqdm)")

    input_shape = X.shape[1]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # espacio de búsqueda reducido pero representativo (para mantener tiempo razonable)
    param_dist = {
        "model__input_shape": [input_shape],
        "model__neurons": [32, 64],
        "model__activation": ["relu", "tanh"],
        "model__optimizer": ["adam", "nadam"],
        "model__learning_rate": [1e-3, 5e-4],
        "batch_size": [32, 64],
        "epochs": [10, 15]
    }

    sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=random_state))

    best_score = -np.inf
    best_params = None
    best_estimator = None

    # barra de progreso para las n_iter pruebas
    for params in tqdm(sampler, desc="NN tuning", unit="trial"):
        # construir un KerasClassifier con esos params (scikeras)
        # NOTA: scikeras recibirá los prefijos model__* para pasar a la función build_keras_model
        kc = KerasClassifier(
            model=build_keras_model,
            verbose=0,
            random_state=random_state,
            model__input_shape=input_shape
        )

        # preparar parámetros explícitos para fit
        fit_params = {}
        batch_size = params.get("batch_size", 32)
        epochs = params.get("epochs", 10)
        # map model__ keys to estimator parameter names for scikeras
        model_params = {}
        for k, v in params.items():
            if k.startswith("model__"):
                model_params[k] = v

        # set params on estimator (scikeras accepts model__ prefixed params)
        kc.set_params(**model_params)

        # cross-validation manual to allow progress and stable behavior with TF (n_jobs=1)
        cv_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            try:
                kc.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)
                preds = kc.predict(X_val)
                score = accuracy_score(y_val, preds)
            except Exception:
                score = np.nan
            cv_scores.append(score)

        # compute mean score ignoring nans
        cv_scores = np.array(cv_scores, dtype=float)
        mean_score = np.nanmean(cv_scores)

        if np.isfinite(mean_score) and mean_score > best_score:
            best_score = mean_score
            best_params = params.copy()
            best_estimator = kc

    logger.info(f"nn tuning finished best score {best_score}")

    # guardar mejor modelo (si existe)
    try:
        if best_estimator is not None:
            # reentrenar mejor_estimator en todo el train set para guardarlo
            best_estimator.fit(X, y, epochs=best_params.get("epochs", 10), batch_size=best_params.get("batch_size", 32), verbose=0)
            best_estimator.model_.save(ARTIFACTS_DIR / "best_nn.h5")
    except Exception:
        pass

    with open(ARTIFACTS_DIR / "best_nn_params.json", "w") as f:
        json.dump(best_params if best_params is not None else {}, f)

    return best_estimator, float(best_score) if np.isfinite(best_score) else None, best_params

# este task tunea random forest (usar RandomizedSearchCV con verbose para ver progreso)
@task
def tune_random_forest(X, y, n_iter=24, random_state=42, cv_folds=4):
    logger = get_run_logger()
    logger.info("tuneando random forest")
    rf = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=2
    )
    start = time.time()
    rs.fit(X, y)
    end = time.time()
    logger.info(f"rf tuning time: {end-start:.1f}s best score {rs.best_score_}")
    joblib.dump(rs.best_estimator_, ARTIFACTS_DIR / 'best_rf.joblib')
    with open(ARTIFACTS_DIR / 'best_rf_params.json', 'w') as f:
        json.dump(rs.best_params_, f)
    return rs.best_estimator_, rs.best_score_, rs.best_params_

# este task tunea xgboost (si no hay GPU, xgboost caerá a hist; usar verbose)
@task
def tune_xgboost(X, y, n_iter=24, random_state=42, cv_folds=4):
    logger = get_run_logger()
    logger.info("tuneando xgboost")
    # preferir gpu_hist si hay GPU disponible; xgboost detecta fallas y usa cpu
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, n_jobs=-1)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=2
    )
    start = time.time()
    rs.fit(X, y)
    end = time.time()
    logger.info(f"xgb tuning time: {end-start:.1f}s best score {rs.best_score_}")
    joblib.dump(rs.best_estimator_, ARTIFACTS_DIR / 'best_xgb.joblib')
    with open(ARTIFACTS_DIR / 'best_xgb_params.json', 'w') as f:
        json.dump(rs.best_params_, f)
    return rs.best_estimator_, rs.best_score_, rs.best_params_

# este task entrena y evalua un modelo baseline decision tree
@task
def baseline_decision_tree(X_train, y_train, X_test, y_test):
    logger = get_run_logger()
    logger.info("entrenando baseline decision tree")
    dt = DecisionTreeClassifier(random_state=42)
    start = time.time()
    dt.fit(X_train, y_train)
    end = time.time()
    preds = dt.predict(X_test)
    probs = dt.predict_proba(X_test)[:,1]
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds)),
        'recall': float(recall_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs)),
        'train_time_s': end-start
    }
    joblib.dump(dt, ARTIFACTS_DIR / 'baseline_dt.joblib')
    with open(ARTIFACTS_DIR / 'baseline_dt_metrics.json', 'w') as f:
        json.dump(metrics, f)
    logger.info(f"baseline metrics: {metrics}")
    return dt, metrics

# este task evalua cualquier modelo ya entrenado
@task
def evaluate_model(model, X_test, y_test, name: str):
    logger = get_run_logger()
    logger.info(f"evaluando modelo {name}")
    preds = model.predict(X_test)
    try:
        probs = model.predict_proba(X_test)[:,1]
    except Exception:
        probs = None
    metrics = {
        'accuracy': float(accuracy_score(y_test, preds)),
        'precision': float(precision_score(y_test, preds)),
        'recall': float(recall_score(y_test, preds)),
        'f1': float(f1_score(y_test, preds)),
        'roc_auc': float(roc_auc_score(y_test, probs)) if probs is not None else None
    }
    with open(ARTIFACTS_DIR / f'{name}_metrics.json', 'w') as f:
        json.dump(metrics, f)
    logger.info(f"{name} metrics: {metrics}")
    return metrics

# flow principal
@flow(name="bank-marketing-full-flow")
def main_flow():
    logger = get_run_logger()
    logger.info("inicio del flujo principal")

    df = load_data()
    X_train, X_test, y_train, y_test, preproc = prepare_data(df)

    dt, dt_metrics = baseline_decision_tree(X_train, y_train, X_test, y_test)

    nn_best, nn_score, nn_info = tune_neural_network(X_train, y_train)
    rf_best, rf_score, rf_params = tune_random_forest(X_train, y_train)
    xgb_best, xgb_score, xgb_params = tune_xgboost(X_train, y_train)

    # si los estimadores son scikeras wrappers (nn_best) y no tienen predict_proba, evaluate_model intentará manejarlo
    nn_metrics = evaluate_model(nn_best, X_test, y_test, 'nn') if nn_best is not None else None
    rf_metrics = evaluate_model(rf_best, X_test, y_test, 'rf')
    xgb_metrics = evaluate_model(xgb_best, X_test, y_test, 'xgb')

    summary = {
        'baseline_dt': dt_metrics,
        'neural_network': {'metrics': nn_metrics, 'cv_score': nn_score, 'best_info': nn_info},
        'random_forest': {'metrics': rf_metrics, 'cv_score': rf_score, 'best_params': rf_params},
        'xgboost': {'metrics': xgb_metrics, 'cv_score': xgb_score, 'best_params': xgb_params}
    }
    with open(ARTIFACTS_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("flujo finalizado, artefactos guardados en artifacts/")
    return summary

if __name__ == "__main__":
    main_flow(return_state=True)
