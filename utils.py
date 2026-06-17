"""Общие функции для всех ноутбуков курсовой работы.

Здесь собраны вспомогательные функции, которые повторялись из ноутбука в
ноутбук: загрузка датасета и единые функции оценки качества для регрессии и
классификации. Вынес их сюда, чтобы не дублировать код (рекомендация по
улучшению из аналитического отчёта, раздел про качество кода).

Использование:
    from utils import load_dataset, evaluate_reg, evaluate_clf
"""

import os
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

DATA_FILENAME = "Данные_для_курсовои_Классическое_МО.xlsx"
TARGET_COLS = ["IC50, mM", "CC50, mM", "SI"]


def load_dataset(filename=DATA_FILENAME):
    """Найти и загрузить датасет, убрать служебный индексный столбец.

    Файл ищется сначала в текущей директории, затем в домашней папке.
    """
    candidates = [filename, os.path.join(pathlib.Path.home(), filename)]
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        raise FileNotFoundError(
            f"Положите '{filename}' рядом с ноутбуками или в домашнюю папку."
        )
    df = pd.read_excel(path)
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def evaluate_reg(name, model, x_tr, x_te, y_tr, y_te, cv=5):
    """Обучить регрессор и посчитать метрики на log- и исходной шкале.

    Цель обучается в log1p, метрики RMSE/MAE приводятся обратно через expm1.
    Возвращает словарь с метриками и обученной моделью.
    """
    model.fit(x_tr, y_tr)
    pred_log = model.predict(x_te)
    r2_log = r2_score(y_te, pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_te, pred_log))

    pred_orig = np.expm1(pred_log)
    y_te_orig = np.expm1(y_te)
    rmse_orig = np.sqrt(mean_squared_error(y_te_orig, pred_orig))
    mae_orig = mean_absolute_error(y_te_orig, pred_orig)

    cv_r2 = cross_val_score(model, x_tr, y_tr, cv=cv, scoring="r2", n_jobs=-1).mean()

    print(
        f"{name:<38}  R2(log)={r2_log:.4f}  RMSE(log)={rmse_log:.4f}  "
        f"RMSE(orig)={rmse_orig:.2f}  MAE(orig)={mae_orig:.2f}  CV-R2={cv_r2:.4f}"
    )
    return {
        "model": name,
        "R2_log": r2_log,
        "RMSE_log": rmse_log,
        "RMSE_orig": rmse_orig,
        "MAE_orig": mae_orig,
        "CV_R2": cv_r2,
        "fitted": model,
    }


def evaluate_clf(name, model, x_tr, x_te, y_tr, y_te, cv_x=None, cv=5, pos_label=None):
    """Обучить классификатор и посчитать Accuracy, F1, ROC-AUC и CV-AUC.

    Для задач с дисбалансом классов передайте ``pos_label=1``, чтобы F1
    считался по миноритарному классу, а не как weighted-среднее.
    """
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    prob = model.predict_proba(x_te)[:, 1]

    acc = accuracy_score(y_te, pred)
    if pos_label is None:
        f1 = f1_score(y_te, pred, average="weighted")
    else:
        f1 = f1_score(y_te, pred, pos_label=pos_label)
    auc = roc_auc_score(y_te, prob)

    cv_data = cv_x if cv_x is not None else x_tr
    cv_auc = cross_val_score(
        model, cv_data, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1
    ).mean()

    print(
        f"{name:<38}  Acc={acc:.4f}  F1={f1:.4f}  "
        f"AUC={auc:.4f}  CV-AUC={cv_auc:.4f}"
    )
    return {
        "model": name,
        "Accuracy": acc,
        "F1": f1,
        "ROC_AUC": auc,
        "CV_AUC": cv_auc,
        "fitted": model,
        "prob": prob,
    }
