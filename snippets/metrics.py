from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

BOLD = '\033[1m'
ITALIC = '\033[3m'
UNDERLINE = '\033[4m'
END = '\033[0m'


# print(f"{BOLD}Это жирный текст{END}")
# print(f"{ITALIC}Это курсив{END}")
# print(f"{UNDERLINE}Это подчёркнутый текст{END}")


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    nonzero = denominator != 0
    return np.mean(np.abs(y_pred[nonzero] - y_true[nonzero]) / denominator[nonzero]) * 100


def mean_squared_log_error(y_true, y_pred):
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)


def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


def adjusted_r2_score(y_true, y_pred, n_features):
    from sklearn.metrics import r2_score
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)


def median_absolute_error(y_true, y_pred):
    return np.median(np.abs(np.array(y_true) - np.array(y_pred)))


"""
Основная функция для выводв всех метрик
"""


def calc_metrics(y_true, y_pred, n_features):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    adj_r2 = adjusted_r2_score(y_true, y_pred, n_features)
    medAE = median_absolute_error(y_true, y_pred)

    return mae, mse, rmse, r2, mape, smape, msle, adj_r2, medAE


def base_metrics_report(y_true, y_pred, n_features):
    mae, mse, rmse, r2, mape, smape, msle, adj_r2, medAE = calc_metrics(y_true, y_pred, n_features)

    print("====== Результаты модели ======")
    print(f"{BOLD}MAE: {END}  {mae:.3f}")
    print(f"{BOLD}MSE: {END}  {mse:.3f}")
    print(f"{BOLD}RMSE: {END}  {rmse:.3f}")
    print(f"{BOLD}R²: {END}  {r2:.3f}")
    print(f"{BOLD}MAPE: {END}  {mape:.3f}")
    print(f"{BOLD}SMAPE: {END}  {smape:.3f}")
    print(f"{BOLD}MSLE: {END}  {msle:.3f}")
    print(f"{BOLD}Adjusted R²: {END}  {adj_r2:.3f}")
    print(f"{BOLD}MedAE: {END}  {medAE:.3f}")


def base_metrics_report_pandas_df(y_true, y_pred, n_features):
    mae, mse, rmse, r2, mape, smape, msle, adj_r2, medAE = calc_metrics(y_true, y_pred, n_features)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'Adjusted R²': adj_r2,
        'MedAE': medAE,
        'MAPE': mape,
        'SMAPE': smape,
        'MSLE': msle,
    }

    return pd.DataFrame(metrics, index=["Model X"]).T


def base_metrics_multi_models_report_pandas_df(y_true, y_preds, n_features, models_names):
    metrics = {
        "MAE": [mean_absolute_error(y_true, y_pred) for y_pred in y_preds],
        "MSE": [mean_squared_error(y_true, y_pred) for y_pred in y_preds],
        "RMSE": [np.sqrt(mean_squared_error(y_true, y_pred)) for y_pred in y_preds],
        "R²": [r2_score(y_true, y_pred) for y_pred in y_preds],
        "Adjusted R²": [adjusted_r2_score(y_true, y_pred, n_features) for y_pred in y_preds],
        "MedAE": [median_absolute_error(y_true, y_pred) for y_pred in y_preds],
        "MAPE": [mean_absolute_percentage_error(y_true, y_pred) for y_pred in y_preds],
        "SMAPE": [symmetric_mean_absolute_percentage_error(y_true, y_pred) for y_pred in y_preds],
        "MSLE": [mean_squared_log_error(y_true, y_pred) for y_pred in y_preds]
    }

    df_metrics = pd.DataFrame(metrics, index=models_names)  # ['Model 1: Linear', 'Model 2: Ridge', 'Model 3: XGBoost']
    return df_metrics.round(3)

# display(df_metrics)
