# нужно отредактировать так, чтобы было универсально и не использовались признаки с работы

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import pickle
from xgboost import XGBRegressor
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import xgboost as xgb
from statsmodels.tsa.seasonal import STL
from sklearn.feature_selection import VarianceThreshold
import copy

def add_clean_anomaly_features(df, columns, window):
    sigma=2 # 1.7
    min_std = 4.4
    new_df=df.copy()
    anomalies_set = []
    for name in columns:
        new_df[f'{name}_clean'] = new_df[name].copy()
        new_df[f'{name}_anomaly'] = 0
        anomalies_set.append(f'{name}_anomaly')
        df_part = new_df.dropna(subset=name)

        rolling_mean = df_part[name].rolling(window=window, center=True).mean()
        rolling_std = df_part[name].rolling(window=window, center=True).std()
        rolling_std = rolling_std.clip(lower=min_std)

        lower_bound = rolling_mean - sigma * rolling_std # 1.7
        upper_bound = rolling_mean + sigma * rolling_std # 1.7

        outliers = (df_part[name] < lower_bound) | (df_part[name] > upper_bound)
        new_df.loc[df_part[outliers].index, f'{name}_anomaly'] = 1
        new_df.loc[df_part[outliers].index, f'{name}_clean'] = rolling_mean

    print(new_df.shape)
    new_df['anomalies'] = new_df.loc[:, anomalies_set].sum(axis=1)
    new_df.loc[new_df['anomalies'] > 0, 'anomalies'] = 1
    new_df.drop(anomalies_set, axis=1, inplace=True)    return new_df

def add_fill_features(df, features):
    new_df=df.copy()
    for name in features:
        new_df[f'{name}_fill'] = new_df[f'{name}'].fillna(method='ffill').fillna(method='bfill')
    return new_df

def add_lag_features(df, features, lags):
    new_df, result_df = df.copy(), df.copy()
    cols_for_result = []
    # ЛАГИ
    # lags = [1, 2, 3, 4, 5, 6, 7, 8]
    for name in features:
        for lag in lags:
            new_df[f'{name}_lag_{int(lag*3)}h'] = new_df[name].copy()
            df_part = new_df.dropna(subset=name)
            new_df.loc[df_part.index, f'{name}_lag_{int(lag*3)}h'] = df_part.loc[:, f'{name}_lag_{int(lag*3)}h'].shift(lag)
            new_df[f'{name}_lag_{int(lag*3)}h'] = new_df[f'{name}_lag_{int(lag*3)}h'].fillna(method='ffill').fillna(method='bfill')
    y_names = ['POT_TEMP_clean_resid', 'POT_TEMP_resid', 'POT_TEMP_trend']
    for y_name in y_names:
        for name in features:
            cols_part = []
            cols_part.extend(new_df.loc[:, new_df.columns.str.contains(f'{name}_lag_')].columns)
            cols_part.extend([y_name])

            corr_df = new_df.loc[:, cols_part].corr()
            list_corr_df = corr_df.loc[(np.abs(corr_df[y_name]) > 0.25) & (np.abs(corr_df[y_name]) < 0.8), y_name]
            sorted_correlations = list_corr_df.sort_values(ascending=False)
            if not sorted_correlations.empty:
                cols_for_result.append(sorted_correlations.index[0])
        if cols_for_result != []:
            for col in cols_for_result:
                result_df[col] = new_df[col].copy()
    return result_df

def add_diff_features(df, features, periods):
    new_df, result_df = df.copy(), df.copy()
    cols_for_result = []
    # periods = [1, 2, 3, 4, 5, 6, 7, 8]
    for name in features:
        for period in periods:
            new_df[f'{name}_diff_period_{period}'] = new_df[name].diff(periods=period)
    # оставить только значимые признаки
    y_names = ['POT_TEMP_clean_resid', 'POT_TEMP_resid', 'POT_TEMP_trend']
    for y_name in y_names:
        for name in features:
            cols_part = []
            cols_part.extend(new_df.loc[:, new_df.columns.str.contains(f'{name}_diff_period_')].columns)
            cols_part.extend([y_name])

            corr_df = new_df.loc[:, cols_part].corr()
            list_corr_df = corr_df.loc[(np.abs(corr_df[y_name]) > 0.25) & (np.abs(corr_df[y_name]) < 0.8), y_name]
            sorted_correlations = list_corr_df.sort_values(ascending=False)
            if not sorted_correlations.empty:
                cols_for_result.append(sorted_correlations.index[0])
        if cols_for_result != []:
            for col in cols_for_result:
                result_df[col] = new_df[col].copy()
    return result_df

def add_feature_diff_trend(df, features, features_trend):
    new_df=df.copy()
    # features и features_dop должны быть одной длины
    for i, name in enumerate(features):
        new_df[f'{name}_diff_{features_trend[i]}'] = new_df[name] - new_df[features_trend[i]]
    # оставить только значимые признаки
    return new_df

def add_rolling_statistics_features(df, features, windows):
    new_df, result_df = df.copy(), df.copy()
    # СКОЛЬЗЯЩИЕ СТАТИСТИКИ
    # windows = [1, 2, 3, 4, 5, 6, 7, 8]
    # windows = [4, 8]
    cols_for_result = []
    for name in features[:len(features)]:
        for window in windows:
            new_df[f'{name}_rolling_avg_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).mean()
            new_df[f'{name}_rolling_std_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).std()
            new_df[f'{name}_rolling_median_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).median()
            new_df[f'{name}_rolling_min_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).min()
            new_df[f'{name}_rolling_max_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).max()
            new_df[f'{name}_rolling_range_{int(window*3)}h'] = new_df[f'{name}_rolling_max_{int(window*3)}h'] - new_df[f'{name}_rolling_min_{int(window*3)}h']
            new_df[f'{name}_rolling_variance_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).var()
            new_df[f'{name}_rolling_skewness_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).skew()
            new_df[f'{name}_rolling_kurtosis_{int(window*3)}h'] = new_df[name].rolling(window=window, center=True).kurt()
        # убрать лишние признаки
        cols_part = list(new_df.loc[:, new_df.columns.str.contains(f'{name}_rolling_avg_')].columns)
        cols_part.extend(['POT_TEMP_trend'])

        corr_df = new_df.loc[:, cols_part].corr()
        list_corr_df = corr_df.loc[np.abs(corr_df['POT_TEMP_trend']) < 0.8, 'POT_TEMP_trend']
        sorted_correlations = list_corr_df.sort_values(ascending=False)
        if not sorted_correlations.empty:
            cols_for_result.append(sorted_correlations.index[0])

        strings = [f'{name}_rolling_std_', f'{name}_rolling_variance_', f'{name}_rolling_max_', f'{name}_rolling_min_', f'{name}_rolling_median_',
                   f'{name}_rolling_skewness_', f'{name}_rolling_kurtosis_', f'{name}_rolling_range_', f'{name}_rolling_avg_']
        for string in strings:
            cols_part = []
            cols_part.extend(new_df.loc[:, new_df.columns.str.contains(string)].columns)
            cols_part.extend(['POT_TEMP_clean_resid'])

            corr_df = new_df.loc[:, cols_part].corr()
            list_corr_df = corr_df.loc[np.abs(corr_df['POT_TEMP_clean_resid']) < 0.8, 'POT_TEMP_clean_resid']
            sorted_correlations = list_corr_df.sort_values(ascending=False)
            if not sorted_correlations.empty:
                cols_for_result.append(sorted_correlations.index[0])
        if cols_for_result != []:
            for col in cols_for_result:
                result_df[col] = new_df[col].copy()

        return result_df

        def add_ratio_features(df, features):
            new_df, result_df = df.copy(), df.copy()
            for feature_A in features:
                for feature_B in features:
                    if feature_A != feature_B:
                        new_df[f'{feature_A}_{feature_B}_ratio'] = new_df[feature_A] / new_df[feature_B]
                        new_df[f'{feature_A}_{feature_B}_multip'] = new_df[feature_A] * new_df[feature_B]
                        new_df[f'{feature_A}_{feature_B}__diff'] = new_df[feature_A] - new_df[feature_B]
                        new_df[f'{feature_A}_{feature_B}_sum'] = new_df[feature_A] + new_df[feature_B]

            # проверка значимости
            strings = ['_ratio', '_multip', '__diff', '_sum']
            y_names = ['POT_TEMP_clean_resid']
            for y_name in y_names:
                for name in features:
                    cols_part = []
                    cols_part.extend(new_df.loc[:, new_df.columns.str.contains(f'{name}_poly_')].columns)
                    cols_part.extend([y_name])

                    corr_df = new_df.loc[:, cols_part].corr()
                    list_corr_df = corr_df.loc[
                        (np.abs(corr_df[y_name]) > 0.35) & (np.abs(corr_df[y_name]) < 0.8), y_name]
                    sorted_correlations = list_corr_df.sort_values(ascending=False)
                    if not sorted_correlations.empty:
                        if len(cols_for_result) > 3:
                            cols_for_result.append(sorted_correlations.index[:3])
                        else:
                            cols_for_result.append(sorted_correlations.index)
                if cols_for_result != []:
                    for col in cols_for_result:
                        result_df[col] = new_df[col].copy()
            return result_df

        # если распределение признака скошено вправо
        def add_log_features(df, features):
            new_df = df.copy()
            for name in features:
                new_df[f'{name}log'] = np.log(new_df[name] + 1)
            return new_df

        def add_one_per_features(df, features):
            new_df, result_df = df.copy(), df.copy()
            cols_for_result = []
            for name in features:
                new_df[f'{name}_one_per'] = 1 / (new_df[name] + 1e-8)
            # проверка значимости
            strings = ['_one_per']
            for string in strings:
                cols_part = []
                cols_part.extend(new_df.loc[:, new_df.columns.str.contains(string)].columns)
                cols_part.extend(['POT_TEMP_clean_resid'])

                corr_df = new_df.loc[:, cols_part].corr()
                list_corr_df = corr_df.loc[(np.abs(corr_df['POT_TEMP_clean_resid']) > 0.25) & (
                            np.abs(corr_df['POT_TEMP_clean_resid']) < 0.8), 'POT_TEMP_clean_resid']
                sorted_correlations = list_corr_df.sort_values(ascending=False)
                if not sorted_correlations.empty:
                    cols_for_result.append(sorted_correlations.index[0])
            if cols_for_result != []:
                for col in cols_for_result:
                    result_df[col] = new_df[col].copy()
            return result_df

        # на попозже
        def add_distance_between_features(df, features):
            pass

        def add_poly_features(df, features):
            new_df, result_df = df.copy(), df.copy()
            cols_for_result = []
            for name in features:
                new_df[f'{name}_poly_2'] = new_df[name] ** 2
                new_df[f'{name}_poly_3'] = new_df[name] ** 3
                new_df[f'{name}_poly_4'] = new_df[name] ** 4
                new_df[f'{name}_poly_5'] = new_df[name] ** 5
                new_df[f'{name}_poly_0.5'] = new_df[name] ** 0.5
                new_df[f'{name}_poly_1/3'] = new_df[name] ** (1 / 3)
                new_df[f'{name}_poly_0.25'] = new_df[name] ** 0.25
                new_df[f'{name}_poly_0.25'] = new_df[name] ** (1 / 5)
            # проверка значимости
            y_names = ['POT_TEMP_clean_resid', 'POT_TEMP_trend']
            for y_name in y_names:
                for name in features:
                    cols_part = []
                    cols_part.extend(new_df.loc[:, new_df.columns.str.contains(f'{name}_poly_')].columns)
                    cols_part.extend([y_name])

            corr_df = new_df.loc[:, cols_part].corr()
            list_corr_df = corr_df.loc[(np.abs(corr_df[y_name]) > 0.35) & (np.abs(corr_df[y_name]) < 0.8), y_name]
            sorted_correlations = list_corr_df.sort_values(ascending=False)
            if not sorted_correlations.empty:
                if len(cols_for_result) > 3:
                    cols_for_result.append(sorted_correlations.index[:3])
                else:
                    cols_for_result.append(sorted_correlations.index)
        if cols_for_result != []:
            for col in cols_for_result:
                result_df[col] = new_df[col].copy()
    return result_df

def add_trend_features(df, features, window):
    new_df=df.copy()
    # ТРЕНДЫ
    new_df = new_df.dropna(subset=features)
    new_df['DATE_TIME_y'] = new_df['DATE_TIME'].copy()
    new_df.set_index('DATE_TIME_y', inplace=True)
    for col in features:
        new_df[f'{col}_trend'] = STL(new_df[col], period=56).fit().trend
        new_df = new_df.dropna(subset=f'{col}_trend')
    new_df.index = range(len(new_df))
    return new_df

def add_time_epoches_features(df):
    new_df=df.copy()
    new_df['time_delta'] = new_df['DATE_TIME'].diff().dt.total_seconds()
    new_df['time_delta_log'] = np.log1p(new_df['time_delta'])
    new_df['is_gap'] = new_df['DATE_TIME'].diff().dt.total_seconds() > 604800
    new_df['epoch_id'] = new_df['is_gap'].cumsum().fillna(0).astype(int)
    return new_df

def clean_features_with_nan(df):
    new_df=df.copy()
    drop_cols = []
    for col in new_df.columns:
        null_count = new_df[col].isnull().sum()
        if null_count > 0:
            drop_cols.append(col)
    new_df.drop(drop_cols, axis=1, inplace=True)
    print(f'после удаления столбцов с пропусками: {new_df.shape}')
    print(f'удалённые признаки: {drop_cols}')
    return new_df

def add_coding_category_features(df, features):
    # features = ['FEED_CNT_POINT_PROBLEM', 'INNER_KALF4_WEIGHT', 'INNER_POT_TAP', 'BOARDS_CNT_POINTS_TEMP', 'epoch_id', 'anomalies']
    new_df = pd.get_dummies(df, columns=features)
    print(new_df.iloc[:, -50:].columns)
    return new_df

def adding_new_features(df):
    pot=df.copy()

    # ГИПОТЕЗА 1: без лаборатории
    lab_columns = ['LAB_KF', 'LAB_CAF2', 'LAB_CR', 'LAB_SI', 'LAB_SI_ME', 'LAB_S', 'LAB_FE']
    pot.drop(lab_columns, axis=1, inplace=True)

    # смысловые признаки
    pot['TOTAL_WEIGHT_FLUOR'] = pot['INNER_FLUOR_WEIGHT'] * pot['INNER_FLUOR_DUMPS']
    pot['R'] = pot['INNER_POT_U'] / pot['POT_I1']
    pot['P'] = pot['INNER_POT_U'] * pot['POT_I1']
    pot['efficienty'] = (pot['FEED_VALUE'] * pot['TEST_M_FACTOR2']) / pot['P']

    # оригинальные переменные

    columns = ['POT_TEMP', 'DU_median', 'TEST_M_FACTOR2', 'FEED_VALUE', 'FEED_SUM_DUMPS', 'POT_I1', 'FEED_WEIGHT_FEED', 'FEED_TOTAL_WEIGHT',
               'BOARDS_MEAN_TEMP', 'INNER_POT_U', 'MEAN_LOW_FREQ_POW_A', 'INNER_FLUOR_WEIGHT', 'INNER_FLUOR_DUMPS', 'TOTAL_WEIGHT_FLUOR',
               'R', 'P', 'efficienty']
    pot = add_clean_anomaly_features(pot, columns, 8)

    pot = add_trend_features(pot, ['POT_TEMP_clean'], window=56)
    pot = add_feature_diff_trend(pot, ['POT_TEMP', 'POT_TEMP_clean'], ['POT_TEMP_clean_trend', 'POT_TEMP_clean_trend'])
    pot['POT_TEMP_trend'] = pot['POT_TEMP_clean_trend'].copy()
    pot['POT_TEMP_resid'] = pot['POT_TEMP_diff_POT_TEMP_clean_trend'].copy()
    pot['POT_TEMP_clean_resid'] = pot['POT_TEMP_clean_diff_POT_TEMP_clean_trend'].copy()

    # pot = add_fill_features(pot, fill_columns)
    pot = pot.dropna(subset=columns)

    original_columns = ['POT_TEMP', 'DU_median', 'TEST_M_FACTOR2', 'FEED_VALUE', 'FEED_SUM_DUMPS', 'POT_I1', 'FEED_WEIGHT_FEED', 'FEED_TOTAL_WEIGHT',
               'BOARDS_MEAN_TEMP', 'INNER_POT_U', 'MEAN_LOW_FREQ_POW_A', 'INNER_FLUOR_WEIGHT', 'INNER_FLUOR_DUMPS', 'TOTAL_WEIGHT_FLUOR',
               'R', 'P', 'efficienty',

               'DU_median_clean', 'TEST_M_FACTOR2_clean', 'FEED_VALUE_clean', 'FEED_SUM_DUMPS_clean', 'POT_I1_clean', 'FEED_WEIGHT_FEED_clean',
                        'FEED_TOTAL_WEIGHT_clean', 'BOARDS_MEAN_TEMP_clean', 'INNER_POT_U_clean',
                        'MEAN_LOW_FREQ_POW_A_clean', 'INNER_FLUOR_WEIGHT_clean',
                        'INNER_FLUOR_DUMPS_clean', 'TOTAL_WEIGHT_FLUOR_clean', 'R_clean', 'P_clean', 'efficienty_clean',

                        'DU_mean', 'DU_min', 'DU_max', 'DU_range', 'DU_std', 'DU_variance', 'DU_skewness',
                        'DU_kurtosis', 'SUM_DUMPS_mean', 'SUM_DUMPS_std',
                        'SUM_DUMPS_median', 'SUM_DUMPS_min', 'SUM_DUMPS_max', 'SUM_DUMPS_range', 'SUM_DUMPS_variance',
                        'SUM_DUMPS_skewness', 'SUM_DUMPS_kurtosis', 'DUMPS_mean', 'DUMPS_std', 'DUMPS_median',
                        'DUMPS_min', 'DUMPS_max', 'DUMPS_range',
                        'DUMPS_variance', 'DUMPS_skewness', 'DUMPS_kurtosis', 'I1_mean', 'I1_std', 'I1_median',
                        'I1_min', 'I1_max', 'I1_range', 'I1_variance',
                        'I1_skewness', 'I1_kurtosis', 'MEAN_LOW_FREQ_POW_A_mean', 'MEAN_LOW_FREQ_POW_A_std',
                        'MEAN_LOW_FREQ_POW_A_median', 'MEAN_LOW_FREQ_POW_A_min',
                        'MEAN_LOW_FREQ_POW_A_max', 'MEAN_LOW_FREQ_POW_A_range', 'MEAN_LOW_FREQ_POW_A_variance',
                        'MEAN_LOW_FREQ_POW_A_skewness',
                        'MEAN_LOW_FREQ_POW_A_kurtosis', 'U_mean', 'U_std', 'U_median', 'U_min', 'U_max', 'U_range',
                        'U_variance', 'U_skewness', 'U_kurtosis']

    # полиномиальные и обратные признаки
    pot = add_poly_features(pot, original_columns)
    pot = add_one_per_features(pot, original_columns)

    pot = add_rolling_statistics_features(pot, original_columns, [1, 2, 4, 8, 16, 24, 56])

    pot = add_diff_features(pot, original_columns, periods=[1, 2, 4, 8, 16, 24, 56])

    full_columns = copy.deepcopy(original_columns)
    strings = ['_ratio', '_multip', '__diff', '_sum', '_poly_', '_diff_period_', '_diff_', '_one_per']
    for string in strings:
        full_columns.extend(pot.loc[:, pot.columns.str.contains(string)].columns)

    pot = add_lag_features(pot, original_columns, lags=[1, 2, 4, 8, 16, 24, 56])
    print(f'full_columns after adding lags: {full_columns}')

    # pot = clean_features_with_nan(pot)

    # КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ: объединение некоторых категорий и кодирование
    pot['FEED_VALUE_clusters'] = pot['FEED_VALUE'].copy()
    pot.loc[pot['FEED_VALUE'] <= 0, 'FEED_VALUE_clusters'] = 0
    pot.loc[pot['FEED_VALUE'] > 0, 'FEED_VALUE_clusters'] = 1
    pot.loc[pot['INNER_KALF4_WEIGHT'] > 0, 'INNER_KALF4_WEIGHT'] = 1
    pot.loc[pot['INNER_POT_TAP'] > 0, 'INNER_POT_TAP'] = 1
    pot.loc[pot['FEED_CNT_POINT_PROBLEM'] > 0, 'FEED_CNT_POINT_PROBLEM'] = 1
    pot.loc[pot['BOARDS_CNT_POINTS_TEMP'] > 0, 'BOARDS_CNT_POINTS_TEMP'] = 1

    df_archive = pot.dropna(subset=full_columns)
    df_archive = add_time_epoches_features(df_archive)

    df_old = df_archive.copy()

    drop_cols = []
    strings = ['POT_TEMP', 'TEMPERATURE']
    for string in strings:
        drop_cols.extend(df_archive.loc[:, df_archive.columns.str.contains(string)].columns)
    df_archive.drop(drop_cols, axis=1, inplace=True)

    df_ = add_trend_features(df_archive, original_columns[1:], window=56)
    trend_columns = list(df_.loc[:, df_.columns.str.contains('_trend')].columns)
    print(f'original_columns: {original_columns}')
    print(f'trend_columns: {trend_columns}')
    df_ = add_feature_diff_trend(df_, original_columns[1:], trend_columns)

    df_ = pd.get_dummies(df_, columns=['FEED_CNT_POINT_PROBLEM', 'FEED_VALUE_clusters', 'INNER_KALF4_WEIGHT',
                                       'INNER_POT_TAP', 'BOARDS_CNT_POINTS_TEMP',
                                       'epoch_id', 'anomalies'])

    df_['POT_TEMP'] = df_old['POT_TEMP'].copy()
    df_['POT_TEMP_clean'] = df_old['POT_TEMP_clean'].copy()
    df_['POT_TEMP_trend'] = df_old['POT_TEMP_clean_trend'].copy()
    df_['POT_TEMP_resid'] = df_old['POT_TEMP_diff_POT_TEMP_clean_trend'].copy()
    df_['POT_TEMP_clean_resid'] = df_['POT_TEMP_clean'] - df_['POT_TEMP_trend']

    print(f'общее количество строк и колонок в датасете: {df_.shape}')
    return df_


def modelXGBoostRegressor(X, y, X_test, y_test):
    model = xgb.XGBRegressor()
    model.fit(X, y)

    res = np.array(model.predict(X_test.values))

    print(res.shape, y_test.shape)

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    mae = mean_absolute_error(y_test, res)
    r2 = r2_score(y_test, res)
    mse = mean_squared_error(y_test, res)
    rmse = np.sqrt(mse)
    print(f"Средний квадрат ошибки (MSE): {mse}")
    print(f"Корень среднего квадрата ошибки (RMSE): {rmse}")
    print(f'MAE: {mae:.5f}')
    print(f'R²: {r2:.5f}')
    return model, res

def forEachModelXGBoostRegressor(X, y, X_test, y_test):
    model = xgb.XGBRegressor()
    model.fit(X, y)

    res = []
    idxs = X_test.index
    cols = X_test.columns
    i = 0
    for idx in idxs:
        y_pred = model.predict(X_test.loc[idx, cols].values.reshape(1, -1))
        if i < len(X_test)-1:
            X_test.loc[:, 'POT_TEMP_PAST'][idx+1] = y_pred
        res.append(y_pred)
        i+=1

    res = np.array(res)

    print(res.shape, y_test.shape)

    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    mae = mean_absolute_error(y_test, res)
    r2 = r2_score(y_test, res)
    mse = mean_squared_error(y_test, res)
    rmse = np.sqrt(mse)
    print(f"Средний квадрат ошибки (MSE): {mse}")
    print(f"Корень среднего квадрата ошибки (RMSE): {rmse}")
    print(f'MAE: {mae:.5f}')
    print(f'R²: {r2:.5f}')
    return model, res

def prepare_dataset_to_model(df, y_name, train_features):
    sample1 = 0.98 # 0.9
    # sample1 = 0.97 # 0.9
    val_size = int(len(df) * sample1)

    df = df.dropna(subset=[y_name]).sort_values(by='DATE_TIME').reset_index(drop=True)

    df_train = df[:val_size]
    df_test = df[val_size:]

    dx_train = df_train[train_features].copy()
    dx_test = df_test[train_features].copy()

    scalerFeatures = StandardScaler()
    scalerTarget = MinMaxScaler()

    normalized_train = scalerFeatures.fit_transform(dx_train)
    df_normalized_train = pd.DataFrame(normalized_train, columns=dx_train.columns)
    df_normalized_train.index = dx_train.index

    normalized_test = scalerFeatures.transform(dx_test)
    df_normalized_test = pd.DataFrame(normalized_test, columns=dx_test.columns)
    df_normalized_test.index = dx_test.index

    train_out = scalerTarget.fit_transform(df_train[[y_name]]).astype('float32')
    test_out = scalerTarget.transform(df_test[[y_name]]).astype('float32')

    print(df_normalized_train.shape, df_normalized_test.shape)
    print(train_out.shape, test_out.shape)

    X, y = df_normalized_train.copy(), train_out.copy()
    X_test, y_test = df_normalized_test.copy(), test_out.copy()
    return X, y, X_test, y_test

def train_prediction_result_model(X, y, X_test, y_test, threshold_gain, threshold_weight):
    threshold=threshold_gain
    mdl, y_pred = modelXGBoostRegressor(X, y, X_test, y_test)
    y_test = np.array(y_test)
    plt.figure(figsize=(18, 5))
    plt.plot(y_test, label="Истинное y")
    plt.plot(y_pred, label="Прогноз")
    plt.xlabel("Время")
    plt.ylabel("y")
    plt.legend()
    plt.title("Предсказание температуры; данные стандартизированы")
    plt.show()

    # Получение важности признаков из модели
    feature_importances = mdl.get_booster().get_score(importance_type='gain')

    # Преобразование в DataFrame для удобства сортировки и отображения
    df_feature_importances = pd.DataFrame({
        'Feature': feature_importances.keys(),
        'Importance': feature_importances.values()
    })

    # Сортировка по важности в порядке убывания
    df_feature_importances = df_feature_importances.sort_values(by='Importance', ascending=False)
    # Вывод топ-20 признаков
    print("Top 20 features by importance (gain):")
    print(df_feature_importances.head(20))

    # Аналогично для 'weight' важности:
    feature_importances_weight = mdl.get_booster().get_score(importance_type='weight')
    df_feature_importances_weight = pd.DataFrame({
        'Feature': feature_importances_weight.keys(),
        'Importance (Weight)': feature_importances_weight.values()
    })
    df_feature_importances_weight = df_feature_importances_weight.sort_values(by='Importance (Weight)', ascending=False)
    print("\nTop 20 features by importance (weight):")
    print(df_feature_importances_weight.head(20))

    # Фильтрация признаков
    selected_features = df_feature_importances.loc[df_feature_importances['Importance'] < threshold, :]
    while (list(selected_features['Feature'].values) == []) & (threshold < 0.009):
        threshold += 0.001
        selected_features = df_feature_importances.loc[df_feature_importances['Importance'] < threshold, :]
        if list(selected_features['Feature'].values) != []:
            break
    print(len(selected_features['Feature'].values), selected_features)
    selected_features_weight = df_feature_importances_weight.loc[df_feature_importances_weight['Importance (Weight)'] > threshold_weight, :]
    print(selected_features_weight)

    if threshold >= 0.009:
        return y_pred, []

    return y_pred, list(selected_features['Feature'].values)

def exp_feature_importance(df, y_name, train_features, epoches):    X, y, X_test, y_test = prepare_dataset_to_model(df, y_name, train_features)
    threshold_gain, threshold_weight = 0.005, 50
    y_pred = []
    for epoch in range(epoches):
        print(f'epoch -- {epoch}')
        y_pred, drop_features = train_prediction_result_model(X, y, X_test, y_test, threshold_gain, threshold_weight)
        if drop_features == []:
            return X_test, y_test, y_pred, X.columns
        # X.drop(drop_features, axis=1, inplace=True)
        # X_test.drop(drop_features, axis=1, inplace=True)
        # sample1 = 0.97 # 0.99
        # train_size, val_size = int(len(df) * sample), int(len(df) * sample1)
        # dy_train, dy_test = df_train['POT_TEMP'].copy(), df_test['POT_TEMP'].copy()
        # print(f'dx_train: {dx_train.shape}, dx_test: {dx_test.shape},  dy_train: {dy_train.shape}, dy_test: {dy_test.shape}')
        # X, y, X_test, y_test = dx_train.copy(), dy_train.copy(), dx_test.copy(), dy_test.copy()
        return X_test, y_test, y_pred, X.columns

# def exp_