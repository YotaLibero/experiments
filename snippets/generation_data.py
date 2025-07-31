import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.special import expit  # сигмоида для логистической регрессии

# ----------------------------------------
# Генерация числовых признаков
# ----------------------------------------
def generate_features(n_samples=100, n_features=1, distribution='normal'):
    dist_map = {
        'normal': lambda: np.random.normal(0, 1, (n_samples, n_features)),
        'binomial': lambda: np.random.binomial(n=1, p=0.5, size=(n_samples, n_features)),
        'poisson': lambda: np.random.poisson(3.0, (n_samples, n_features)),
        'exponential': lambda: np.random.exponential(scale=1.0, size=(n_samples, n_features)),
        'uniform': lambda: np.random.uniform(-1, 1, (n_samples, n_features)),
        'cauchy': lambda: np.random.standard_cauchy((n_samples, n_features))
    }
    return dist_map[distribution]()

# ----------------------------------------
# Добавление шума
# ----------------------------------------
def add_noise(y, noise_type='normal', scale=1.0):
    if noise_type == 'none':
        return y
    elif noise_type == 'normal':
        return y + np.random.normal(0, scale, size=y.shape)
    elif noise_type == 'uniform':
        return y + np.random.uniform(-scale, scale, size=y.shape)
    elif noise_type == 'poisson':
        return y + np.random.poisson(scale, size=y.shape)
    else:
        raise ValueError("Неподдерживаемый тип шума")

# ----------------------------------------
# Генерация отклика (target)
# ----------------------------------------
def generate_regression(X, regression_type='linear', coef=None):
    if coef is None:
        coef = np.random.uniform(-3, 3, size=(X.shape[1],))
    
    if regression_type == 'linear':
        y = X @ coef
    elif regression_type == 'polynomial':
        y = X @ coef + (X**2) @ (coef / 2)
    elif regression_type == 'logistic':
        y_prob = expit(X @ coef)
        y = (y_prob > 0.5).astype(int)
    elif regression_type == 'multiple':
        y = X @ coef + np.random.normal(0, 1, size=(X.shape[0],))
    else:
        raise ValueError("Неподдерживаемый тип регрессии")
    
    return y

# ----------------------------------------
# Добавление пропусков
# ----------------------------------------
def add_missing_values(X, missing_rate=0.1):
    X_missing = X.copy()
    n_missing = int(np.prod(X.shape) * missing_rate)
    indices = np.unravel_index(np.random.choice(X.size, n_missing, replace=False), X.shape)
    X_missing[indices] = np.nan
    return X_missing

# ----------------------------------------
# Добавление категориальных признаков
# ----------------------------------------
def add_categorical_features(X, n_cat_features=1, n_categories=3):
    cats = np.random.randint(0, n_categories, size=(X.shape[0], n_cat_features))
    encoder = OneHotEncoder(sparse_output=False)
    cat_encoded = encoder.fit_transform(cats)
    return np.hstack([X, cat_encoded])

# ----------------------------------------
# Добавление бинарных признаков
# ----------------------------------------
def add_binary_features(X, n_bin_features=1):
    binary = np.random.randint(0, 2, size=(X.shape[0], n_bin_features))
    return np.hstack([X, binary])

# ----------------------------------------
# Общая функция генерации датасета
# ----------------------------------------
def generate_dataset(
    n_samples=100,
    n_features=2,
    regression_type='linear',
    noise_type='normal',
    distribution='normal',
    add_missing=False,
    missing_rate=0.1,
    add_categorical=False,
    n_cat_features=1,
    add_binary=False,
    n_bin_features=1
):
    """
    n_samples :             Кол-во объектов
    n_features :            Число численных признаков
    regression_type	:       Тип зависимости linear, polynomial, logistic, multiple
    noise_type :            Тип шума: none, normal, uniform, poisson
    noise_level :           Уровень шума
    distribution :          Распределение признаков
    include_missing :       Добавлять ли пропуски
    missing_rate :          Доля пропусков
    include_categorical :   Добавлять ли категориальные признаки
    include_binary:         Добавлять ли бинарный признак
    random_state :          Фиксированное зерно генератора случайностей
    """
    X = generate_features(n_samples, n_features, distribution)

    if add_categorical:
        X = add_categorical_features(X, n_cat_features)

    if add_binary:
        X = add_binary_features(X, n_bin_features)

    y = generate_regression(X, regression_type)
    y = add_noise(y, noise_type)

    if add_missing:
        X = add_missing_values(X, missing_rate)

    X_df = pd.DataFrame(X, columns=[f'X{i+1}' for i in range(X.shape[1])])
    y_series = pd.Series(y, name='target')
    return X_df, y_series
