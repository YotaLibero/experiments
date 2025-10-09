# feature_analysis_industrial.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(Path("../config.py").resolve()), "../../..")))

from config import PATH_DATASET_FOR_ANALYSE, PATH_DATASET_WITH_NAN

sns.set(style="whitegrid")

def load_data(path):
    df = pd.read_csv(path)
    # ensure Fault_Type is present
    if 'Fault_Type' not in df.columns:
        raise ValueError("CSV must contain 'Fault_Type' column")
    return df

def compute_corr_matrix(df, cols=None, method='pearson'):
    if cols is None:
        cols = [c for c in df.columns if c != 'Fault_Type']
    corr = df[cols].corr(method=method)
    return corr

def plot_clustered_corr(corr, title='Clustered correlation heatmap', outpath=None):
    # distance for clustering: 1 - |corr|
    dist = 1 - np.abs(corr.values)
    # hierarchical clustering
    link = linkage(dist, method='average')
    order = leaves_list(link)
    ordered_cols = corr.columns[order]
    corr_ord = corr.loc[ordered_cols, ordered_cols]

    plt.figure(figsize=(12,10))
    im = plt.imshow(corr_ord.values, vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, fraction=0.03)
    plt.xticks(range(len(ordered_cols)), ordered_cols, rotation=90)
    plt.yticks(range(len(ordered_cols)), ordered_cols)
    plt.title(title)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()
    return ordered_cols

def mutual_info_vs_target(df, target='Fault_Type', n_top=20, outpath=None):
    X = df.drop(columns=[target])
    # impute missing with median
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    y = LabelEncoder().fit_transform(df[target].astype(str))
    mi = mutual_info_classif(X_imp, y, discrete_features=False, random_state=0)
    mi_ser = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    mi_ser.head(n_top).plot.bar()
    plt.title('Mutual Information with Fault_Type (top {})'.format(n_top))
    plt.ylabel('MI')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()
    return mi_ser

def physical_vs_fft_corr(df, physical, fft_prefix, outpath=None):
    # collect fft columns that start with fft_prefix
    fft_cols = [c for c in df.columns if c.startswith(fft_prefix)]
    res = {}
    for c in fft_cols:
        # pairwise drop NaNs
        sub = df[[physical, c]].dropna()
        if len(sub) < 3:
            corr = np.nan
        else:
            corr = sub[physical].corr(sub[c])
        res[c] = corr
    res_ser = pd.Series(res).sort_values(key=lambda s: np.abs(s), ascending=False)
    plt.figure(figsize=(8,4))
    res_ser.plot.bar()
    plt.title(f'Correlation between {physical} and {fft_prefix}* bins')
    plt.ylabel('Pearson r')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()
    return res_ser

def pca_loadings(df, n_components=3, outpath=None):
    X = df.drop(columns=['Fault_Type'])
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_imp)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns,
                            columns=[f'PC{i+1}' for i in range(n_components)])
    plt.figure(figsize=(10,6))
    sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
    plt.title('PCA loadings')
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()
    return loadings

def pairplot_top_features(df, features, target='Fault_Type', sample_frac=0.2, outpath=None):
    # sample if too big
    df_sub = df[[*features, target]].sample(frac=sample_frac, random_state=0) if sample_frac < 1.0 else df[[*features, target]]
    sns.pairplot(df_sub, hue=target, corner=True, plot_kws={'alpha':0.6, 's':20})
    plt.suptitle('Pairplot for top features', y=1.02)
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()

def rf_feature_importance(df, target='Fault_Type', n_top=20, outpath=None):
    X = df.drop(columns=[target])
    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    y = LabelEncoder().fit_transform(df[target].astype(str))
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(X_imp, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8,6))
    importances.head(n_top).plot.bar()
    plt.title('RandomForest feature importances (top {})'.format(n_top))
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=200)
    plt.show()
    # permutation importance (more robust)
    perm = permutation_importance(rf, X_imp, y, n_repeats=10, random_state=0, n_jobs=-1)
    perm_ser = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    return importances, perm_ser

if __name__ == '__main__':
    csv_path = PATH_DATASET_FOR_ANALYSE # <- put your path here
    df = load_data(csv_path)
    print('Loaded', df.shape)

    # 1. correlation heatmap (clustered)
    corr = compute_corr_matrix(df)
    ordered = plot_clustered_corr(corr, title='Clustered correlation (all features)')

    # 2. mutual info vs Fault_Type
    mi_ser = mutual_info_vs_target(df, n_top=30)

    # 3. physical vs FFT correlations (examples)
    vib_vs_fft = physical_vs_fft_corr(df, 'Vibration', 'FFT_Vib_')
    temp_vs_fft = physical_vs_fft_corr(df, 'Temperature', 'FFT_Temp_')
    pres_vs_fft = physical_vs_fft_corr(df, 'Pressure', 'FFT_Pres_')

    # 4. PCA loadings
    loadings = pca_loadings(df, n_components=3)

    # 5. pairplot for top 6 features by MI
    top_feats = list(mi_ser.index[:6])
    pairplot_top_features(df, top_feats, sample_frac=0.3)

    # 6. RF importance + permutation
    imp_rf, imp_perm = rf_feature_importance(df)
    print('Top RF features:', imp_rf.head(10))
    print('Top permutation features:', imp_perm.head(10))
