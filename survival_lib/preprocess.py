import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def qc_and_scale(expr, max_missing, top_genes):
    expr = expr.dropna(axis=0, how="all")
    missing_frac = expr.isna().mean(axis=1)
    expr = expr.loc[missing_frac <= max_missing]
    expr = expr.fillna(expr.median(axis=1))
    expr = expr.dropna(axis=0, how="any")

    variances = expr.var(axis=1)
    if top_genes is not None and top_genes < len(variances):
        keep = variances.nlargest(top_genes).index
        expr = expr.loc[keep]

    x = expr.T
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.values)
    x_scaled = pd.DataFrame(x_scaled, index=x.index, columns=x.columns)
    return x_scaled


def fit_pca(x, n_components, random_state):
    pca = PCA(n_components=n_components, random_state=random_state)
    z = pca.fit_transform(x.values)
    z = pd.DataFrame(z, index=x.index, columns=[f"PC{i+1}" for i in range(z.shape[1])])
    return pca, z
