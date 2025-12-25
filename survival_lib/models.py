import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

from .stats import benjamini_hochberg


def cv_cox(z, durations, events, n_splits, random_state, penalizer, l1_ratio=0.0):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    c_indexes = []
    for train_idx, test_idx in kf.split(z):
        z_train = z.iloc[train_idx]
        z_test = z.iloc[test_idx]
        df_train = z_train.copy()
        df_train["duration"] = durations.iloc[train_idx].values
        df_train["event"] = events.iloc[train_idx].values

        model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        model.fit(df_train, duration_col="duration", event_col="event")

        risk = model.predict_partial_hazard(z_test)
        c_index = concordance_index(
            durations.iloc[test_idx].values,
            -risk.values.ravel(),
            events.iloc[test_idx].values,
        )
        c_indexes.append(c_index)
    return c_indexes


def univariate_cox(x, durations, events):
    results = []
    for gene in x.columns:
        df = pd.DataFrame(
            {
                "duration": durations.values,
                "event": events.values,
                "gene": x[gene].values,
            }
        )
        try:
            model = CoxPHFitter()
            model.fit(df, duration_col="duration", event_col="event")
            summary = model.summary.loc["gene"]
            results.append(
                {
                    "gene": gene,
                    "coef": summary["coef"],
                    "exp(coef)": summary["exp(coef)"],
                    "p": summary["p"],
                }
            )
        except Exception:
            results.append(
                {"gene": gene, "coef": np.nan, "exp(coef)": np.nan, "p": np.nan}
            )
    res = pd.DataFrame(results).dropna(subset=["p"])
    if not res.empty:
        res["q"] = benjamini_hochberg(res["p"].values)
        res = res.sort_values("p")
    return res


def fit_elastic_net_cox(x, durations, events, penalizer, l1_ratio):
    df = x.copy()
    df["duration"] = durations.values
    df["event"] = events.values
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    model.fit(df, duration_col="duration", event_col="event")
    summary = model.summary.reset_index()
    if "covariate" in summary.columns:
        summary = summary.rename(columns={"covariate": "gene"})
    elif "index" in summary.columns:
        summary = summary.rename(columns={"index": "gene"})
    return summary
