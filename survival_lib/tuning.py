import numpy as np
import pandas as pd

from .models import cv_cox


def tune_elastic_net(
    x, durations, events, penalizers, l1_ratios, n_splits, random_state
):
    rows = []
    for penalizer in penalizers:
        for l1_ratio in l1_ratios:
            c_indexes = cv_cox(
                x,
                durations,
                events,
                n_splits,
                random_state,
                penalizer,
                l1_ratio,
            )
            rows.append(
                {
                    "penalizer": penalizer,
                    "l1_ratio": l1_ratio,
                    "c_index_mean": float(np.mean(c_indexes)),
                    "c_index_std": float(np.std(c_indexes)),
                }
            )
    result = pd.DataFrame(rows).sort_values("c_index_mean", ascending=False)
    return result
