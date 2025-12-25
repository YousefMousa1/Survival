import numpy as np
import pandas as pd


def top_genes_by_component(pca, gene_ids, top_k):
    rows = []
    components = pca.components_
    for idx, comp in enumerate(components):
        series = pd.Series(comp, index=gene_ids)
        top = series.reindex(series.abs().sort_values(ascending=False).index).head(top_k)
        for gene, weight in top.items():
            rows.append({"component": f"PC{idx+1}", "gene": gene, "weight": weight})
    return pd.DataFrame(rows)


def build_combined_report(metrics, uni, enet_by_ratio):
    report = {"metrics": metrics, "univariate": {}, "elastic_net": {}}
    if uni is not None and not uni.empty:
        uni_sig = uni[uni["q"] < 0.05]
        report["univariate"] = {
            "n_tested": int(uni.shape[0]),
            "n_significant_q05": int(uni_sig.shape[0]),
            "top_by_p": uni.head(20).to_dict(orient="records"),
        }
    for ratio, enet in enet_by_ratio.items():
        if enet is None or enet.empty:
            continue
        enet = enet.copy()
        enet["abs_coef"] = enet["coef"].abs()
        tag = f"l1_ratio_{ratio:.2f}"
        report["elastic_net"][tag] = {
            "n_features": int(enet.shape[0]),
            "n_nonzero": int((enet["coef"].abs() > 1e-8).sum()),
            "top_by_abs_coef": enet.sort_values("abs_coef", ascending=False)
            .head(20)
            .drop(columns=["abs_coef"])
            .to_dict(orient="records"),
        }
    return report
