#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from survival_lib.io import load_clinical, load_expression, load_expression_genes
from survival_lib.models import cv_cox, fit_elastic_net_cox, univariate_cox
from survival_lib.preprocess import fit_pca, qc_and_scale
from survival_lib.reports import build_combined_report, top_genes_by_component
from survival_lib.tuning import tune_elastic_net


def main():
    parser = argparse.ArgumentParser(description="METABRIC survival pipeline baseline.")
    parser.add_argument(
        "--data-dir",
        default="brca_metabric",
        help="Path to METABRIC data directory.",
    )
    parser.add_argument(
        "--expression-file",
        default="data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        help="Expression matrix filename inside data-dir.",
    )
    parser.add_argument(
        "--ihc4-expression-file",
        default="data_mrna_illumina_microarray.txt",
        help="Expression matrix filename for reduced IHC4 mode.",
    )
    parser.add_argument(
        "--clinical-file",
        default="data_clinical_patient.txt",
        help="Clinical patient filename inside data-dir.",
    )
    parser.add_argument(
        "--duration-col",
        default="OS_MONTHS",
        help="Clinical duration column (e.g., OS_MONTHS or RFS_MONTHS).",
    )
    parser.add_argument(
        "--event-col",
        default="OS_STATUS",
        help="Clinical event column (e.g., OS_STATUS or RFS_STATUS).",
    )
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--top-genes", type=int, default=5000)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Read expression in chunks (rows per chunk) for large matrices.",
    )
    parser.add_argument("--reduced-ihc4", action="store_true")
    parser.add_argument("--max-missing", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument(
        "--embedding",
        default="pca",
        choices=["pca", "tjepa"],
        help="Embedding method to learn Z.",
    )
    parser.add_argument("--tjepa-config", default=None)
    parser.add_argument("--tjepa-embed-dim", type=int, default=64)
    parser.add_argument("--tjepa-num-layers", type=int, default=4)
    parser.add_argument("--tjepa-num-heads", type=int, default=8)
    parser.add_argument("--tjepa-mlp-dim", type=int, default=256)
    parser.add_argument("--tjepa-dropout", type=float, default=0.1)
    parser.add_argument("--tjepa-activation", type=str, default="relu")
    parser.add_argument("--tjepa-pred-embed-dim", type=int, default=64)
    parser.add_argument("--tjepa-pred-num-layers", type=int, default=2)
    parser.add_argument("--tjepa-pred-num-heads", type=int, default=4)
    parser.add_argument("--tjepa-pred-dropout", type=float, default=0.1)
    parser.add_argument("--tjepa-pred-dim-feedforward", type=int, default=256)
    parser.add_argument("--tjepa-mask-allow-overlap", action="store_true")
    parser.add_argument("--tjepa-mask-min-ctx-share", type=float, default=0.2)
    parser.add_argument("--tjepa-mask-max-ctx-share", type=float, default=0.4)
    parser.add_argument("--tjepa-mask-min-trgt-share", type=float, default=0.2)
    parser.add_argument("--tjepa-mask-max-trgt-share", type=float, default=0.4)
    parser.add_argument("--tjepa-mask-num-preds", type=int, default=4)
    parser.add_argument("--tjepa-mask-num-encs", type=int, default=1)
    parser.add_argument("--tjepa-n-cls-tokens", type=int, default=1)
    parser.add_argument("--tjepa-epochs", type=int, default=100)
    parser.add_argument("--tjepa-batch-size", type=int, default=64)
    parser.add_argument("--tjepa-lr", type=float, default=1e-4)
    parser.add_argument("--tjepa-weight-decay", type=float, default=1e-5)
    parser.add_argument("--tjepa-momentum", type=float, default=0.996)
    parser.add_argument("--tjepa-device", default=None)
    parser.add_argument("--tjepa-forward-only", action="store_true")
    parser.add_argument("--top-genes-per-component", type=int, default=25)
    parser.add_argument(
        "--min-embed-variance",
        type=float,
        default=0.0,
        help="Drop embedding dimensions with variance below this threshold.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-univariate-cox", action="store_true")
    parser.add_argument("--run-elastic-net-cox", action="store_true")
    parser.add_argument("--enet-penalizer", type=float, default=0.1)
    parser.add_argument(
        "--enet-l1-ratios",
        default="0.0,1.0",
        help="Comma-separated l1_ratio values to fit.",
    )
    parser.add_argument("--tune-elastic-net", action="store_true")
    parser.add_argument(
        "--enet-penalizers",
        default="0.001,0.01,0.1,1.0",
        help="Comma-separated penalizer values for tuning.",
    )
    parser.add_argument(
        "--enet-tune-l1-ratios",
        default="0.0,0.5,1.0",
        help="Comma-separated l1_ratio values for tuning.",
    )
    parser.add_argument("--combined-report", action="store_true")
    args = parser.parse_args()
    if args.tjepa_config:
        config = json.loads(Path(args.tjepa_config).read_text())
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    data_dir = Path(args.data_dir)
    expr_path = data_dir / args.expression_file
    clinical_path = data_dir / args.clinical_file

    clinical = load_clinical(clinical_path, args.duration_col, args.event_col)

    if args.reduced_ihc4:
        clinical_full = pd.read_csv(clinical_path, sep="\t", comment="#", low_memory=False)
        col_candidates = {
            "sample_id": ["PATIENT_ID", "Patient Identifier"],
            "age": ["Age at Diagnosis", "AGE_AT_DIAGNOSIS"],
            "chemotherapy": ["Chemotherapy", "CHEMOTHERAPY"],
            "er_ihc": ["ER status measured by IHC", "ER_IHC"],
            "hormone_therapy": ["Hormone Therapy", "HORMONE_THERAPY"],
            "radio_therapy": ["Radio Therapy", "RADIO_THERAPY"],
        }
        resolved = {}
        for target, candidates in col_candidates.items():
            for cand in candidates:
                if cand in clinical_full.columns:
                    resolved[target] = cand
                    break
        missing = [k for k in col_candidates if k not in resolved]
        if missing:
            raise SystemExit(f"Missing clinical columns for IHC4: {missing}")
        clinical_red = clinical_full[list(resolved.values())].rename(
            columns={v: k for k, v in resolved.items()}
        )

        def to_binary(series):
            values = series.astype(str).str.strip().str.lower()
            mapping = {
                "yes": 1,
                "no": 0,
                "positive": 1,
                "positve": 1,
                "negative": 0,
            }
            return values.map(mapping)

        clinical_red["age"] = pd.to_numeric(clinical_red["age"], errors="coerce")
        clinical_red["chemotherapy"] = to_binary(clinical_red["chemotherapy"])
        clinical_red["er_ihc"] = to_binary(clinical_red["er_ihc"])
        clinical_red["hormone_therapy"] = to_binary(clinical_red["hormone_therapy"])
        clinical_red["radio_therapy"] = to_binary(clinical_red["radio_therapy"])
        clinical_red = clinical_red.dropna()

        genes = ["EGFR", "PGR", "ERBB2", "MKI67"]
        expr_genes = load_expression_genes(
            data_dir / args.ihc4_expression_file,
            genes,
            sample_ids=clinical_red["sample_id"].tolist(),
            chunk_size=args.chunk_size if args.chunk_size > 0 else None,
        )
        if expr_genes.empty:
            raise SystemExit("No IHC4 genes found in expression file.")
        expr_genes = expr_genes.T

        merged = clinical_red.set_index("sample_id").join(expr_genes, how="inner")
        merged = merged.dropna()
        if merged.empty:
            raise SystemExit("No samples after merging clinical and gene features.")

        continuous_cols = ["age"] + genes
        scaler = StandardScaler()
        merged[continuous_cols] = scaler.fit_transform(merged[continuous_cols])
        x_scaled = merged[
            continuous_cols
            + ["chemotherapy", "er_ihc", "hormone_therapy", "radio_therapy"]
        ]
        clinical = clinical.set_index("sample_id").loc[merged.index]
        pca = None
        z = x_scaled
    else:
        expr = load_expression(
            expr_path,
            sample_ids=clinical["sample_id"].tolist(),
            chunk_size=args.chunk_size if args.chunk_size > 0 else None,
        )

        shared_samples = sorted(set(expr.columns).intersection(clinical["sample_id"]))
        if not shared_samples:
            raise SystemExit("No shared samples between expression and clinical data.")

        expr = expr[shared_samples]
        clinical = clinical.set_index("sample_id").loc[shared_samples]

        x_scaled = qc_and_scale(expr, args.max_missing, args.top_genes)
        if args.embedding == "pca":
            pca, z = fit_pca(x_scaled, args.n_components, args.random_state)
        else:
            from survival_lib.tjepa import train_tjepa, tjepa_forward_only

            if args.tjepa_forward_only:
                z_array = tjepa_forward_only(
                    x_scaled,
                    embed_dim=args.tjepa_embed_dim,
                    num_layers=args.tjepa_num_layers,
                    num_heads=args.tjepa_num_heads,
                    mlp_dim=args.tjepa_mlp_dim,
                    dropout=args.tjepa_dropout,
                    activation=args.tjepa_activation,
                    n_cls_tokens=args.tjepa_n_cls_tokens,
                    seed=args.random_state,
                    device=args.tjepa_device,
                )
            else:
                z_array = train_tjepa(
                    x_scaled,
                    embed_dim=args.tjepa_embed_dim,
                    num_layers=args.tjepa_num_layers,
                    num_heads=args.tjepa_num_heads,
                    mlp_dim=args.tjepa_mlp_dim,
                    dropout=args.tjepa_dropout,
                    activation=args.tjepa_activation,
                    pred_embed_dim=args.tjepa_pred_embed_dim,
                    pred_num_layers=args.tjepa_pred_num_layers,
                    pred_num_heads=args.tjepa_pred_num_heads,
                    pred_dropout=args.tjepa_pred_dropout,
                    pred_dim_feedforward=args.tjepa_pred_dim_feedforward,
                    mask_allow_overlap=args.tjepa_mask_allow_overlap,
                    mask_min_ctx_share=args.tjepa_mask_min_ctx_share,
                    mask_max_ctx_share=args.tjepa_mask_max_ctx_share,
                    mask_min_trgt_share=args.tjepa_mask_min_trgt_share,
                    mask_max_trgt_share=args.tjepa_mask_max_trgt_share,
                    mask_num_preds=args.tjepa_mask_num_preds,
                    mask_num_encs=args.tjepa_mask_num_encs,
                    n_cls_tokens=args.tjepa_n_cls_tokens,
                    epochs=args.tjepa_epochs,
                    batch_size=args.tjepa_batch_size,
                    lr=args.tjepa_lr,
                    weight_decay=args.tjepa_weight_decay,
                    momentum=args.tjepa_momentum,
                    seed=args.random_state,
                    device=args.tjepa_device,
                )
            z = pd.DataFrame(
                z_array,
                index=x_scaled.index,
                columns=[f"TJEPA{i+1}" for i in range(z_array.shape[1])],
            )
            pca = None

    if args.min_embed_variance > 0:
        embed_var = z.var(axis=0)
        keep_cols = embed_var[embed_var >= args.min_embed_variance].index
        z = z[keep_cols]

    c_indexes = cv_cox(
        z,
        clinical["duration"],
        clinical["event"],
        args.n_splits,
        args.random_state,
        args.penalizer,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z.to_csv(output_dir / "embedding.csv")
    if pca is not None:
        top_genes = top_genes_by_component(
            pca, x_scaled.columns, args.top_genes_per_component
        )
        top_genes.to_csv(output_dir / "top_genes_by_component.csv", index=False)

    uni = None
    if args.run_univariate_cox:
        uni = univariate_cox(x_scaled, clinical["duration"], clinical["event"])
        uni.to_csv(output_dir / "univariate_cox.csv", index=False)

    enet_by_ratio = {}
    if args.run_elastic_net_cox:
        l1_ratios = [
            float(v.strip())
            for v in args.enet_l1_ratios.split(",")
            if v.strip() != ""
        ]
        for ratio in l1_ratios:
            enet = fit_elastic_net_cox(
                x_scaled,
                clinical["duration"],
                clinical["event"],
                args.enet_penalizer,
                ratio,
            )
            enet_by_ratio[ratio] = enet
            tag = f"enet_l1_{ratio:.2f}".replace(".", "p")
            enet.to_csv(output_dir / f"elastic_net_cox_{tag}.csv", index=False)

    metrics = {
        "c_index_mean": float(np.mean(c_indexes)),
        "c_index_std": float(np.std(c_indexes)),
        "c_index_folds": c_indexes,
        "n_samples": int(z.shape[0]),
        "n_genes": int(x_scaled.shape[1]),
        "duration_col": args.duration_col,
        "event_col": args.event_col,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.tune_elastic_net:
        penalizers = [
            float(v.strip())
            for v in args.enet_penalizers.split(",")
            if v.strip() != ""
        ]
        l1_ratios = [
            float(v.strip())
            for v in args.enet_tune_l1_ratios.split(",")
            if v.strip() != ""
        ]
        tuning = tune_elastic_net(
            x_scaled,
            clinical["duration"],
            clinical["event"],
            penalizers,
            l1_ratios,
            args.n_splits,
            args.random_state,
        )
        tuning.to_csv(output_dir / "elastic_net_tuning.csv", index=False)

    if args.combined_report:
        report = build_combined_report(metrics, uni, enet_by_ratio)
        (output_dir / "combined_report.json").write_text(
            json.dumps(report, indent=2)
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
