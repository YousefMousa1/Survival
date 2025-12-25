import numpy as np
import pandas as pd


def parse_event(series):
    values = series.astype(str).str.split(":", n=1, expand=True)[0]
    values = values.replace({"NA": np.nan, "nan": np.nan})
    return pd.to_numeric(values, errors="coerce")


def load_clinical(path, duration_col, event_col):
    clinical = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    cols = ["PATIENT_ID", duration_col, event_col]
    missing = [c for c in cols if c not in clinical.columns]
    if missing:
        raise ValueError(f"Missing clinical columns: {missing}")
    df = clinical[cols].copy()
    df = df.rename(columns={"PATIENT_ID": "sample_id"})
    df["duration"] = pd.to_numeric(df[duration_col], errors="coerce")
    df["event"] = parse_event(df[event_col])
    df = df.dropna(subset=["duration", "event"])
    df["event"] = df["event"].astype(int)
    return df[["sample_id", "duration", "event"]]


def _normalize_expression(df):
    df["gene_id"] = df["Hugo_Symbol"].astype(str) + "_" + df["Entrez_Gene_Id"].astype(
        str
    )
    df = df.drop(columns=["Hugo_Symbol", "Entrez_Gene_Id"])
    df = df.set_index("gene_id")
    df = df.apply(pd.to_numeric, errors="coerce")
    if not df.index.is_unique:
        df = df.groupby(level=0).mean()
    return df


def get_expression_samples(path):
    header = pd.read_csv(path, sep="\t", nrows=0)
    return [col for col in header.columns if col not in ("Hugo_Symbol", "Entrez_Gene_Id")]


def load_expression(path, sample_ids=None, chunk_size=None):
    usecols = None
    if sample_ids is not None:
        available = set(get_expression_samples(path))
        keep = [sid for sid in sample_ids if sid in available]
        usecols = ["Hugo_Symbol", "Entrez_Gene_Id"] + keep

    if chunk_size:
        chunks = []
        for chunk in pd.read_csv(
            path, sep="\t", low_memory=False, usecols=usecols, chunksize=chunk_size
        ):
            chunks.append(_normalize_expression(chunk))
        if not chunks:
            return pd.DataFrame()
        expr = pd.concat(chunks, axis=0)
        if not expr.index.is_unique:
            expr = expr.groupby(level=0).mean()
        return expr

    expr = pd.read_csv(path, sep="\t", low_memory=False, usecols=usecols)
    if "Hugo_Symbol" not in expr.columns:
        raise ValueError("Expression file missing Hugo_Symbol")
    if "Entrez_Gene_Id" not in expr.columns:
        raise ValueError("Expression file missing Entrez_Gene_Id")
    return _normalize_expression(expr)
