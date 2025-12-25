# Survival

Baseline survival pipeline for METABRIC using PCA + Cox (lifelines).

Usage:
```bash
python survival_pipeline.py \
  --data-dir brca_metabric \
  --duration-col OS_MONTHS \
  --event-col OS_STATUS
```

Outputs:
- `outputs/embedding.csv`
- `outputs/top_genes_by_component.csv`
- `outputs/metrics.json`
