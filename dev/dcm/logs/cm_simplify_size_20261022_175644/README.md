# cm_simplify_size_20261022_175644

All-LUT CM size comparison: `legacy` / `canonical` × `raw` / `nei` (no edge increase) / `full` simplify.

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261022_175644` |
| mode | `full` |
| BUILDERS | `legacy,canonical` |
| MODES | `raw,nei,full` |
| MAX_K | `5` |
| SAMPLE_N / SAMPLE_SEED | `2048` / `0` |
| effective_max_k | `5` |

## Results

- **rows:** 26232 (4340 functions × 2 builders × 3 modes, aggregated across k)

Mean nodes / edges by (builder, mode):

| builder | mode | mean nodes | mean edges | n |
|---------|------|------------|------------|---|
| legacy | raw | 29.95 | 65.88 | 4372 |
| legacy | nei | 29.86 | 65.78 | 4372 |
| legacy | full | 29.86 | 65.78 | 4372 |
| canonical | raw | 36.17 | 72.09 | 4372 |
| canonical | nei | 29.86 | 65.78 | 4372 |
| canonical | full | 29.86 | 65.78 | 4372 |

Mean nodes by (k, builder, mode):

| k | legacy/raw | legacy/nei | legacy/full | canonical/raw | canonical/nei | canonical/full |
|------|------|------|------|------|------|------|
| 1 | 4.5 | 3.0 | 3.0 | 4.5 | 3.0 | 3.0 |
| 2 | 7.6 | 6.0 | 6.0 | 8.0 | 6.0 | 6.0 |
| 3 | 13.1 | 12.4 | 12.4 | 14.4 | 12.4 | 12.4 |
| 4 | 22.2 | 22.1 | 22.1 | 25.7 | 22.1 | 22.1 |
| 5 | 40.1 | 40.1 | 40.1 | 49.6 | 40.1 | 40.1 |

## Artifacts

- `cm_simplify_size_20261022_175644.log`
- `sizes.csv`
- `01_nodes_boxplot_by_k.png`
- `01_edges_boxplot_by_k.png`
- `02_mean_nodes_edges_vs_k.png`
- `03_scatter_legacy_vs_canonical_nodes.png`
- `03_scatter_legacy_vs_canonical_edges.png`
- `04_delta_full_minus_nei.png`
- `05_sample_functions_six_combos.png`
- `README.md`
