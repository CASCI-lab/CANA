# dcm_simplify_all_lut_roundtrip_20261021_225610

All-LUT CM → stock DCM simplify (True/False) → LUT roundtrip for `legacy` and `canonical` builders (4 combinations).

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261021_225610` |
| mode | `full` |
| BUILDERS | `legacy,canonical` |
| SIMPLIFY_MODES | `False,True` |
| MAX_K | `5` |
| SAMPLE_N / SAMPLE_SEED | `2048` / `0` |
| effective_max_k | `5` |

## Results

| k | mode | canonical / simplify=False fail | canonical / simplify=True fail | legacy / simplify=False fail | legacy / simplify=True fail |
|------|------|------|------|------|------|
| 1 | exhaustive | 0 / 4 | 0 / 4 | 0 / 4 | 0 / 4 |
| 2 | exhaustive | 0 / 16 | 0 / 16 | 0 / 16 | 0 / 16 |
| 3 | exhaustive | 0 / 256 | 0 / 256 | 0 / 256 | 0 / 256 |
| 4 | sample | 0 / 2048 | 0 / 2048 | 0 / 2048 | 0 / 2048 |
| 5 | sample | 0 / 2048 | 0 / 2048 | 0 / 2048 | 0 / 2048 |

Per-(builder, simplify) totals:

- **canonical / simplify=False:** 0 / 4372 = **0.00%**
- **canonical / simplify=True:** 0 / 4372 = **0.00%**
- **legacy / simplify=False:** 0 / 4372 = **0.00%**
- **legacy / simplify=True:** 0 / 4372 = **0.00%**

- **overall function fail rate:** `0/17488 = 0.000000`
- **timing:** cm ≈ 19.426s, lut ≈ 30.078s

## Artifacts

- `dcm_simplify_all_lut_roundtrip_20261021_225610.log`
- `01_fail_rate_vs_k.png`
- `02_outcomes_per_k.png`
- `03_timing_totals_vs_k.png`
- `04_coverage.png`
- `README.md`
