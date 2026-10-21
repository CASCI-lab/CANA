# cm_all_lut_roundtrip_20261021_171750

Run **after implementing both `legacy` and `canonical` builders**.

Same LUT sampling as earlier runs, but each function is roundtripped twice —
once per builder. Both builders pass completely.

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261021_171750` |
| BUILDERS | `legacy`, `canonical` |
| MAX_K | 5 |
| SAMPLE_N / SAMPLE_SEED | 2048 / 0 |
| k=1..3 | exhaustive |
| k=4..5 | sample |

## Results

Per builder (identical eval counts):

| k | mode | eval | legacy fail | canonical fail |
|---|------|------|-------------|----------------|
| 1 | exhaustive | 4 | 0 | 0 |
| 2 | exhaustive | 16 | 0 | 0 |
| 3 | exhaustive | 256 | 0 | 0 |
| 4 | sample | 2048 | 0 | 0 |
| 5 | sample | 2048 | 0 | 0 |

- **overall function fail rate:** 0 / 8744 = **0%** (4372 LUTs × 2 builders)
- **per builder:** legacy 0/4372, canonical 0/4372
- **failing k's:** none
- **timing:** cm ≈ 8.99s, lut ≈ 16.29s (both builders combined)

## Artifacts

- `cm_all_lut_roundtrip_20261021_171750.log`
- `01_fail_rate_vs_k.png` … `04_coverage.png`
