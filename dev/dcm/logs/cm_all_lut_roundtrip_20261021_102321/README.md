# cm_all_lut_roundtrip_20261021_102321

Run **after fixing the permutation-group** issue (before the fusion-wiring fix).

Same protocol as the baseline (`mode=full`, MAX_K=5, SAMPLE_N=2048, seed=0).
Overall fail rate drops vs baseline, but k=3 still has the same 8 conflicts, and
higher-k samples still fail heavily — pointing at remaining fusion-wiring bugs.

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261021_102321` |
| MAX_K | 5 |
| SAMPLE_N / SAMPLE_SEED | 2048 / 0 |
| k=1..3 | exhaustive |
| k=4..5 | sample |

## Results

| k | mode | eval | fail | notes |
|---|------|------|------|-------|
| 1 | exhaustive | 4 | 0 | |
| 2 | exhaustive | 16 | 0 | |
| 3 | exhaustive | 256 | 8 | same 8 LUTs as baseline |
| 4 | sample | 2048 | 356 | was 415 |
| 5 | sample | 2048 | 991 | was 1151 |

- **function fail rate:** 1355 / 4372 ≈ **31.0%** (baseline ~36.0%)
- **failing k's:** 3 / 5
- **fail reasons:** 1355× `conflict` (no `incomplete`)
- **timing:** cm ≈ 4.89s, lut ≈ 7.64s

## Artifacts

- `cm_all_lut_roundtrip_20261021_102321.log`
- `01_fail_rate_vs_k.png` … `05_fail_reasons.png`
