# cm_all_lut_roundtrip_20261021_103226

Run **after also fixing the fusion-wiring** issue (permutation-group fix already in).

First clean pass on the single-builder (pre–legacy/canonical split) harness:
**0 failures** across all evaluated LUTs.

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261021_103226` |
| MAX_K | 5 |
| SAMPLE_N / SAMPLE_SEED | 2048 / 0 |
| k=1..3 | exhaustive |
| k=4..5 | sample |

## Results

| k | mode | eval | fail |
|---|------|------|------|
| 1 | exhaustive | 4 | 0 |
| 2 | exhaustive | 16 | 0 |
| 3 | exhaustive | 256 | 0 |
| 4 | sample | 2048 | 0 |
| 5 | sample | 2048 | 0 |

- **function fail rate:** 0 / 4372 = **0%**
- **failing k's:** none
- **timing:** cm ≈ 4.69s, lut ≈ 7.59s

Compared to the previous run (`…_102321`), the remaining k=3 conflicts and
higher-k sample failures are gone.

## Artifacts

- `cm_all_lut_roundtrip_20261021_103226.log`
- `01_fail_rate_vs_k.png` … `04_coverage.png` (no fail-reasons plot — nothing failed)
