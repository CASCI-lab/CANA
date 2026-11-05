# cm_all_lut_roundtrip_20261021_005731

Baseline run **before any fix**.

CM ↔ LUT roundtrip over Boolean LUTs (`all_lut_cm_roundtrip.py`, `mode=full`).
Fails when reconstructing the LUT from the canalizing map does not match the original
(mostly `reason=conflict`).

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261021_005731` |
| MAX_K | 5 |
| SAMPLE_N / SAMPLE_SEED | 2048 / 0 |
| k=1..3 | exhaustive |
| k=4..5 | sample |

## Results

| k | mode | eval | fail | notes |
|---|------|------|------|-------|
| 1 | exhaustive | 4 | 0 | |
| 2 | exhaustive | 16 | 0 | |
| 3 | exhaustive | 256 | 8 | all `conflict` |
| 4 | sample | 2048 | 415 | |
| 5 | sample | 2048 | 1151 | |

- **function fail rate:** 1574 / 4372 ≈ **36.0%**
- **failing k's:** 3 / 5
- **fail reasons:** 1572× `conflict`, 2× `incomplete`
- **timing:** cm ≈ 5.63s, lut ≈ 8.90s

## Artifacts

- `cm_all_lut_roundtrip_20261021_005731.log`
- `01_fail_rate_vs_k.png` … `05_fail_reasons.png`
