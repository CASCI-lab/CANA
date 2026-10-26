# cm_bbm_roundtrip_20261020_235736

Baseline run **before any fix**.

CM ↔ LUT roundtrip over real Boolean models from the BBM corpus
(`bbm_cm_roundtrip.py`, `mode=full`). For each node: build a canalizing map,
reconstruct the LUT, and compare. This log is a **partial** run (models
`id=001`–`100` of 285 listed in `summary.csv`); no final `SUMMARY` block was
written.

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261020_235736` |
| MAX_K | 12 (nodes with k > 12 skipped; LUT not built) |
| TIMEOUT_CM_S / TIMEOUT_LUT_S | 30.0 / 30.0 |
| BBM models | `/home/kyuhyongpark/CANA/BBM/models` |
| models in log | 100 / 285 selected |

## Results

| outcome | count |
|---------|------:|
| OK | 4911 |
| FAIL | 175 |
| SKIP | 65 |
| ERROR | 0 |

- **function fail rate:** 175 / 5086 ≈ **3.44%** (among evaluated nodes)
- **model fail rate:** 38 / 100 = **38%** (any evaluated node failed)
- **fail reasons:** 175× `conflict`
- **skips:** 42× `TIMEOUT_CM`, 23× `k > MAX_K`
- **timing (sum of model totals):** cm ≈ 1489s (~24.8 min), lut ≈ 102s

### Failures by input size k

| k | fail / eval |
|---|-------------|
| 1–2 | 0 / 3359 |
| 3 | 1 / 659 |
| 4 | 31 / 411 |
| 5 | 34 / 272 |
| 6 | 43 / 177 |
| 7 | 26 / 85 |
| 8 | 18 / 75 |
| 9 | 15 / 44 |
| 10 | 5 / 28 |
| 11 | 2 / 11 |
| ≥12 | 0 (mostly skipped or rare) |

### Top failing models (by fail count)

| id | name | fail / eval |
|----|------|-------------|
| 079 | COLORECTAL-TUMORIGENESIS | 40 / 190 |
| 004 | ERBB-RECEPTOR-SIGNALING | 31 / 225 |
| 002 | SIGNAL-TRANSDUCTION-IN-FIBROBLASTS | 27 / 133 |
| 075 | INFLAMMATORY-BOWEL-DISEASE | 9 / 44 |
| 060 | STOMATAL-OPENING | 7 / 49 |

## Artifacts

- `cm_bbm_roundtrip_20261020_235736.log`
- `01_outcomes.png` … `05_model_fail_fraction.png`
  (from `cm_bbm_roundtrip_summary.py`)
