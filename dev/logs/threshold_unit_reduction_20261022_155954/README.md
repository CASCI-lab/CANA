# threshold_unit_reduction_20261022_155954

Raw CM → per-threshold-unit reduction (AND/OR/LINEAR) → LUT roundtrip.

## Config

| Setting | Value |
|---------|-------|
| timestamp | `20261022_155954` |
| mode | `full` |
| BUILDERS | `legacy,canonical` |
| MAX_K | `5` |
| SAMPLE_N / SAMPLE_SEED | `2048` / `0` |
| effective_max_k | `5` |

## Results

| k | mode | canonical fail/attempt | legacy fail/attempt |
|------|------|------|------|
| 1 | exhaustive | 0 / 4 | 0 / 4 |
| 2 | exhaustive | 0 / 32 | 0 / 24 |
| 3 | exhaustive | 0 / 768 | 0 / 184 |
| 4 | sample | 0 / 12926 | 0 / 187 |
| 5 | sample | 0 / 32903 | 0 / 9 |

- **overall attempt fail rate:** `0/47041 = 0.000000`

## Per-class totals

### canonical

| class | seen | attempt | ok | fail | skip |
|-------|------|---------|----|------|------|
| LINEAR | 20889 | 20889 | 20889 | 0 | 0 |
| AND | 56074 | 6966 | 6966 | 0 | 49108 |
| OR | 31819 | 18778 | 18778 | 0 | 13041 |
| OTHER | 2302 | 0 | 0 | 0 | 2302 |

### legacy

| class | seen | attempt | ok | fail | skip |
|-------|------|---------|----|------|------|
| LINEAR | 364 | 364 | 364 | 0 | 0 |
| AND | 49444 | 0 | 0 | 0 | 49444 |
| OR | 31819 | 44 | 44 | 0 | 31775 |
| OTHER | 2302 | 0 | 0 | 0 | 2302 |

## Artifacts

- `threshold_unit_reduction_20261022_155954.log`
- `01_fail_rate_vs_k.png`
- `02_outcomes_per_k.png`
- `03_by_class.png`
- `README.md`
