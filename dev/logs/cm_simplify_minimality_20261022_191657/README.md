# cm_simplify_minimality_20261022_191657

Raw CM → full simplify (`skip_if_edges_increase=False`) → force-reduce each remaining threshold unit → LUT roundtrip.

Minimality: after full simplify, every forced unit reduction must break LUT roundtrip; no `reduction_compatible` units should remain.

## Config

| Setting | Value |
|---------|-------|
| SMOKE | `False` |
| MAX_K | `5` |
| SAMPLE_N | `2048` |
| BUILDERS | `legacy, canonical` |

## Findings

- Functions evaluated: **8744**
- Status `ok` (minimal): **7972**
- Anomaly `simplify_fail`: **0** (expect 0)
- Anomaly `still_compatible`: **0** (expect 0)
- Anomaly `force_ok`: **772** functions (expect 0)

## Unit force-reduce trials

- fail (expected): **166104**
- ok (anomaly): **938**
- error: **0**

## Artifacts

- `cm_simplify_minimality_20261022_191657.log`
- `functions.csv`
- `unit_trials.csv`
- `01_force_ok_frac_vs_k.png`
- `02_mean_units_vs_k.png`
- `03_force_by_skip_reason.png`

