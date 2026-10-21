# DCM figures — MAMMALIAN-CELL-CYCLE-2006

- **MODEL_ID**: `023`
- **network nodes**: 10 (constants=0)
- **model dir**: `/home/kyuhyongpark/CANA/BBM/models/[id-023]__[var-9]__[in-1]__[MAMMALIAN-CELL-CYCLE-2006]`

Four Dynamics Canalization Map variants: `legacy` / `canonical` × simplify on/off.

Node types are `variable`, `threshold`, and (legacy only) `fusion`.
Canonical threshold units carry **roles** (`schema`, `symmetry`, `polarity0`, `polarity1`); legacy thresholds typically have no role and may use fusion nodes instead of nested polarity/symmetry thresholds.

## Comparison

|  | `legacy_full` | `legacy_simplify` | `canonical_full` | `canonical_simplify` |
| --- | --- | --- | --- | --- |
| figure | `legacy_full.png` | `legacy_simplify.png` | `canonical_full.png` | `canonical_simplify.png` |
| nodes | 66 | 53 | 77 | 48 |
| edges | 129 | 116 | 140 | 111 |
| node type `fusion` | 11 | 11 | 0 | 0 |
| node type `threshold` | 35 | 22 | 57 | 28 |
| node type `variable` | 20 | 20 | 20 | 20 |
| edge type `fused` | 11 | 6 | 22 | 6 |
| edge type `fusing` | 23 | 23 | 0 | 0 |
| edge type `literal` | 60 | 52 | 83 | 65 |
| edge type `out` | 35 | 35 | 35 | 40 |
| threshold τ=1 | 13 | 0 | 35 | 6 |
| threshold τ=2 | 13 | 13 | 13 | 13 |
| threshold τ=3 | 4 | 4 | 4 | 4 |
| threshold τ=4 | 5 | 5 | 5 | 5 |
| threshold role `(none)` | 35 | 22 | 0 | 0 |
| threshold role `polarity0` | 0 | 0 | 2 | 0 |
| threshold role `polarity1` | 0 | 0 | 9 | 0 |
| threshold role `schema` | 0 | 0 | 35 | 22 |
| threshold role `symmetry` | 0 | 0 | 11 | 6 |

