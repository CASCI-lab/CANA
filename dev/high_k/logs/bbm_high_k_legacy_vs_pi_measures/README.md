# bbm_high_k_legacy_vs_pi_measures — summary

- `MIN_K=13` (timed rules with `k >= 13`; assume all `k < 13` succeed)
- `TIMEOUT_S=10`
- high-*k* rules timed: **402**
- Cell Collective tag in metadata: `cell-collective`
- figures: `success_ratio_by_k.png`, `completion_curves.png`

## All BBM models

### Prime implicants on high-*k* rules

Fraction of timed rules (`k >= MIN_K`) where the PI / one-symbol schemata stage finished `ok` within the per-stage timeout.

| pipeline | stage | method | ok | n | fraction |
|---|---|---|---:|---:|---:|
| legacy | `qm` | LUT → Quine–McCluskey one-symbol schemata | 47 | 402 | 11.7% |
| new | `pi` | expression-cascade one-symbol schemata | 320 | 402 | 79.6% |
| either | `qm` ∨ `pi` | at least one pipeline | 320 | 402 | 79.6% |

### Measure success on high-*k* rules

| measure | pipeline | ok/n | fraction |
|---|---|---:|---:|
| bias | legacy | 277/402 | 68.9% |
| bias | new | 316/402 | 78.6% |
| activities | legacy | 266/402 | 66.2% |
| activities | new | 317/402 | 78.9% |
| edge redundancy | legacy | 46/402 | 11.4% |
| edge redundancy | new | 316/402 | 78.6% |

### Fully processable BBM models

Assumption: every update rule with `k < 13` can be processed. A model is then fully processable iff **every** rule with `k >= 13` obtained prime implicants under that pipeline.

| | count | / 285 models |
|---|---:|---:|
| models in cohort | 285 | 100% |
| models with no `k >= 13` rule (auto-ok) | 239 | 83.9% |
| models with ≥1 high-*k* rule | 46 | 16.1% |
| fully processable — **legacy** (`qm`) | 244 | 85.6% |
| fully processable — **new** (`pi`) | 271 | 95.1% |
| fully processable — **either** | 271 | 95.1% |

Of the high-*k*-bearing models only:

- legacy PI for all high-*k* rules: **5** / 46
- new PI for all high-*k* rules: **32** / 46
- either pipeline for all high-*k* rules: **32** / 46

## Cell Collective models (`cell-collective`)

Subset of BBM models whose `metadata.json` keywords include `cell-collective` (79 models; 20 high-*k* rules timed).

### Prime implicants on high-*k* rules

| pipeline | stage | method | ok | n | fraction |
|---|---|---|---:|---:|---:|
| legacy | `qm` | LUT → Quine–McCluskey one-symbol schemata | 2 | 20 | 10.0% |
| new | `pi` | expression-cascade one-symbol schemata | 20 | 20 | 100.0% |
| either | `qm` ∨ `pi` | at least one pipeline | 20 | 20 | 100.0% |

### Measure success on high-*k* rules

| measure | pipeline | ok/n | fraction |
|---|---|---:|---:|
| bias | legacy | 19/20 | 95.0% |
| bias | new | 20/20 | 100.0% |
| activities | legacy | 17/20 | 85.0% |
| activities | new | 20/20 | 100.0% |
| edge redundancy | legacy | 2/20 | 10.0% |
| edge redundancy | new | 20/20 | 100.0% |

### Fully processable Cell Collective models

Assumption: every update rule with `k < 13` can be processed. A model is then fully processable iff **every** rule with `k >= 13` obtained prime implicants under that pipeline.

| | count | / 79 models |
|---|---:|---:|
| models in cohort | 79 | 100% |
| models with no `k >= 13` rule (auto-ok) | 73 | 92.4% |
| models with ≥1 high-*k* rule | 6 | 7.6% |
| fully processable — **legacy** (`qm`) | 74 | 93.7% |
| fully processable — **new** (`pi`) | 79 | 100.0% |
| fully processable — **either** | 79 | 100.0% |

Of the high-*k*-bearing models only:

- legacy PI for all high-*k* rules: **1** / 6
- new PI for all high-*k* rules: **6** / 6
- either pipeline for all high-*k* rules: **6** / 6
