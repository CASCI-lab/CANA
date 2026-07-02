# Changelog — branch `LUT-generation-and-schema-search`

This is a **branch-specific** changelog. The `master` branch does not currently
maintain a changelog, so this file documents how the
`LUT-generation-and-schema-search` branch diverges from `master`
(branch point: commit `de25baf`). It is reconstructed from the branch's commit
history and is intended to be folded into a project-wide changelog if/when
`master` adopts one.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — `LUT-generation-and-schema-search`

### Added

- **Partial look-up table (LUT) generation.**
  - `cana.utils.fill_out_lut()` — expand a partial LUT (with wildcard input
    patterns) into a complete `2**k` LUT, marking unspecified states `'?'` and
    conflicting states `'!'`.
  - `BooleanNode.from_partial_lut()` — build a node from a partial LUT, with
    `fill_clashes` and `fill_missing_output_randomly` options.
  - `BooleanNode.generate_with_required_bias()` and
    `generate_with_required_effective_connectivity()` — generators that fill in
    missing (`'?'`) outputs to hit a target node bias / effective connectivity.
  - `BooleanNode.fill_missing_output_randomly()` — generator filling `'?'`
    outputs randomly.
  - Tutorial `tutorials/Generating from Partial LUTs.ipynb` and
    `tutorials/partial_LUT_demo_nodes.txt`.
- **Annihilation / generation ("anni_gen") analysis.**
  - `BooleanNode.get_annihilation_generation_rules()` (wildcard and two-symbol
    variants), `get_anni_gen_coverage()`, `input_symmetry_mean_anni_gen()`, and
    annihilation/generation input-redundancy and effective-connectivity measures.
  - `automata/schema_search_tools.py` — schema-search utilities
    (`annihilation_generation_rules`, `maintenance_rules`,
    `combine_output_lists`, `shuffle_and_generate`,
    `shuffle_wildcards_in_schemata`).
- **Cython acceleration.**
  - New `cana/cboolean_node.pyx` helpers, and fast two-symbol coverage helpers in
    `cana/canalization/cboolean_canalization.pyx`
    (`pi_covers_fast`, `expand_ts_logic_fast`, `ts_covers_fast`).
- **Drawing.**
  - `cana/drawing/plot_look_up_table.py` and
    `plot_anni_gen_schemata()` in `cana/drawing/schema_vis.py` (side-by-side
    annihilation/generation schemata plots).
- **Datasets & tests.** `GP()` in `cana/datasets/bools.py`; new dataset package
  inits; test suites `tests/test_fill_out_lut.py`, `tests/test_ts_coverage.py`,
  `tests/test_schema_search_tools.py`, `tests/test_schemata_ordering.py`, and
  extensive additions to `tests/test_boolean_node.py`.

### Changed

- **Deterministic, canonical ordering of schemata output (reproducibility fix).**
  Prime implicants are stored as Python `set`s, whose iteration order is
  randomized per process (`PYTHONHASHSEED`), so schemata output previously
  permuted from run to run (content was always correct; only order varied). Output
  order is now sorted / canonical everywhere it is exposed:
  - `BooleanNode.schemata_look_up_table(type="pi")` iterates the PI set in sorted
    order.
  - `find_two_symbols_v2()` sorts both its prime-implicant input and its returned
    two-symbol list, so `_two_symbols` (and every consumer — the `type="ts"`
    schemata LUT, two-symbol coverage, the canalizing map, and the drawing code)
    is reproducible.
  - `computes_pi_coverage()` builds each covering list in sorted order.
  - `cana/drawing/schema_vis.py` renders prime-implicant rows in sorted order.
  - `BooleanNode.get_annihilation_generation_rules()` sorts its PI/TS output.
  - **No content change** — only ordering is now stable across processes and
    machines. Downstream code that pinned `PYTHONHASHSEED` to get reproducible
    results no longer needs to.

### Fixed

- **`fill_out_lut` wildcard `'2'` bug.** `'2'` was accepted and routed into the
  wildcard-expansion branch but never actually expanded, leaving spurious
  non-binary keys (e.g. `"12"`) in the output and dropping coverage. `'2'` now
  behaves like the other wildcards (`-`, `#`, `x`).
- Corrected a misleading comment in `automata/schema_search_tools.py` about clash
  resolution: with `fill_clashes=True`, `fill_out_lut` keeps the **later** entry
  (not the first).
- Various division-by-zero guards for nodes with no annihilation/generation rules.

### Performance

- **Two-symbol coverage.** `computes_ts_coverage()` rewritten with loop inversion
  (expand each schema once and mark covered minterms directly instead of
  re-expanding per input state), plus memoization of `find_two_symbols_v2()` on
  `(k, frozenset(prime_implicants))`. Output is byte-for-byte identical.
- **`fill_out_lut`.** Rewritten to expand wildcards with integer bit math and emit
  the LUT already in sorted order (no final sort), ~1.4–1.9× faster; output
  identical for all previously-valid inputs.
