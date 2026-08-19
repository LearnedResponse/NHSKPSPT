# NHSKPSPT — Normalized Hessian Spaces and Kronecker Products of Symmetric Persistent Tensors

Paper and exact verification bundle.

Preprint DOI: [10.5281/zenodo.22004540](https://doi.org/10.5281/zenodo.22004540)
Software bundle DOI: [10.5281/zenodo.22004644](https://doi.org/10.5281/zenodo.22004644)
(published 2026-08-19)

We revisit the Kronecker-closure claim for symmetric persistent tensors
in Gharahi's preprint (arXiv:2608.11182), isolate a Segre-ideal
obstruction in the proof of its Proposition 10 (no counterexample is
claimed), and develop a normalized polarized-Hessian mechanism giving
replacement results: a triangularizable–nil criterion, a
factorized-certificate criterion, a trace-zero Jordan-closure criterion
via Jacobson's generalized Engel theorem, unconditional Kronecker
closure when one factor has dimension at most three (every degree) and
for cubics at dimension at most four, a Veronese-contraction reduction
to a quartic base case, and, for persistent cubics, the result that
Jordan closure forces a commutative associative Hessian algebra.

## Layout

- `paper/` — the note (`main.tex` + compiled PDF). License: CC-BY-4.0.
- `verification/` — exact symbolic verification scripts and their
  committed output snapshots (Python 3, SymPy 1.12, rational arithmetic
  throughout, all randomness seeded). License: Apache-2.0. See
  `verification/README.md` for the result-to-script map and
  `verification/MANIFEST.sha256` for file pins.
- `RIGHTS.md` — the mixed-license grant. `CITATION.cff` — how to cite.

## Quick verification

```
pip install -r verification/requirements.txt
python3 verification/verify_all.py
```

The fast tier reruns seven deterministic scripts and byte-compares
their output against the committed snapshots. Longer runs (the 7175-form
quartic sweep, the full-product-space Kronecker tests) are listed by
`verify_all.py --list` with honest runtimes; known data caveats
(coefficient-one enumeration, no GL-orbit reduction, the incomplete
quintic block) are stated in `verification/README.md` and in the
paper's §8.

## Status

v0.1 — proof candidates pending external review; see the paper's
"A note on process" section. Both Zenodo records are published
(2026-08-19).
