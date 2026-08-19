# Ancillary code and exact outputs — "Normalized Hessian Spaces and Kronecker Products of Symmetric Persistent Tensors"

Exact symbolic verification scripts (Python 3, SymPy 1.12, rational
arithmetic throughout — no floating point) supporting the paper. Every
random draw is seeded; every script is deterministic and reproduces its
committed `out_*.txt` snapshot byte-for-byte. Run `python3 verify_all.py`
for the fast tier (regenerates and byte-compares the quick outputs), or
see the map below to run individual scripts.

## Map: paper results ↔ scripts ↔ output snapshots

| Paper item | Script(s) | Snapshot | Runtime |
|---|---|---|---|
| §2 gap analysis: polarized identity holds at nondecomposable points (probes; a failure would have refuted the theorem) | `probe.py`, `run_probes.py` | `out_probes.txt` | minutes |
| Source-review checks (condition (a), decomposable identity, Cauchy structure, Example 13, triangular normal forms) | `check_a.py`, `check_b.py`, `check_cauchy.py`, `check_ex13.py`, `check_tri.py` | `out_a.txt`, `out_b.txt`, `out_cauchy.txt`, `out_ex13.txt`, `out_tri.txt` | seconds each |
| Theorem 3.3 (T + Nil) end-to-end on explicit cases, incl. rank-3 slots | `repair1/e2e.py`, `repair1/e2e_more.py`, `repair1/e2e_r3.py` | `repair1/out_e2e*.txt` | seconds–minutes |
| Theorem 3.5 (factorized certificate) / contraction | `repair1/contract.py`, `repair1/closure_oneslot.py` | `repair1/out_contract.txt`, `repair1/out_closure_oneslot.txt` | minutes |
| Gate implementations (TR, JC, CK, Nil, T) and diagonal-base machinery | `repair2/gates.py`, `repair1/nspace.py`, `klib.py` | (libraries) | — |
| §8 census: ten named forms | `repair2/run_named.py` | `repair2/out_named.txt` | minutes |
| §8 census: quartic sweep (7175 forms → 156 certified) and quintic partial block (151) | `repair1/hunt.py` (certificates), `repair2/run_hunt.py` (gates) | `repair1/out_hunt.txt`, `repair2/out_hunt_gates_44.txt`, `_45.txt` | hours |
| §8 census: four Kronecker products built in the full product space (D = 4, 6, 8, 9) | `repair2/run_kron.py` | `repair2/out_kron_gates.txt` | ~1 h (the D=9 symbolic certificate times out by design and falls back to the decomposable route, as disclosed in the paper) |
| §8 negative controls (6 nonpersistent forms fail JC; the JC-passes/TR-fails toy; T-without-JC span) | `repair2/controls.py`, `repair1/toy_sep.py` | `repair2/out_controls.txt`, `repair1/out_toy_sep.txt` | minutes |
| §8 pair/equation bookkeeping (2204 deduplicated pairs / 320 distinct inputs) | `repair2/jstats.py` | `repair2/out_jstats.txt` | seconds |
| §7 Veronese contraction: seeded randomized checks of the contraction identity | `repair2/veronese.py` | `repair2/out_veronese.txt` | minutes |
| T-closure / one-slot closure exploration | `repair1/run_T.py` | `repair1/out_T.txt` | minutes |
| Memory calibration for the long runs (no mathematical content) | `repair2/calib.py` | `repair2/out_calib.txt` | seconds |

## Known data caveats (stated in the paper, §8)

- The quartic sweep enumerates **coefficient-one** sums of at most three
  distinct monomials (7175 forms), with **no GL-orbit reduction**.
- `repair1/out_hunt.txt` is an **incomplete** run: the `(d=4, n=4)` block
  is complete (156/7175 certified); the `(d=4, n=5)` block terminated
  early at 151 accumulated certified forms (host interruption), with
  zero gate failures among them. `out_hunt_snapshot.txt` is an earlier
  mid-run snapshot retained for provenance.
- All 307 certified forms pass Nil and T; no Jordan-closure failure was
  observed anywhere in the corpus.

## Reproducing

```
pip install -r requirements.txt   # sympy 1.12
python3 verify_all.py             # fast tier: regenerate + byte-compare
python3 verify_all.py --list      # show all tiers
```

`MANIFEST.sha256` pins every file in this bundle.  Code license:
Apache-2.0 (see `LICENSE`).
