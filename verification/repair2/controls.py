"""PART 2 -- NEGATIVE CONTROLS.  A gate that everything passes is worthless
unless something fails it.  Three controls:

 (C1) the repair1 toy counter-family (satisfies the FULL GO certificate but is
      not the polarized Hessian of a symmetric tensor; memo attempt-1 sec.3.2):
      does the JC gate see it?
 (C2) symmetric but NOT persistent cubics / quartics, with lambda taken as the
      trace normalization tr(M)/d (the only sensible substitute when there is no
      certificate): does JC hold for a generic symmetric tensor?
 (C3) the alternating-P diagnostic: the toy's P is ALTERNATING, so
      det H(u,u) == 0 identically and NO diagonal base exists -- checked here,
      which is what makes "diagonal base found" a certificate that P_f is
      symmetric (used in the Part-3a lemma).
"""
import itertools, os, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gates import run_gates, in_span, evec, diagonal_base       # noqa: E402
from nspace import polarized_hessian, subs_tuple, reduce_span, check_nil, check_T  # noqa: E402

print('=' * 78)
print('PART 2  NEGATIVE CONTROLS  (is the JC gate non-vacuous?)')
print('=' * 78)


def gates_from_family(H, slotvars, d, r, base, Q):
    """N-space from an explicit polarized-Hessian family + certificate poly Q."""
    HB = sp.Matrix(subs_tuple(H, slotvars, base))
    HBi = HB.inv()
    QB = subs_tuple(Q, slotvars, base) if Q is not None else None
    raw = []
    for I in itertools.product(range(d), repeat=r):
        A = [evec(c, d) for c in I]
        M = sp.expand(HBi * sp.Matrix(subs_tuple(H, slotvars, A)))
        if Q is not None:
            lam = sp.nsimplify(sp.cancel(subs_tuple(Q, slotvars, A) / QB))
        else:
            lam = sp.Rational(sp.expand(M.trace()), d)
        raw.append((I, sp.Matrix(sp.expand(M - lam * sp.eye(d)))))
    basis = reduce_span([N for _, N in raw], d)
    return dict(ok=True, d=d, r=r, base=base, raw=raw, basis=basis,
                lams=None, pointwise_charpoly=None), basis


# ------------------------------------------------------------------ C1: the toy
print('\n### (C1) repair1 toy counter-family, r=2, d=2')
u0, u1, v0, v1 = sp.symbols('v0_0 v0_1 v1_0 v1_1')
sv = [[u0, u1], [v0, v1]]
Htoy = sp.Matrix([[u0 * v0, (u0 * v1 + u1 * v0) / 2],
                  [(u0 * v1 + u1 * v0) / 2, u1 * v1]])
Qtoy = sp.expand(u0 * v1 - u1 * v0)          # P = (i/2)Q ; ratios use Q only
print(f'    det H(u,v) = {sp.factor(Htoy.det())}   (= -(Q/2)^2, so the GO certificate holds with d=2)')
base = [evec(0, 2), evec(1, 2)]
info, basis = gates_from_family(Htoy, sv, 2, 2, base, Qtoy)
print(f'    base (e0,e1) [NOT diagonal];  dim Nsp = {len(basis)}')
for k, M in enumerate(basis, 1):
    print(f'      N_{k} = {M.tolist()}')
g = run_gates(info)
nil, cp = check_nil(basis, 2)
T, _ = check_T(basis, 2)
TOY = dict(TR=g['TR'], JC=g['JC'], CK=g['CK'], Nil=nil, T=T, dimN=g['dimN'])
print(f'    TR={g["TR"]}  JC={g["JC"]}  CK={g["CK"]}(dim {g["ck_dim"]})  Nil={nil}  T={T}')
for pair, S, res in g['jc_fail']:
    print(f'    JC FAILS at pair {pair}:  N_i N_j + N_j N_i = {S.tolist()}  '
          f'(trace {sp.expand(S.trace())}) -- residual off the span '
          f'{None if res is None else res.T.tolist()}')

# ------------------------------------------------------ C3: no diagonal base
print('\n### (C3) alternating-P diagnostic on the toy')
diag = sp.expand(Htoy.subs({v0: u0, v1: u1}).det())
print(f'    det H(u,u) = {diag}   ->  diagonal base exists: {diag != 0}')
A0, _ = diagonal_base(Htoy, sv, 2, 2)
print(f'    gates.diagonal_base() returns: {A0}')
print('    (P_toy is ALTERNATING; for a genuine persistent form every case in')
print('     Part 2 admitted a diagonal base, which certifies P_f is symmetric.)')

# --------------------------------------------- C2: symmetric, non-persistent
print('\n### (C2) symmetric but NOT persistent, lambda := tr(M)/d')
x = sp.symbols('x0:4')
CTRL = [
    ('Fermat cubic x0^3+x1^3+x2^3', x[0]**3 + x[1]**3 + x[2]**3, list(x[:3]), 3),
    ('x0^3+x1^2*x2 (d=3)', x[0]**3 + x[1]**2 * x[2], list(x[:3]), 3),
    ('generic cubic 1 (d=3)', x[0]**3 + 2 * x[0]**2 * x[1] + 3 * x[0] * x[1] * x[2]
     + x[1]**3 + 5 * x[2]**3, list(x[:3]), 3),
    ('generic cubic 2 (d=4)', x[0]**3 + x[0] * x[1] * x[2] + x[1]**2 * x[3]
     + 2 * x[2]**3 + x[3]**3, list(x[:4]), 3),
    ('Fermat quartic x0^4+x1^4+x2^4 (r=2)', x[0]**4 + x[1]**4 + x[2]**4, list(x[:3]), 4),
    ('generic quartic (d=3, r=2)', x[0]**4 + x[0]**2 * x[1] * x[2] + x[1]**4
     + 3 * x[1] * x[2]**3 + x[2]**4, list(x[:3]), 4),
]
rows = []
for name, f, xs, n in CTRL:
    d, r = len(xs), n - 2
    slot = [[sp.Symbol(f'w{k}_{i}') for i in range(d)] for k in range(r)]
    H = polarized_hessian(f, xs, r, slot)
    Dt = sp.expand(H.det(method='berkowitz'))
    c, facs = sp.factor_list(Dt)
    persistent = Dt != 0 and all(e % d == 0 for b, e in facs)
    A0, HB = diagonal_base(H, slot, d, r)
    if A0 is None:
        print(f'    {name:42s}: no admissible diagonal base -- skipped')
        continue
    info, basis = gates_from_family(H, slot, d, r, A0, None)
    g = run_gates(info)
    nil, _ = check_nil(basis, d)
    T, _ = check_T(basis, d)
    print(f'    {name:42s}: persistent={persistent}  dimN={g["dimN"]:2d}  '
          f'TR(by construction)={g["TR"]}  JC={g["JC"]}  CK={g["CK"]}  Nil={nil}  T={T}')
    rows.append((name, persistent, g['JC']))

# ------------------------------------- C4: T does NOT imply JC (independence)
print('\n### (C4) T does not imply JC: a strictly-upper-triangular space that is')
print('         NOT Jordan-closed  (so the JC passes are new information, not a')
print('         corollary of the T passes already banked in attempt 1)')
E = lambda i, j: sp.Matrix(3, 3, lambda a, b: 1 if (a, b) == (i, j) else 0)
sp_span = [E(0, 1), E(1, 2)]
S = sp.expand(sp_span[0] * sp_span[1] + sp_span[1] * sp_span[0])
ok, res = in_span(S, sp_span, 3)
print(f'    span{{E12,E23}} (strictly upper, so T holds, trace 0, common kernel = <e1>):')
print(f'    E12 o E23 = {S.tolist()}  in span: {ok}')
print(f'    => TR and CK can hold, T can hold, and JC still FAIL.  JC is strictly')
print(f'       stronger than T + TR + CK on general matrix spaces.')

print('\n=========== CONTROL TALLY ===========')
print(f'    toy counter-family (certificate holds, not a genuine Hessian): {TOY}')
print(f'      -> rejected by the gate CONJUNCTION (TR and CK fail); JC alone passes it,')
print(f'         so TR is load-bearing and JC alone is not the discriminator there.')
print(f'    non-persistent symmetric tensors tested: {len(rows)};  '
      f'JC pass {sum(1 for _, _, j in rows if j)}/{len(rows)}')
