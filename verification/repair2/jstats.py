"""PART 3b support -- structure of the Jordan closure, and HOW MANY exact scalar
equations the JC gate actually imposes (i.e. how unexpected the pass is).

For every generator N of N_f built at a diagonal base B:
  * G := H_f(B) is symmetric and M = G^{-1}H(A), so  G M = H(A) = H(A)^T = M^T G:
    every element of N_f is G-SELF-ADJOINT.  Hence so is every Jordan product.
  * under Nil the trace form vanishes: tr(XY) = 0 for X,Y in N_f  (verified).
So each Jordan product is constrained a priori only to
    S := { G-self-adjoint, trace 0 },   dim S = d(d+1)/2 - 1,
and JC asserts it lands in the dim-(dim N_f) subspace N_f.  Each unordered pair
therefore imposes  dim S - dim N_f  independent scalar equations over Q.
"""
import itertools, os, re, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gates import nspace_diag, run_gates, named_cases          # noqa: E402

HUNT = os.path.join(os.path.dirname(HERE), 'repair1', 'out_hunt.txt')

print('=' * 78)
print('PART 3b SUPPORT -- self-adjointness, trace form, squaring-closure, and the')
print('                  exact equation count behind each JC pass')
print('=' * 78)


def analyse(name, f, xs, n):
    info = nspace_diag(f, xs, n)
    if not info['ok']:
        return None
    d, basis = info['d'], info['basis']
    G = info['H']
    from nspace import subs_tuple
    GB = sp.Matrix(subs_tuple(G, info['slotvars'], info['base']))
    selfadj = all(sp.expand(GB * N - (GB * N).T).is_zero_matrix for N in basis)
    traceform = all(sp.expand((basis[i] * basis[j]).trace()) == 0
                    for i in range(len(basis)) for j in range(len(basis)))
    # squaring closure on a generic element (equivalent to JC by polarization)
    cs = sp.symbols(f'c0:{len(basis)}')
    X = sp.zeros(d, d)
    for c, N in zip(cs, basis):
        X += c * N
    X2 = sp.expand(X * X)
    B = sp.Matrix([list(m) for m in basis]).T      # rational entries, full col rank
    v = sp.Matrix(d * d, 1, list(X2))              # symbolic in c0..c_{k-1}
    # exact projection onto the span via the (rational) normal equations; the
    # residual is then a symbolic vector that must vanish identically.
    coef = (B.T * B).inv() * (B.T * v)
    sq_ok = sp.expand(v - B * coef).is_zero_matrix
    sq_coeffs = sp.simplify(coef.T)
    dimS = d * (d + 1) // 2 - 1
    k = len(basis)
    npairs = k * (k + 1) // 2
    eqs = npairs * (dimS - k)
    return dict(name=name, d=d, r=info['r'], dimN=k, selfadj=selfadj,
                traceform=traceform, sq=sq_ok, dimS=dimS, npairs=npairs, eqs=eqs,
                sq_coeffs=sq_coeffs)


print('\n--- ten named forms ---')
tot = 0
for name, f, xs, n in named_cases():
    a = analyse(name, f, xs, n)
    print(f'  {a["name"][:38]:38s} d={a["d"]} r={a["r"]} dimN={a["dimN"]} '
          f'G-self-adjoint={a["selfadj"]} traceform=0:{a["traceform"]} '
          f'squaring-closed(generic)={a["sq"]}  dim S={a["dimS"]} pairs={a["npairs"]} '
          f'equations={a["eqs"]}')
    print(f'      generic X = sum c_k N_k  =>  X^2 = {list(a["sq_coeffs"])} . (N_1..N_{a["dimN"]})')
    tot += a['eqs']
print(f'  named-form equation total: {tot}')

print('\n--- equation count for the hunt blocks (from the recorded dim N_f) ---')
for want in [(4, 4), (4, 5)]:
    blk, dims = None, []
    for L in open(HUNT):
        m = re.match(r'### scan d=(\d+) n=(\d+)', L)
        if m:
            blk = (int(m.group(1)), int(m.group(2)))
        m = re.match(r'\s+PERSISTENT: f = .*?dimNsp=(\d+)', L)
        if m and blk == want:
            dims.append(int(m.group(1)))
    d = want[0]
    dimS = d * (d + 1) // 2 - 1
    tot = sum((k * (k + 1) // 2) * (dimS - k) for k in dims)
    from collections import Counter
    print(f'  (d,n)={want}: {len(dims)} forms, dim N distribution {dict(sorted(Counter(dims).items()))}, '
          f'dim S={dimS}; equations = ' +
          ' + '.join(f'{c}*({k*(k+1)//2}*({dimS}-{k}))' for k, c in sorted(Counter(dims).items())) +
          f' = {tot}')

print('\n--- Kronecker products (dim N_h from out_kron_gates.txt) ---')
for nm, D, k in [('W_4 x W_4', 4, 3), ('F3_4 x W_4', 6, 5), ('W_3 x CC', 8, 7),
                 ('F3_4 x F3_4', 9, 8)]:
    dimS = D * (D + 1) // 2 - 1
    np_ = k * (k + 1) // 2
    print(f'  {nm:14s} D={D} dimN={k} dim S={dimS} pairs={np_} equations={np_*(dimS-k)}')
