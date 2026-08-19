"""repair2/gates.py -- Sol ROUTE-1 gates over the normalized Hessian space N_f.

Extends repair1/nspace.py (imported, NOT modified) with three exact gates,
computed at a DIAGONAL admissible base B = (b,...,b) (Sol response-3 erratum F4:
the contraction-lemma normalization uses a diagonal base):

    TR :  tr N_f(I) == 0   for every basis-tuple generator I
    JC :  N_I N_J + N_J N_I  in  span{N_K}   for every pair I,J   (Jordan closure)
    CK :  dim ( intersect_I ker N_I ) > 0    (common kernel)

All arithmetic exact over Q.  Two remarks on the reductions used:

  * TR is linear, so trace-zero on the d^r basis-tuple generators is equivalent
    to trace-zero on all of N_f.  We test the raw generators anyway (Sol's
    wording) and additionally the reduced span basis.
  * The Jordan product (X,Y) -> XY+YX is BILINEAR, so closure on a spanning set
    is equivalent to closure on the whole space.  We test all unordered pairs
    (i<=j) of a reduced basis of N_f -- this is exactly equivalent to, and much
    cheaper than, all pairs of the d^r raw generators.  Membership is decided by
    an exact rank comparison  rank[B] == rank[B | vec(S)]  over Q.

Diagonal base, and why its existence is itself a diagnostic: det H_f(v,...,v) is
the diagonal restriction of P_f(...)^d.  If P_f were ALTERNATING in its r slots
(the case memo attempt-1 sec.8.7 flags as not-ruled-out, and which the toy
counter-family realizes), that restriction vanishes identically and NO diagonal
base exists.  So "diagonal base found" certifies P_f is not alternating.
"""
import itertools, os, sys

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.dirname(HERE)
REPAIR1 = os.path.join(OUTDIR, 'repair1')
sys.path.insert(0, REPAIR1)
sys.path.insert(0, OUTDIR)

from nspace import (polarized_hessian, certificate, subs_tuple, reduce_span,
                    check_nil, check_T)                      # noqa: E402
from klib import kron, flat_Z                                 # noqa: E402


# ----------------------------------------------------------------- base tuples
def diagonal_base(H, slotvars, d, r, base_vec=None):
    """first DIAGONAL admissible base B=(b,..,b): standard basis vectors, then
       small integer vectors.  Returns (A, HB) or (None, None).
       base_vec, if given, is tried first (used to pin the product base to
       z_{b_f b_g} so that an externally supplied lambda is the right ratio)."""
    cands = []
    if base_vec is not None:
        cands.append([sp.Integer(t) for t in base_vec])
    cands += [[sp.Integer(1) if i == c else sp.Integer(0) for i in range(d)]
              for c in range(d)]
    for combo in itertools.product([0, 1, -1, 2], repeat=d):
        if any(combo):
            cands.append([sp.Integer(t) for t in combo])
    for b in cands:
        A = [list(b) for _ in range(r)]
        HB = sp.Matrix(subs_tuple(H, slotvars, A))
        if sp.expand(HB.det()) != 0:
            return A, HB
    return None, None


def evec(i, d):
    return [sp.Integer(1) if k == i else sp.Integer(0) for k in range(d)]


# ------------------------------------------------------- N-space at a diag base
def nspace_diag(f, xs, n, lam_override=None, verify_pointwise=True, base_vec=None):
    """Generators of N_f at a DIAGONAL base, indexed by the basis tuple I.

    lam_override: optional callable I -> lambda_f(I) used INSTEAD of the symbolic
    certificate ratio (used for Kronecker products where the symbolic det of the
    D x D polarized Hessian is out of budget).  When used, every value is checked
    pointwise against charpoly(M(I)) == (t - lambda)^D, which is a strictly
    stronger verification than det M(I) == lambda^D.
    """
    d = len(xs)
    r = n - 2
    if lam_override is None:
        r_, slotvars, H, c, Q = certificate(f, xs, n)
        assert r_ == r
    else:
        slotvars = [[sp.Symbol(f'v{k}_{i}') for i in range(d)] for k in range(r)]
        H = polarized_hessian(f, xs, r, slotvars)
        Q = None
    A0, HB = diagonal_base(H, slotvars, d, r, base_vec=base_vec)
    if A0 is None:
        return dict(ok=False, why='no diagonal admissible base', d=d, r=r)
    HBi = HB.inv()
    QB = subs_tuple(Q, slotvars, A0) if Q is not None else None
    raw = []          # (I, N_I)
    lams = {}
    pointwise = True
    for I in itertools.product(range(d), repeat=r):
        A = [evec(cc, d) for cc in I]
        HA = sp.Matrix(subs_tuple(H, slotvars, A))
        M = sp.expand(HBi * HA)
        if Q is not None:
            lam = sp.nsimplify(sp.cancel(subs_tuple(Q, slotvars, A) / QB))
        else:
            lam = lam_override(I)
        if verify_pointwise:
            t = sp.Symbol('t')
            cp = sp.expand(M.charpoly(t).as_expr())
            if sp.expand(cp - (t - lam)**d) != 0:
                pointwise = False
        lams[I] = lam
        N = sp.expand(M - lam * sp.eye(d))
        raw.append((I, sp.Matrix(N)))
    basis = reduce_span([N for _, N in raw], d)
    return dict(ok=True, d=d, r=r, base=A0, raw=raw, basis=basis, lams=lams,
                Q=Q, H=H, slotvars=slotvars, pointwise_charpoly=pointwise)


# --------------------------------------------------------------- the three gates
def in_span(S, basis, d):
    """exact membership of S in span(basis) over Q; returns (bool, coeffs|residual)."""
    if not basis:
        return S.is_zero_matrix, None
    B = sp.Matrix([list(m) for m in basis]).T          # (d*d) x k, row-major vec
    s = sp.Matrix(d * d, 1, list(S))
    aug = B.row_join(s)
    if B.rank() == aug.rank():
        sol = B.solve_least_squares(s) if B.rows != B.cols else None
        try:
            coeffs = sp.Matrix(sp.linsolve((B, s)).args[0]) if sp.linsolve((B, s)) else None
        except Exception:
            coeffs = sol
        return True, coeffs
    # inconsistent: exact residual orthogonal to the span (Gram normal equations)
    G = (B.T * B)
    rhs = B.T * s
    try:
        c = G.solve(rhs)
        res = sp.expand(s - B * c)
    except Exception:
        res = None
    return False, res


def run_gates(info, jordan_on_raw=False):
    """TR / JC / CK on an nspace_diag result."""
    d = info['d']
    raw, basis = info['raw'], info['basis']
    # --- TR
    tr_raw = [(I, sp.expand(N.trace())) for I, N in raw]
    TR = all(t == 0 for _, t in tr_raw)
    tr_bad = [(I, t) for I, t in tr_raw if t != 0]
    TR_basis = all(sp.expand(N.trace()) == 0 for N in basis)
    # --- JC
    JC = True
    jc_fail = []
    jc_struct = []
    pairs = list(itertools.combinations_with_replacement(range(len(basis)), 2))
    for i, j in pairs:
        S = sp.expand(basis[i] * basis[j] + basis[j] * basis[i])
        ok, dat = in_span(S, basis, d)
        if not ok:
            JC = False
            jc_fail.append(((i, j), S, dat))
        else:
            jc_struct.append(((i, j), None if S.is_zero_matrix else dat))
    if jordan_on_raw and JC:
        # redundant by bilinearity; run as an independent confirmation
        for a in range(len(raw)):
            for b in range(a, len(raw)):
                S = sp.expand(raw[a][1] * raw[b][1] + raw[b][1] * raw[a][1])
                ok, dat = in_span(S, basis, d)
                if not ok:
                    JC = False
                    jc_fail.append((('raw', raw[a][0], raw[b][0]), S, dat))
    # --- CK
    if basis:
        stack = sp.Matrix.vstack(*basis)
        ck = stack.nullspace()
    else:
        ck = [sp.eye(d).col(i) for i in range(d)]
    CK = len(ck) > 0
    return dict(TR=TR, TR_basis=TR_basis, tr_bad=tr_bad, JC=JC, jc_fail=jc_fail,
                jc_struct=jc_struct, CK=CK, ck_dim=len(ck), dimN=len(basis))


def report_case(name, f, xs, n, verbose=True, lam_override=None, do_nilT=True,
                base_vec=None):
    try:
        info = nspace_diag(f, xs, n, lam_override=lam_override, base_vec=base_vec)
    except Exception as e:
        if verbose:
            print(f'  !! {type(e).__name__}: {e}')
        return dict(name=name, n=n, d=len(xs), error=f'{type(e).__name__}: {e}')
    if not info['ok']:
        if verbose:
            print(f"  !! {info['why']}")
        return dict(name=name, n=n, d=len(xs), error=info['why'])
    g = run_gates(info)
    row = dict(name=name, n=n, d=info['d'], r=info['r'], dimN=g['dimN'],
               TR=g['TR'], JC=g['JC'], CK=g['CK'], ckdim=g['ck_dim'],
               base=info['base'][0], pw=info['pointwise_charpoly'])
    if do_nilT:
        nil, _ = check_nil(info['basis'], info['d'])
        T, step = check_T(info['basis'], info['d'])
        row['Nil'], row['T'] = nil, T
    if verbose:
        print(f"  d={row['d']} r={row['r']} dimN={row['dimN']} diag-base b={[str(t) for t in info['base'][0]]}"
              f"  TR={row['TR']} JC={row['JC']} CK={row['CK']}(dim {row['ckdim']})"
              + (f" Nil={row.get('Nil')} T={row.get('T')}" if do_nilT else '')
              + (f"  [pointwise charpoly {info['pointwise_charpoly']}]" if lam_override else ''))
        if not g['JC']:
            print('  !!!!!! JC FAILURE !!!!!!')
            for pair, S, res in g['jc_fail']:
                print(f'    pair {pair}:  N_i N_j + N_j N_i = {S.tolist()}')
                print(f'    residual off the span = {None if res is None else res.T.tolist()}')
        if not g['TR']:
            print(f"  !!!!!! TR FAILURE: {[(I, str(t)) for I, t in g['tr_bad'][:5]]}")
    row['_info'] = info
    row['_gates'] = g
    return row


NAMED = None


def named_cases():
    x = sp.symbols('x0:6')
    cases = []
    for n in [3, 4, 5, 6]:
        cases.append((f'W_{n} = x0^{n-1}*x1', x[0]**(n - 1) * x[1], list(x[:2]), n))
    for n in [3, 4, 5]:
        cases.append((f'F3_{n} = x0^{n-2}(x0*x2+x1^2)',
                      x[0]**(n - 2) * (x[0] * x[2] + x[1]**2), list(x[:3]), n))
    cases.append(('CC = x0^2x3+x0x1x2+x1^3',
                  x[0]**2 * x[3] + x[0] * x[1] * x[2] + x[1]**3, list(x[:4]), 3))
    cases.append(('QT = x0^2x3+x0x1x2', x[0]**2 * x[3] + x[0] * x[1] * x[2], list(x[:4]), 3))
    cases.append(('GOEx28 = x0x2^3+x1x2^2x3+x3^4',
                  x[0] * x[2]**3 + x[1] * x[2]**2 * x[3] + x[3]**4, list(x[:4]), 4))
    return cases
