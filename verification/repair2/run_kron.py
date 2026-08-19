"""PART 2(d) -- the three gates on the ACTUAL Kronecker products repair1 built,
with N_h constructed FROM THE PRODUCT TENSOR h = f box g DIRECTLY (polarized
Hessian of h in the D = d1*d2 variables z_ij), never from the factors'
triangularizing bases.  These are the spaces where T was INHERITED (memo
Theorem 6) rather than proved, so they are the sharpest test of JC.

Two routes to lambda_h(I) = P_h(I)/P_h(base):
  [SYM]  full symbolic GO certificate for h  (det of the D x D polarized
         Hessian factored as a perfect D-th power).  Then lambda comes from
         P_h alone and the TR gate is an INDEPENDENT test.
  [DEC]  when [SYM] is out of budget: lambda_h(I) = lambda_f(I_V)*lambda_g(I_W)
         on the decomposable basis tuples z_{i j}, i.e. the paper's defining
         formula for P_h.  Every such value is then VERIFIED pointwise by
         charpoly(M_h(I)) == (t - lambda)^D, which pins lambda intrinsically as
         the unique eigenvalue of M_h(I) (so lambda = tr M_h(I)/D and the TR gate
         is, on this route, IMPLIED by the pointwise verification rather than
         independent -- stated as such in the memo).
Both routes give the same N_h whenever both run; that is checked where possible.
"""
import itertools, os, signal, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gates import (nspace_diag, run_gates, diagonal_base, evec)   # noqa: E402
from nspace import certificate, subs_tuple, check_nil, check_T    # noqa: E402
from klib import kron, flat_Z                                     # noqa: E402


class TO(Exception):
    pass


def _to(sig, frm):
    raise TO()


x = sp.symbols('x0:4')
y = sp.symbols('y0:4')
W4f = x[0]**3 * x[1]
W4g = y[0]**3 * y[1]
F34f = x[0]**2 * (x[0] * x[2] + x[1]**2)
F34g = y[0]**2 * (y[0] * y[2] + y[1]**2)
W3f = x[0]**2 * x[1]
CCg = y[0]**2 * y[3] + y[0] * y[1] * y[2] + y[1]**3

CASES = [
    ('W_4 x W_4      (Sym^4 C^4)', W4f, list(x[:2]), W4g, list(y[:2]), 4),
    ('F3_4 x W_4     (Sym^4 C^6)', F34f, list(x[:3]), W4g, list(y[:2]), 4),
    ('W_3 x CC       (Sym^3 C^8)', W3f, list(x[:2]), CCg, list(y[:4]), 3),
    ('F3_4 x F3_4    (Sym^4 C^9)', F34f, list(x[:3]), F34g, list(y[:3]), 4),
]

SYM_BUDGET = int(os.environ.get('SYM_BUDGET', '900'))   # seconds for the [SYM] route

print('=' * 78)
print('PART 2(d)  TR / JC / CK on the Kronecker products, built from h directly')
print('=' * 78)

rows = []
for name, f, xs, g, ys, n in CASES:
    d1, d2, D, r = len(xs), len(ys), len(xs) * len(ys), n - 2
    print(f'\n### {name}   n={n}  d1={d1} d2={d2}  D={D}  r={r}')
    Z = [[sp.Symbol(f'z{i}_{j}') for j in range(d2)] for i in range(d1)]
    zs = flat_Z(Z)
    h = sp.expand(kron(f, xs, g, ys, n, Z))
    print(f'    h has {len(sp.Add.make_args(h))} monomials in {D} variables')

    # ---- factor certificates + diagonal bases (needed for the [DEC] route)
    rf, svf, Hf, cf, Qf = certificate(f, xs, n)
    rg, svg, Hg, cg, Qg = certificate(g, ys, n)
    Bf, HfB = diagonal_base(Hf, svf, d1, r)
    Bg, HgB = diagonal_base(Hg, svg, d2, r)
    bf = [k for k in range(d1) if Bf[0][k] == 1][0]
    bg = [k for k in range(d2) if Bg[0][k] == 1][0]
    QfB = subs_tuple(Qf, svf, Bf)
    QgB = subs_tuple(Qg, svg, Bg)

    def lam(I, svf=svf, svg=svg, Qf=Qf, Qg=Qg, QfB=QfB, QgB=QgB, d1=d1, d2=d2):
        iv = [a // d2 for a in I]
        iw = [a % d2 for a in I]
        Av = [evec(t, d1) for t in iv]
        Aw = [evec(t, d2) for t in iw]
        return sp.nsimplify(sp.cancel(subs_tuple(Qf, svf, Av) / QfB)
                            * sp.cancel(subs_tuple(Qg, svg, Aw) / QgB))

    base_vec = evec(bf * d2 + bg, D)

    # ---- [SYM] route, budgeted
    sym_info = None
    signal.signal(signal.SIGALRM, _to)
    signal.alarm(SYM_BUDGET)
    try:
        sym_info = nspace_diag(h, zs, n, base_vec=base_vec, verify_pointwise=False)
        signal.alarm(0)
        print(f'    [SYM] full symbolic certificate for h : OK   '
              f"(base z_{bf}{bg}, dim N_h = {len(sym_info['basis'])})")
    except TO:
        signal.alarm(0)
        print(f'    [SYM] full symbolic certificate for h : TIMEOUT at {SYM_BUDGET}s -- using [DEC]')
    except Exception as e:
        signal.alarm(0)
        print(f'    [SYM] full symbolic certificate for h : {type(e).__name__}: {e} -- using [DEC]')

    # ---- [DEC] route (always run)
    dec_info = nspace_diag(h, zs, n, lam_override=lam, base_vec=base_vec,
                           verify_pointwise=True)
    print(f"    [DEC] lambda from P_h on decomposables; pointwise "
          f"charpoly(M_h(I)) == (t-lambda)^{D} at all {D**r} basis tuples : "
          f"{dec_info['pointwise_charpoly']}")

    if sym_info is not None:
        A = sp.Matrix([list(m) for m in sym_info['basis']]).T
        B = sp.Matrix([list(m) for m in dec_info['basis']]).T
        agree = (A.rank() == B.rank() == A.row_join(B).rank())
        same_lams = all(sp.expand(sym_info['lams'][I] - dec_info['lams'][I]) == 0
                        for I in sym_info['lams'])
        print(f'    [SYM] vs [DEC]: same N-space {agree}; identical lambdas {same_lams}')

    info = sym_info if sym_info is not None else dec_info
    gg = run_gates(info)
    nil, _ = check_nil(info['basis'], D)
    T, step = check_T(info['basis'], D)
    print(f"    dim N_h = {gg['dimN']}   TR={gg['TR']}  JC={gg['JC']}  "
          f"CK={gg['CK']} (dim {gg['ck_dim']})   Nil={nil}  T={T}"
          f"   [route {'SYM' if sym_info is not None else 'DEC'}]")
    if not gg['JC']:
        print('    !!!!!! JC FAILURE !!!!!!')
        for pair, S, res in gg['jc_fail']:
            print(f'      pair {pair}: S = {S.tolist()}')
            print(f'      residual = {None if res is None else res.T.tolist()}')
    npairs = len(info['basis']) * (len(info['basis']) + 1) // 2
    print(f'    Jordan pairs tested: {npairs}')
    rows.append(dict(name=name, n=n, D=D, r=r, dimN=gg['dimN'], TR=gg['TR'],
                     JC=gg['JC'], CK=gg['CK'], ckdim=gg['ck_dim'], Nil=nil, T=T,
                     route='SYM' if sym_info is not None else 'DEC', pairs=npairs))

print('\n\n=========== SUMMARY (d) ===========')
print(f'{"product":30s} {"n":>2s} {"D":>2s} {"r":>2s} {"dimN":>4s} {"pairs":>5s} '
      f'{"TR":>5s} {"JC":>5s} {"CK":>5s} {"Nil":>5s} {"T":>5s} {"route":>5s}')
for r in rows:
    print(f'{r["name"][:30]:30s} {r["n"]:2d} {r["D"]:2d} {r["r"]:2d} {r["dimN"]:4d} '
          f'{r["pairs"]:5d} {str(r["TR"]):>5s} {str(r["JC"]):>5s} {str(r["CK"]):>5s} '
          f'{str(r["Nil"]):>5s} {str(r["T"]):>5s} {r["route"]:>5s}')
print(f'\nTALLY (d): {len(rows)} products; TR {sum(1 for r in rows if r["TR"])}/{len(rows)}; '
      f'JC {sum(1 for r in rows if r["JC"])}/{len(rows)}; '
      f'CK {sum(1 for r in rows if r["CK"])}/{len(rows)}; '
      f'Jordan pairs tested {sum(r["pairs"] for r in rows)}')
