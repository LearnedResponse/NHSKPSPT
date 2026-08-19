"""HUNT: is there a persistent f with r = n-2 >= 2 for which Nil FAILS?
(Such an f would show the repaired theorem's hypothesis is not vacuous-strong,
and, if the determinant identity still held there, that T is sufficient-not-
necessary.  It would also answer the generalized appendix question negatively.)

Cheap filter first: GO condition (d) -- persistent => Hess(f) = g^d.  For d<=4
GO Thm 5 gives Hess(f) = lambda * ell^{d(n-2)}.  We enumerate small-support
forms, filter on 'Hess(f) is a perfect d-th power', then run the full
polarized certificate + Nil/T test."""
import sys, os, itertools
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nspace import nspace, check_nil, check_T, certificate

def monomials(xs, n):
    out = []
    d = len(xs)
    for c in itertools.combinations_with_replacement(range(d), n):
        m = sp.Integer(1)
        for i in c: m *= xs[i]
        out.append(sp.expand(m))
    return sorted(set(out), key=str)

def hess_is_power(f, xs):
    d = len(xs)
    H = sp.Matrix(d, d, lambda i,j: sp.diff(f, xs[i], xs[j]))
    Dt = sp.expand(H.det(method='berkowitz'))
    if Dt == 0: return None
    c, facs = sp.factor_list(Dt)
    if all(e % d == 0 for b,e in facs):
        return sp.prod([b**(e//d) for b,e in facs])
    return None

found = []
for (d, n, maxsupp) in [(4,4,3), (4,5,3), (5,4,3)]:
    xs = list(sp.symbols(f'x0:{d}'))
    mons = monomials(xs, n)
    print(f"\n### scan d={d} n={n}: {len(mons)} monomials, supports of size <= {maxsupp}")
    cnt = 0; hits = 0
    for k in range(1, maxsupp+1):
        for supp in itertools.combinations(mons, k):
            f = sp.expand(sum(supp))
            cnt += 1
            gpow = hess_is_power(f, xs)
            if gpow is None: continue
            # need genuinely d-variable (Hessian nondegenerate direction) + persistent
            try:
                r, sv, H, c, Q = certificate(f, xs, n)
            except Exception:
                continue
            hits += 1
            info = nspace(f, xs, n, verbose=False)
            nil, cp = check_nil(info['gens'], d)
            T, step = check_T(info['gens'], d)
            found.append((d, n, str(f), len(info['gens']), nil, T))
            print(f"   PERSISTENT: f = {f}   dimNsp={len(info['gens'])} Nil={nil} T={T}"
                  + ("   <<<<<< Nil FAILS" if not nil else "")
                  + ("   <<<<<< T FAILS" if nil and not T else ""))
    print(f"   scanned {cnt} forms, {hits} passed the persistence certificate")

print("\n\n=========== HUNT SUMMARY ===========")
bad = [r for r in found if not r[4] or not r[5]]
print(f"persistent forms found: {len(found)};  with Nil or T failing: {len(bad)}")
for r in bad: print("   ", r)
