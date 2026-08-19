"""Global (non-Segre) probe of Prop 10's identity (22).

For persistent f in Sym^n C^{d1}, g in Sym^n C^{d2} we
  1. extract P_f, P_g from the polarized Hessian certificate (Thm 3 / GO Thm 2(c));
  2. build P_{f|X|g} by the UNIVERSAL-PROPERTY extension the paper uses
     (P_{f|X|g}(v1(x)w1,...) := (1/n!) P_f P_g, extended multilinearly);
  3. evaluate BOTH sides of (22) at random NON-DECOMPOSABLE polarization tuples
     U^(1),...,U^(r) in V(x)W and compare exactly over Q.

Failure at a random rational point would disprove the THEOREM.
Success is evidence for the theorem; it says nothing about the proof.
"""
import itertools, random, sympy as sp
from sympy import factorial, Rational
from klib import kron, flat_Z, polarize, hess_det


def cert(f, xs, n, pre='a'):
    """return (c, Q) with Hess(f_{v^(1..r)}(x)) = c * Q^d, Q multilinear rational,
       i.e. P_f = c^(1/d) * Q."""
    d, r = len(xs), n - 2
    V = [[sp.Symbol(f'{pre}{k}_{i}') for i in range(d)] for k in range(r)]
    fp = polarize(f, xs, V)
    _, D = hess_det(fp, xs)
    D = sp.expand(D)
    if D == 0:
        raise ValueError("polarized Hessian identically zero")
    c, facs = sp.factor_list(D)
    Q = 1
    for b, e in facs:
        if e % d:
            raise ValueError(f"NOT a perfect d={d} power: exponent {e} on {b}")
        Q *= b**(e // d)
    Q = sp.expand(Q)
    # verify multidegree (1,...,1)
    for k in range(r):
        for mono in sp.Poly(Q, *V[k]).monoms():
            assert sum(mono) == 1, f"P not multilinear in slot {k}: {Q}"
    assert sp.expand(c * Q**d - D) == 0
    return c, Q, V


def extend(Qf, Vv, Qg, Vw, U):
    """universal-property (multilinear) extension of Qf(v)*Qg(w) to (V(x)W)^{xr}."""
    r = len(Vv)
    prod_ = sp.expand(Qf * Qg)
    out = 0
    for term in sp.Add.make_args(prod_):
        coeff, rest = term.as_coeff_Mul()
        pw = rest.as_powers_dict()
        idx = []
        for k in range(r):
            i = [t for t in range(len(Vv[k])) if Vv[k][t] in pw]
            j = [t for t in range(len(Vw[k])) if Vw[k][t] in pw]
            assert len(i) == 1 and len(j) == 1
            idx.append((i[0], j[0]))
        m = coeff
        for k, (i, j) in enumerate(idx):
            m *= U[k][i][j]
        out += m
    return sp.expand(out)


def probe(name, f, xs, g, ys, n, trials=6, seed=0, symbolic=False):
    d1, d2, r = len(xs), len(ys), n - 2
    D = d1 * d2
    print(f"\n### {name}:  n={n}, d1={d1}, d2={d2}, r={r}, D=d1*d2={D}")
    cf, Qf, Vv = cert(f, xs, n, pre='a')
    cg, Qg, Vw = cert(g, ys, n, pre='b')       # distinct symbols: no name collision
    print(f"  P_f = ({cf})^(1/{d1}) * {Qf}")
    print(f"  P_g = ({cg})^(1/{d2}) * {Qg}")

    Z = [[sp.Symbol(f'z{i}_{j}') for j in range(d2)] for i in range(d1)]
    zs = flat_Z(Z)
    h = kron(f, xs, g, ys, n, Z)

    U = [[[sp.Symbol(f'U{k}_{i}_{j}') for j in range(d2)] for i in range(d1)]
         for k in range(r)]
    Ext = extend(Qf, Vv, Qg, Vw, U)
    # [P_{f|X|g}]^D  =  (cf^d2 * cg^d1 / n!^D) * Ext^D      (all rational)
    pref = Rational(cf**d2 * cg**d1) / factorial(n)**D
    print(f"  P_{{f|X|g}}(U) = ({sp.nsimplify(pref)})^(1/{D}) * ({Ext})")

    rng = random.Random(seed)
    ok = True
    for t in range(trials):
        sub = {}
        pts = []
        for k in range(r):
            M = sp.Matrix(d1, d2, lambda i, j: sp.Integer(rng.randint(-5, 5)))
            while M.rank() < min(d1, d2):          # force NON-decomposable (rank>1)
                M = sp.Matrix(d1, d2, lambda i, j: sp.Integer(rng.randint(-5, 5)))
            pts.append(M)
            for i in range(d1):
                for j in range(d2):
                    sub[U[k][i][j]] = M[i, j]
        hp = polarize(h, zs, [[sub[U[k][i][j]] for i in range(d1) for j in range(d2)]
                              for k in range(r)])
        _, lhs = hess_det(hp, zs)
        lhs = sp.expand(lhs)
        rhs = sp.expand(pref * Ext.subs(sub)**D)
        match = sp.simplify(lhs - rhs) == 0
        ok &= match
        print(f"    trial {t}: ranks={[M.rank() for M in pts]} (>1 => NON-decomposable)"
              f"  LHS={lhs}  RHS={rhs}  MATCH={match}")
    print(f"  ==> all non-decomposable trials match: {ok}")

    if symbolic:
        Uflat = [[U[k][i][j] for i in range(d1) for j in range(d2)] for k in range(r)]
        hp = polarize(h, zs, Uflat)
        _, lhs = hess_det(hp, zs)
        eq = sp.expand(sp.expand(lhs) - sp.expand(pref * Ext**D))
        print(f"  FULL SYMBOLIC identity (22) holds identically on (V(x)W)^x{r}: {eq == 0}")
        if eq != 0:
            print("  difference:", sp.factor(eq))
    return ok
