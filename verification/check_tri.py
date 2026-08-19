"""Appendix probe (Question 19 in instances): for persistent cubics f, is the
normalized Hessian space N_f(ker ell_f) simultaneously STRICTLY triangularizable?
If yes for one factor, Corollary 18 gives a GLOBAL proof of persistence of f|X|g
independent of Proposition 10 -- i.e. those test cases are covered anyway.

Test used: a linear space of matrices is simultaneously strictly triangularizable
iff the associative algebra it generates is nilpotent (A^N = 0 for some N)."""
import itertools, sympy as sp

x = sp.symbols('x0:5')


def normalized_hessian_space(f, xs):
    d = len(xs)
    H = sp.Matrix(d, d, lambda i, j: sp.diff(f, xs[i], xs[j]))
    Hess = sp.factor(sp.expand(H.det()))
    print("  Hess(f) =", Hess)
    c, facs = sp.factor_list(sp.expand(H.det()))
    assert len(facs) == 1 and facs[0][1] == d, f"not lambda*ell^d: {facs}"
    ell = facs[0][0]
    print("  ell_f =", ell, "  lambda =", c)
    # v0 with ell(v0)=1
    sol = None
    for i in range(d):
        e = [0]*d
        e[i] = 1
        if ell.subs(dict(zip(xs, e))) != 0:
            val = ell.subs(dict(zip(xs, e)))
            e = [sp.Rational(t, 1)/val for t in e]
            sol = e
            break
    H0 = H.subs(dict(zip(xs, sol)))
    H0inv = H0.inv()
    basis = []
    for i in range(d):          # N_f is linear; take images of a basis of ker ell
        pass
    # basis of ker ell
    ker = sp.Matrix([[sp.diff(ell, v) for v in xs]]).nullspace()
    for u in ker:
        Hu = H.subs(dict(zip(xs, list(u))))
        N = sp.expand(H0inv*Hu - ell.subs(dict(zip(xs, list(u))))*sp.eye(d))
        basis.append(sp.Matrix(N))
    return basis, d


def algebra_nilpotent(basis, d):
    """generate the associative algebra; return True if it is nilpotent."""
    cur = [sp.Matrix(b) for b in basis]
    for step in range(1, d + 2):
        if all(sp.expand(M).is_zero_matrix for M in cur):
            return True, step
        cur = [sp.expand(A*B) for A in cur for B in basis]
        # reduce to a spanning set to keep it small
        rows = sp.Matrix([list(M) for M in cur])
        rr, piv = rows.rref()
        cur = [sp.Matrix(d, d, list(rr.row(i))) for i in piv and range(len(piv))]
    return False, None


for name, f, d in [("W_3 = x0^2 x1 (d=2)", x[0]**2*x[1], 2),
                   ("C3 = x0^2 x2 + x0 x1^2 (d=3)", x[0]**2*x[2] + x[0]*x[1]**2, 3),
                   ("Chasles-Cayley x0^2x3+x0x1x2+x1^3 (d=4)",
                    x[0]**2*x[3] + x[0]*x[1]*x[2] + x[1]**3, 4),
                   ("quadric+tangent plane x0^2x3+x0x1x2 (d=4)",
                    x[0]**2*x[3] + x[0]*x[1]*x[2], 4)]:
    print(f"\n### {name}")
    B, dd = normalized_hessian_space(f, x[:d])
    print("  dim ker ell =", len(B))
    for N in B:
        print("   N =", N.tolist(), " nilpotent:", (N**dd).is_zero_matrix)
    nil, step = algebra_nilpotent(B, dd)
    print(f"  associative algebra nilpotent (=> simultaneously STRICTLY "
          f"triangularizable): {nil} (vanishes at word length {step})")
