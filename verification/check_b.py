"""(b) W_3 |X| W_3 in Sym^3 C^4.  n=3 so r=n-2=1: ONE polarization slot U.
    Compute the polarized Hessian determinant at a fully GENERIC (hence
    non-decomposable) U and test whether it is a perfect d1*d2 = 4th power of a
    multilinear (here: linear) form in U.  Then (c) compare the root against
    P_{f|X|g} as constructed in the paper's Prop 10.
    NOTE: for n=3 the r-fold polarized Hessian IS the ordinary Hessian matrix
    evaluated at the point U, up to the restitution constant."""
import sympy as sp
from klib import kron, flat_Z, polarize, hess_det

x0, x1, y0, y1 = sp.symbols('x0 x1 y0 y1')
Z = [[sp.Symbol(f'z{i}{j}') for j in range(2)] for i in range(2)]
zs = flat_Z(Z)
U = [sp.Symbol(f'U{i}{j}') for i in range(2) for j in range(2)]   # U00 U01 U10 U11

f = x0**2 * x1          # W_3
g = y0**2 * y1          # W_3
h = kron(f, [x0, x1], g, [y0, y1], 3, Z)
print("W_3 |X| W_3 =", sp.expand(h))
print("  (in Sym^3 C^4, coords z00 z01 z10 z11)")

# r = 1 polarization at a fully generic U  (NOT decomposable: U is 4 free params,
# the Segre cone {v(x)w} is the 3-dim quadric cone U00*U11 - U01*U10 = 0)
hU = polarize(h, zs, [U])
HM, D = hess_det(hU, zs)
D = sp.expand(D)
print("\npolarized Hessian matrix H (entries linear in U):")
sp.pprint(HM)
print("\nHess( (f|X|g)_U (z) )  as a polynomial in U00,U01,U10,U11:")
print("  ", sp.factor(D))
print("  total degree in U:", sp.Poly(D, *U).total_degree(), "(expected d1*d2 = 4)")

# --- perfect 4th power test ---
fl = sp.factor_list(D)
print("\nfactor_list:", fl)
is4 = all(e % 4 == 0 for _, e in fl[1])
print("PERFECT 4th POWER of a multilinear form? ", is4)
root = fl[0]**sp.Rational(1, 4) * sp.prod([b**sp.Rational(e, 4) for b, e in fl[1]])
print("  4th root (up to 4th root of unity):", sp.nsimplify(root))

# --- (c) compare with P_{f|X|g} built by the paper's recipe ---
# P_f(v) = 2i v0, P_g(w) = 2i w0  (from check_a); n! = 6
# P_{f|X|g}(v(x)w) := (1/n!) P_f(v) P_g(w) = (1/6)(2i v0)(2i w0) = -(2/3) v0 w0
# universal-property (multilinear) extension in the single slot: v0*w0 -> U00.
P_paper = sp.Rational(-2, 3) * U[0]
print("\nP_{f|X|g}(U) from Prop 10 recipe + multilinear extension:", P_paper)
print("  [P_paper]^4 - Hess  ==  0 ?  ", sp.expand(P_paper**4 - D) == 0)

# does the identity hold on the *whole* space (not just Segre)?
print("\n  Segre ideal generator (2x2 minor) det =", sp.expand(U[0]*U[3] - U[1]*U[2]))
print("  D depends only on U00 ->  identity holds GLOBALLY here, not merely on Segre.")

# explicit non-decomposable spot-checks
for pt in [(1, 0, 0, 1), (2, 1, 3, 5), (1, 7, -3, 2), (0, 1, 1, 0)]:
    sub = dict(zip(U, pt))
    minor = pt[0]*pt[3] - pt[1]*pt[2]
    print(f"  U={pt} (2x2 minor={minor}, decomposable iff 0):"
          f" Hess={D.subs(sub)}  [P]^4={sp.expand(P_paper**4).subs(sub)}")
