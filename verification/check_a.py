"""(a) W_3 = x0^2 x1 in Sym^3 C^2: persistence certificate (n-2 = 1 polarization,
    Hess of the polarized quadratic is a d=2 power of a multilinear form).
    Also (validation) reproduce Gharahi-Ottaviani Example 3 (i)-(iii)."""
import sympy as sp
from klib import polarize, hess_det

x0, x1, v0, v1, u0, u1 = sp.symbols('x0 x1 v0 v1 u0 u1')
xs = [x0, x1]

print("=== (a) W_3 = x0^2*x1 in Sym^3 C^2, d=2, n=3, r=n-2=1 ===")
f = x0**2 * x1
fv = polarize(f, xs, [[v0, v1]])
print("  f_v(x) =", sp.expand(fv))
H, D = hess_det(fv, xs)
print("  Hess matrix =", H.tolist())
print("  Hess(f_v(x)) =", sp.factor(D))
# is D a perfect square of a linear (multidegree (1)) form?
P = sp.sqrt(sp.factor(D))
print("  sqrt          =", sp.simplify(P), " -> P_f(v) = 2i*v0")
print("  check [P_f]^2 == Hess:", sp.simplify((2*sp.I*v0)**2 - D) == 0)
print("  => W_3 persistent by Thm 3 (GO Thm 2(c)). ell_f = x0, deg-1 root.")

Hf, Df = hess_det(f, xs)
print("  ordinary Hess(f) =", sp.factor(Df), "= lambda*ell^d with lambda=-4, ell=x0, d=2")

print()
print("=== validation vs Gharahi-Ottaviani arXiv:2510.07404 Example 3 ===")
for name, g in [("x0^4+x1^4 (NOT persistent)", x0**4 + x1**4),
                ("x0^2*x1^2 (NOT persistent)", x0**2 * x1**2),
                ("x0^3*x1 = W_4 (persistent)", x0**3 * x1)]:
    gp = polarize(g, xs, [[u0, u1], [v0, v1]])
    _, Dg = hess_det(gp, xs)
    print(f"  {name}: Hess_x(polarized) = {sp.factor(Dg)}")
