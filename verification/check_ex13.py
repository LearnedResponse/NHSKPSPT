"""Validation of the Kronecker-product implementation against
   Gharahi arXiv:2608.11182 Example 13."""
import sympy as sp
from klib import kron, flat_Z, hess_det

x0, x1, y0, y1 = sp.symbols('x0 x1 y0 y1')
Z = [[sp.Symbol(f'z{i}{j}') for j in range(2)] for i in range(2)]
zs = flat_Z(Z)

f = (x0 + x1)**3 * (x0 + 2*x1)
g = (y0 + 2*y1)**3 * (2*y0 + 3*y1)
h = kron(f, [x0, x1], g, [y0, y1], 4, Z)

L00 = zs[0] + 2*zs[1] + zs[2] + 2*zs[3]
L01 = 2*zs[0] + 3*zs[1] + 2*zs[2] + 3*zs[3]
L10 = zs[0] + 2*zs[1] + 2*zs[2] + 4*zs[3]
L11 = 2*zs[0] + 3*zs[1] + 4*zs[2] + 6*zs[3]
claim = sp.Rational(3, 4)*L00**2*L01*L10 + sp.Rational(1, 4)*L00**3*L11
print("f|X|g matches paper's L-form expression :", sp.expand(h - claim) == 0)

_, Dh = hess_det(h, zs)
print("Hess(f|X|g) =", sp.factor(Dh))
print("matches (3/4)^4 * L00^8                 :",
      sp.expand(Dh - sp.Rational(3, 4)**4 * L00**8) == 0)
