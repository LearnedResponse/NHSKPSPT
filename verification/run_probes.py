import sympy as sp
from probe import probe

x = sp.symbols('x0:6')
y = sp.symbols('y0:6')

W3x, W3y = x[0]**2*x[1], y[0]**2*y[1]
W4x, W4y = x[0]**3*x[1], y[0]**3*y[1]
# GO Thm 5 (d=3): persistent cubic / quartic in 3 vars
C3x = x[0]**2*x[2] + x[0]*x[1]**2                 # Sym^3 C^3  (paper's Fig.1 g)
C3y = y[0]**2*y[2] + y[0]*y[1]**2
Q3x = x[0]**3*x[2] + x[0]**2*x[1]**2              # Sym^4 C^3
# GO Thm 25: Chasles-Cayley persistent cubic in Sym^3 C^4
CCx = x[0]**2*x[3] + x[0]*x[1]*x[2] + x[1]**3
# GO Example 28: persistent quartic in Sym^4 C^4, no linear factor, Hess = 9 x2^8
E28x = x[0]*x[2]**3 + x[1]*x[2]**2*x[3] + x[3]**4

res = {}
res['(b) W3 |X| W3  (Sym^3 C^4)'] = probe('W3 |X| W3', W3x, x[:2], W3y, y[:2], 3,
                                          seed=1, symbolic=True)
res['W4 |X| W4  (Sym^4 C^4, r=2, beyond appendix)'] = probe(
    'W4 |X| W4', W4x, x[:2], W4y, y[:2], 4, seed=2, symbolic=True)
res['W3 |X| C3  (Sym^3 C^6)'] = probe('W3 |X| C3', W3x, x[:2], C3y, y[:3], 3,
                                      seed=3, symbolic=True)
res['C3 |X| C3  (Sym^3 C^9)'] = probe('C3 |X| C3', C3x, x[:3], C3y, y[:3], 3,
                                      seed=4, trials=4)
res['CC |X| W3  (Sym^3 C^8, Chasles-Cayley)'] = probe(
    'CC |X| W3', CCx, x[:4], W3y, y[:2], 3, seed=5, trials=4)
res['Q3 |X| W4  (Sym^4 C^6, r=2)'] = probe('Q3 |X| W4', Q3x, x[:3], W4y, y[:2], 4,
                                           seed=6, trials=4)
res['E28 |X| W4 (Sym^4 C^8, r=2)'] = probe('E28 |X| W4', E28x, x[:4], W4y, y[:2], 4,
                                           seed=7, trials=3)
# generic GL-twist so that ell_f, ell_g are GENERIC linear forms: this actually
# exercises the multilinear (universal-property) extension off the coordinate axes.
Tw = C3y.subs({y[0]: y[0] + 2*y[1] - y[2], y[1]: y[1] + 3*y[2], y[2]: y[0] - y[2]},
              simultaneous=True)
res['W3 |X| GL-twisted C3 (Sym^3 C^6)'] = probe('W3 |X| twist(C3)', W3x, x[:2],
                                                sp.expand(Tw), y[:3], 3, seed=8,
                                                trials=4, symbolic=True)
Tw4 = E28x.subs({x[0]: x[0] + x[1], x[1]: x[1] - x[3], x[2]: x[2] + 2*x[0],
                 x[3]: x[3] + x[2]}, simultaneous=True)
res['GL-twisted E28 |X| W4 (Sym^4 C^8)'] = probe('twist(E28) |X| W4', sp.expand(Tw4),
                                                 x[:4], W4y, y[:2], 4, seed=9, trials=3)

print("\n" + "=" * 70)
for k, v in res.items():
    print(f"  {k:48s} : {'PASS' if v else 'FAIL'}")
