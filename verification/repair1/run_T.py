import sympy as sp
from nspace import report

x = sp.symbols('x0:6')

cases = []
# --- binary family W_n = x0^{n-1} x1  (d=2), all n
for n in [3,4,5,6]:
    cases.append((f'W_{n} = x0^{n-1}*x1', x[0]**(n-1)*x[1], list(x[:2]), n))
# --- d=3 family  x0^{n-2}(x0 x2 + x1^2), all n  (GO Thm 5)
for n in [3,4,5]:
    cases.append((f'F3_{n} = x0^{n-2}(x0*x2+x1^2)', x[0]**(n-2)*(x[0]*x[2]+x[1]**2), list(x[:3]), n))
# --- (d,n)=(4,3) classified cubics
cases.append(('CC = x0^2x3+x0x1x2+x1^3 (Chasles-Cayley)', x[0]**2*x[3]+x[0]*x[1]*x[2]+x[1]**3, list(x[:4]), 3))
cases.append(('QT = x0^2x3+x0x1x2 (quadric+tangent plane)', x[0]**2*x[3]+x[0]*x[1]*x[2], list(x[:4]), 3))
# --- GO Example 28: n=4, d=4
cases.append(('GOEx28 = x0x2^3+x1x2^2x3+x3^4', x[0]*x[2]**3+x[1]*x[2]**2*x[3]+x[3]**4, list(x[:4]), 4))

rows = []
for name, f, xs, n in cases:
    r = report(name, f, xs, n)
    if r: rows.append(r)

print('\n\n=========== SUMMARY ===========')
print(f'{"case":45s} {"n":>2s} {"d":>2s} {"r":>2s} {"dimN":>5s} {"Nil":>6s} {"T":>6s}')
for r in rows:
    print(f'{r["name"][:45]:45s} {r["n"]:2d} {r["d"]:2d} {r["r"]:2d} {r["dimN"]:5d} {str(r["nil"]):>6s} {str(r["T"]):>6s}')
