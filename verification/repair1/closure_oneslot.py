"""(A) STEP-4 closure, verified computationally: T(f),T(g) => T(f box g).
   (B) STEP-2 one-slot char-poly lemma, verified SYMBOLICALLY (all constants)."""
import sys, os, itertools
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klib import kron, flat_Z
from nspace import report, certificate, subs_tuple, polarized_hessian

print("="*70); print("(A) CLOSURE: compute Nsp / Nil / T for actual Kronecker products")
print("="*70)
x = list(sp.symbols('x0:4')); y = list(sp.symbols('y0:4'))
prods = [
 ("W_4 box W_4 in Sym^4 C^4", x[0]**3*x[1], x[:2], y[0]**3*y[1], y[:2], 4),
 ("F3_4 box W_4 in Sym^4 C^6", x[0]**2*(x[0]*x[2]+x[1]**2), x[:3], y[0]**3*y[1], y[:2], 4),
 ("W_3 box CC  in Sym^3 C^8", x[0]**2*x[1], x[:2],
  y[0]**2*y[3]+y[0]*y[1]*y[2]+y[1]**3, y[:4], 3),
]
for name, f, xs, g, ys, n in prods:
    d1, d2 = len(xs), len(ys)
    Z = [[sp.Symbol(f'z{i}_{j}') for j in range(d2)] for i in range(d1)]
    h = kron(f, list(xs), g, list(ys), n, Z)
    report(name, h, flat_Z(Z), n)

print(); print("="*70)
print("(B) STEP-2 LEMMA (one-slot lines), symbolic, constants included:")
print("    det( t*H_f(B) - H_f(B with slot j replaced by a) )")
print("      = [ t*P_f(B) - P_f(B with slot j -> a) ]^d   and   det H_f(B)=P_f(B)^d")
print("    => charpoly( H_f(B)^-1 H_f(A) ) = (t - P_f(A)/P_f(B))^d  on one-slot lines")
print("="*70)
t = sp.Symbol('t')
cases = [("W_5", x[0]**4*x[1], x[:2], 5), ("F3_5", x[0]**3*(x[0]*x[2]+x[1]**2), x[:3], 5),
         ("GOEx28", x[0]*x[2]**3+x[1]*x[2]**2*x[3]+x[3]**4, x[:4], 4)]
for name, f, xs, n in cases:
    d, r = len(xs), n-2
    rr, sv, H, c, Q = certificate(f, xs, n)
    e = lambda i: [sp.Integer(1) if k==i else sp.Integer(0) for k in range(d)]
    # base: all slots = e_i0 with det != 0
    B = None
    for i0 in range(d):
        Bt = [e(i0)]*r
        if sp.Matrix(subs_tuple(H, sv, Bt)).det() != 0: B = Bt; break
    HB = sp.Matrix(subs_tuple(H, sv, B)); QB = subs_tuple(Q, sv, B)
    print(f"\n  {name}: d={d}, r={r}, det H_f(B)={HB.det()}, c*Q(B)^d={c*QB**d} -> "
          f"equal: {sp.expand(HB.det()-c*QB**d)==0}")
    a = [sp.Symbol(f'a_{i}') for i in range(d)]
    ok_all = True
    for j in range(r):
        A = [list(B[k]) for k in range(r)]; A[j] = a
        HA = sp.Matrix(subs_tuple(H, sv, A)); QA = sp.expand(subs_tuple(Q, sv, A))
        lhs = sp.expand((t*HB - HA).det(method='berkowitz'))
        rhs = sp.expand(c*(t*QB - QA)**d)
        M = sp.expand(HB.inv()*HA)
        cp = sp.expand(M.charpoly(t).as_expr())
        tgt = sp.expand((t - QA/QB)**d)
        ok = (sp.expand(lhs-rhs)==0) and (sp.expand(cp-tgt)==0)
        ok_all &= ok
        print(f"    slot {j}: det(tH_B - H_A) == c*(tQ_B-Q_A)^d : {sp.expand(lhs-rhs)==0}"
              f" ; charpoly(M) == (t-lambda)^d : {sp.expand(cp-tgt)==0}")
    print(f"    ==> STEP 2 (one-slot) VERIFIED for {name}: {ok_all}")
