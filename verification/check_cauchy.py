"""Quantifies the restriction kernel the critique names.
Cauchy:  Sym^D(V (x) W) = (+)_{lambda |- D} S_lambda(V) (x) S_lambda(W),
terms with l(lambda) > min(d1,d2) vanishing.  Restricting a degree-D form on
V(x)W to the Segre cone {v(x)w} lands in Sym^D V* (x) Sym^D W* = the lambda=(D)
"Cartan" summand; the kernel is exactly (+)_{lambda != (D)} = I(Seg)_D.
Per polarization slot of Prop 10, D = d1*d2."""
import sympy as sp
from sympy.utilities.iterables import partitions


def s_dim(lam, d):                      # Weyl dim formula for S_lambda(C^d)
    lam = list(lam) + [0]*(d - len(lam))
    if len(lam) > d:
        return 0
    num = den = 1
    for i in range(d):
        for j in range(i+1, d):
            num *= (lam[i] - lam[j] + j - i)
            den *= (j - i)
    return num // den


print(f"{'d1':>3} {'d2':>3} {'D=d1d2':>7} {'dim Sym^D(VxW)*':>17} "
      f"{'Cartan (restriction img)':>25} {'ker = I(Seg)_D':>16}")
for d1, d2 in ((2,2),(2,3),(3,3),(2,4),(3,4)):
    if True:
        D = d1*d2
        tot = int(sp.binomial(D + d1*d2 - 1, d1*d2 - 1))
        cart = int(sp.binomial(D + d1 - 1, d1 - 1) * sp.binomial(D + d2 - 1, d2 - 1))
        chk = 0
        for p in partitions(D):
            lam = sorted([k for k, v in p.items() for _ in range(v)], reverse=True)
            chk += s_dim(lam, d1) * s_dim(lam, d2)
        assert chk == tot, (chk, tot)      # Cauchy identity check
        print(f"{d1:>3} {d2:>3} {D:>7} {tot:>17} {cart:>25} {tot-cart:>16}")
print("\nkernel is nonzero in every row: agreement on the Segre pins down only the")
print("Cartan component; the rest of Sym^D(VxW)* is invisible to decomposable tests.")
