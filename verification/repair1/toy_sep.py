"""HOSTILE PROBE against STEP 2 -> STEP 3.

Step 2 of the sketch proves: for tuples differing from the base in ONE slot,
charpoly(M_f) = (t-lambda)^d.  Step 3 CONSUMES: charpoly(M_f(A)) = (t-lambda(A))^d
for ARBITRARY tuples A (the multi-index expansion of Step 1 produces tuples whose
slots come from different rank-one components, i.e. all slots differ from base).

QUESTION: is "arbitrary tuple" a FORMAL consequence of
   (i)  H symmetric multilinear in the r slots, and
   (ii) det H(A) = P(A)^d identically, P multilinear?
ANSWER (this script): NO.  Explicit symmetric bilinear family (r=2, d=2) with
det H(u,v) = P(u,v)^2 identically, yet M(A) has TWO DISTINCT eigenvalues for
some A, and span{N(A)} contains non-nilpotent elements.
"""
import sympy as sp

u0,u1,v0,v1,t = sp.symbols('u0 u1 v0 v1 t')

p = u0*v0
q = (u0*v1 + u1*v0)/2
s = u1*v1
H = sp.Matrix([[p, q],[q, s]])

print("H(u,v) =", H.tolist(), "   (symmetric in the two slots u<->v: ",
      sp.expand(H.subs({u0:v0,u1:v1,v0:u0,v1:u1},simultaneous=True) - H).is_zero_matrix, ")")
det = sp.expand(H.det())
print("det H(u,v) =", sp.factor(det))
P = sp.I*(u0*v1 - u1*v0)/2
print("P(u,v) =", P, "  multilinear, and det H - P^2 =", sp.expand(det - P**2))
assert sp.expand(det - P**2) == 0

# base tuple B = (e0, e1)
sub_B = {u0:1,u1:0,v0:0,v1:1}
HB = H.subs(sub_B); PB = P.subs(sub_B)
print("\nbase B=(e0,e1):  H(B) =", HB.tolist(), " det =", HB.det(), " P(B) =", PB)
HBi = HB.inv()

M = sp.expand(HBi*H)
lam = sp.simplify(P/PB)
cp = sp.factor(sp.expand(M.charpoly(t).as_expr()))
print("\nM(A) = H(B)^-1 H(A) =", M.tolist())
print("lambda(A) = P(A)/P(B) =", sp.expand(lam))
print("charpoly M(A) =", sp.expand(cp))
print("(t - lambda)^2  =", sp.expand((t-lam)**2))
print("DIFFERENCE     =", sp.expand(cp - (t-lam)**2))
print("=> single-eigenvalue for ARBITRARY tuples:",
      sp.expand(cp - (t-lam)**2) == 0)
print("det M(A) - lambda^2 =", sp.expand(sp.expand(M.det()) - lam**2),
      "  (so the DETERMINANT half of the certificate does hold)")

print("\n-- explicit witness --")
for name, sb in [("A=(e0,e0)", {u0:1,u1:0,v0:1,v1:0}),
                 ("A=(e1,e1)", {u0:0,u1:1,v0:0,v1:1}),
                 ("A=((1,1),(1,1))", {u0:1,u1:1,v0:1,v1:1})]:
    Ma = M.subs(sb); la = sp.simplify(lam.subs(sb))
    N = sp.expand(Ma - la*sp.eye(2))
    print(f"  {name}: M={Ma.tolist()} lambda={la} N={N.tolist()} "
          f"eigs(M)={sorted(Ma.eigenvals().keys(), key=str)} N nilpotent={(N**2).is_zero_matrix}")

# span of the N's
N1 = sp.Matrix([[0,0],[2,0]]); N2 = sp.Matrix([[0,2],[0,0]])
print("\nN(e0,e0) =", N1.tolist(), " N(e1,e1) =", N2.tolist(),
      " both nilpotent, but N1+N2 =", (N1+N2).tolist(),
      " eigenvalues", sorted((N1+N2).eigenvals().keys(), key=str))
print("=> Nil (span nilpotent) FAILS for this formal family.")
print("\nCONCLUSION: 'certificate + multilinearity' does NOT formally imply either")
print("  (single eigenvalue at arbitrary tuples) or (Nil).  Step 2 is strictly")
print("  weaker than what Step 3 consumes whenever r = n-2 >= 2.")
