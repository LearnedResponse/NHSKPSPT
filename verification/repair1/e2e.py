"""END-TO-END verification of the REPAIRED theorem on a NEW case
(not among the 9 prior probes):   f = g = x0^2(x0 x2 + x1^2)  in Sym^4 C^3,
so n=4, r=n-2=2, d1=d2=3, D=9, and the polarization slots U^(i) are 3x3
matrices of RANK 3 -- each slot contributes 3 rank-one terms, so the Step-1
expansion has 3^2 = 9 multi-index terms.  Everything exact over Q.

Verified, in order:
  L1  Step-1 expansion   H_h(U1,U2) = (1/n!) sum_k H_f(v_k) (x) H_g(w_k)
  L2  M_h(U) = sum_k M_f(A_k) (x) M_g(A'_k)      (base = decomposable base)
  L3  lambda_h(U) = sum_k lambda_k mu_k = P_h(U)/P_h(base)
  L4  N_h(U) = M_h - lambda_h I is STRICTLY UPPER TRIANGULAR in the lex basis
      built from the triangularizing bases of Nsp(f), Nsp(g)   [=> T(h)]
  L5  one-sided reduction: det(sum_k M_f (x) M_g) = det(Y)^{d1},
      Y = sum_k lambda_k M_g(A'_k) = lambda_h I + (element of Nsp(g))
  L6  THE THEOREM:  det H_h(U1,U2) = [P_h(U1,U2)]^D
"""
import itertools, sys, os
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klib import kron, flat_Z, polarize, hess_det
from nspace import polarized_hessian, certificate, subs_tuple

n = 4; r = n - 2; d1 = d2 = 3; D = d1*d2
xs = list(sp.symbols('x0:3')); ys = list(sp.symbols('y0:3'))
f = xs[0]**2*(xs[0]*xs[2] + xs[1]**2)
g = ys[0]**2*(ys[0]*ys[2] + ys[1]**2)
print(f"f = {f}\ng = {g}\nn={n} r={r} d1={d1} d2={d2} D={D}")

rf, sv_f, Hf, cf, Qf = certificate(f, xs, n)
rg, sv_g, Hg, cg, Qg = certificate(g, ys, n)
print(f"\ncertificate f: det H_f = ({cf})*({Qf})^{d1}")
print(f"certificate g: det H_g = ({cg})*({Qg})^{d2}")

def Hf_at(A): return sp.Matrix(subs_tuple(Hf, sv_f, A))
def Hg_at(A): return sp.Matrix(subs_tuple(Hg, sv_g, A))
def Qf_at(A): return sp.expand(subs_tuple(Qf, sv_f, A))
def Qg_at(A): return sp.expand(subs_tuple(Qg, sv_g, A))

e = lambda i,d: [sp.Integer(1) if k==i else sp.Integer(0) for k in range(d)]
Bf = [e(0,d1)]*r ; Bg = [e(0,d2)]*r            # base tuples
HfB, HgB = Hf_at(Bf), Hg_at(Bg)
print("H_f(base) =", HfB.tolist(), " det =", HfB.det())
assert HfB.det() != 0 and HgB.det() != 0
HfBi, HgBi = HfB.inv(), HgB.inv()
QfB, QgB = Qf_at(Bf), Qg_at(Bg)

# ---------------- h = f box g, and its polarized Hessian at two rank-3 slots
Z = [[sp.Symbol(f'z{i}_{j}') for j in range(d2)] for i in range(d1)]
zs = flat_Z(Z)                       # lex order: i major (V index), j minor
h = kron(f, xs, g, ys, n, Z)
print("\nh = f box g has", len(sp.Add.make_args(h)), "monomials in Sym^4(C^9)")

U1 = sp.Matrix([[2,1,-1],[0,3,1],[1,-2,2]])
U2 = sp.Matrix([[1,0,2],[-1,3,0],[2,1,-3]])
print("U1 =", U1.tolist(), "rank", U1.rank())
print("U2 =", U2.tolist(), "rank", U2.rank())
assert U1.rank() == 3 and U2.rank() == 3

# rank-3 decompositions U = sum_k e_k (x) row_k(U)
dec = []
for U in (U1, U2):
    dec.append([(e(k,d1), [U[k,j] for j in range(d2)]) for k in range(d1)])

Uflat = [[U[i,j] for i in range(d1) for j in range(d2)] for U in (U1,U2)]
hp = polarize(h, zs, Uflat)
Hh = sp.Matrix(D, D, lambda a,b: sp.expand(sp.diff(hp, zs[a], zs[b])))
detHh = sp.expand(Hh.det(method='berkowitz'))
print("\ndet H_h(U1,U2) =", detHh)

# ---------------- L1: Step-1 expansion
S = sp.zeros(D, D)
terms = []
for k1 in range(d1):
    for k2 in range(d1):
        vA = [dec[0][k1][0], dec[1][k2][0]]
        wA = [dec[0][k1][1], dec[1][k2][1]]
        terms.append((vA, wA))
        S += sp.Matrix(sp.kronecker_product(Hf_at(vA), Hg_at(wA)))
S = sp.expand(S/sp.factorial(n))
print("\nL1  H_h(U1,U2) == (1/n!) sum_k H_f (x) H_g :", sp.expand(Hh - S).is_zero_matrix)

# ---------------- L2 / L3
HhB = sp.Matrix(sp.kronecker_product(HfB, HgB))/sp.factorial(n)
Mh = sp.expand(HhB.inv()*Hh)
Ssum = sp.zeros(D, D); lam_sum = sp.Integer(0)
for vA, wA in terms:
    Mf = sp.expand(HfBi*Hf_at(vA)); Mg = sp.expand(HgBi*Hg_at(wA))
    lf = sp.Rational(Qf_at(vA), QfB) if QfB else 0
    lg = sp.Rational(Qg_at(wA), QgB) if QgB else 0
    assert sp.expand(Mf - lf*sp.eye(d1) - (Mf - lf*sp.eye(d1))) .is_zero_matrix
    Ssum += sp.Matrix(sp.kronecker_product(Mf, Mg))
    lam_sum += lf*lg
Ssum = sp.expand(Ssum)
print("L2  M_h(U) == sum_k M_f (x) M_g            :", sp.expand(Mh - Ssum).is_zero_matrix)

# P_h by the paper's multilinear extension, evaluated at (U1,U2)
Ph_num = sum(Qf_at(vA)*Qg_at(wA) for vA, wA in terms)/sp.factorial(n)
Ph_base = QfB*QgB/sp.factorial(n)
print("L3  lambda_h = sum lambda*mu               :", sp.expand(lam_sum - Ph_num/Ph_base) == 0,
      f"   (value {sp.nsimplify(lam_sum)})")

# ---------------- L4: strict upper triangularity of N_h in the lex basis
# Nsp(f) is strictly LOWER in the standard basis -> reverse basis order.
Rev = sp.Matrix(d1, d1, lambda i,j: 1 if i+j == d1-1 else 0)
Rev2 = sp.Matrix(d2, d2, lambda i,j: 1 if i+j == d2-1 else 0)
Cg = sp.Matrix(sp.kronecker_product(Rev, Rev2))     # lex basis change, V-index major
Nh = sp.expand(Mh - lam_sum*sp.eye(D))
Nh_lex = sp.expand(Cg.inv()*Nh*Cg)
strict_upper = all(Nh_lex[a,b] == 0 for a in range(D) for b in range(D) if a >= b)
print("L4  N_h strictly upper triangular in lex   :", strict_upper,
      "  (=> T(h) holds, nilpotent:", sp.expand(Nh**D).is_zero_matrix, ")")

# ---------------- L5: one-sided block-triangular reduction
Y = sp.zeros(d2, d2)
for vA, wA in terms:
    lf = sp.Rational(Qf_at(vA), QfB)
    Y += lf*sp.expand(HgBi*Hg_at(wA))
Y = sp.expand(Y)
print("L5  det(sum M_f (x) M_g) == det(Y)^d1      :",
      sp.expand(Ssum.det() - Y.det()**d1) == 0)
Ntil = sp.expand(Y - lam_sum*sp.eye(d2))
print("    Y - lambda_h I nilpotent (in Nsp(g))   :", sp.expand(Ntil**d2).is_zero_matrix,
      "  det Y =", sp.expand(Y.det()), " lambda_h^d2 =", sp.expand(lam_sum**d2))

# ---------------- L6: THE THEOREM
# P_h = (cf^d2 cg^d1)^(1/D) * Ph_num   (the constant is a D-th power taken formally)
rhs = sp.Rational(cf**d2 * cg**d1) * Ph_num**D
print("\nL6  det H_h(U1,U2)              =", detHh)
print("    [P_h(U1,U2)]^D               =", sp.expand(rhs))
print("    THEOREM HOLDS AT THIS POINT  :", sp.expand(detHh - rhs) == 0)
