"""CONTRACTION LEMMA (the step that makes the theorem FULLY one-sided).

If P_f(v^(1),..,v^(r)) = lambda_f * ell(v^(1))...ell(v^(r))   [GO condition (a)],
then with  lhat(v) := ell(v)/ell(b),  the block-diagonal part
        Y = sum_k lambda_k M_g(A'_k)
of the T(f)-triangularized product is itself the value of M_g at ONE tuple:
        Y = M_g(W^(1),...,W^(r)),   W^(i) := sum_k lhat(v^{(i,k)}) w^{(i,k)}.
Hence det Y = lambda_g(W)^{d2} by the CERTIFICATE ALONE -- no hypothesis on g.
Checked exactly on F3_4 (x) GOEx28, n=4, r=2, rank-3 slots."""
import sys, os, itertools, random
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nspace import certificate, subs_tuple

n=4; r=2
xs=list(sp.symbols('x0:3')); ys=list(sp.symbols('y0:4'))
f=xs[0]**2*(xs[0]*xs[2]+xs[1]**2)
g=ys[0]*ys[2]**3+ys[1]*ys[2]**2*ys[3]+ys[3]**4
d1,d2=3,4
_,svf,Hf,cf,Qf=certificate(f,xs,n); _,svg,Hg,cg,Qg=certificate(g,ys,n)
print("P_f ~", Qf, "   (product of linear forms in each slot => GO condition (a))")
print("P_g ~", Qg)
e=lambda i,d:[sp.Integer(1) if k==i else sp.Integer(0) for k in range(d)]
Bf=[e(0,d1)]*r; Bg=[e(2,d2)]*r
HfB=sp.Matrix(subs_tuple(Hf,svf,Bf)); HgB=sp.Matrix(subs_tuple(Hg,svg,Bg))
print("det H_f(B)=",HfB.det()," det H_g(B')=",HgB.det())
QfB=subs_tuple(Qf,svf,Bf); QgB=subs_tuple(Qg,svg,Bg); HgBi=HgB.inv(); HfBi=HfB.inv()
# lhat: P_f = c * (v0_0 * v1_0) so ell(v)=v_0, ell(b)=1
lhat=lambda v: v[0]
rng=random.Random(11); ok=True
for trial in range(4):
    Us=[sp.Matrix(d1,d2,lambda i,j: sp.Integer(rng.randint(-4,4))) for _ in range(r)]
    # NON-trivial rank decomposition: U = P * (P^-1 U) = sum_k col_k(P) (x) row_k(P^-1 U)
    while True:
        P=sp.Matrix(d1,d1,lambda i,j: sp.Integer(rng.randint(-3,3)))
        if P.det()!=0: break
    R=P.inv()*Us[0]*0  # placeholder
    decs=[]
    for U in Us:
        Q=P.inv()*U
        decs.append([([P[i,k] for i in range(d1)], [Q[k,j] for j in range(d2)])
                     for k in range(d1)])
    Y=sp.zeros(d2,d2); lam_mu=sp.Integer(0)
    for idx in itertools.product(range(d1),repeat=r):
        vA=[decs[s][idx[s]][0] for s in range(r)]; wA=[decs[s][idx[s]][1] for s in range(r)]
        lk=sp.Rational(subs_tuple(Qf,svf,vA),QfB)
        Mg=sp.expand(HgBi*sp.Matrix(subs_tuple(Hg,svg,wA)))
        Y+=lk*Mg
        lam_mu+=lk*sp.Rational(subs_tuple(Qg,svg,wA),QgB)
    Y=sp.expand(Y)
    W=[[sum(lhat(decs[s][k][0])*decs[s][k][1][j] for k in range(d1)) for j in range(d2)]
       for s in range(r)]
    MgW=sp.expand(HgBi*sp.Matrix(subs_tuple(Hg,svg,W)))
    c1=sp.expand(Y-MgW).is_zero_matrix
    lamW=sp.Rational(subs_tuple(Qg,svg,W),QgB)
    c2=sp.expand(lamW-lam_mu)==0
    c3=sp.expand(Y.det()-lam_mu**d2)==0
    ok&=(c1 and c2 and c3)
    print(f"  trial {trial}: ranks={[U.rank() for U in Us]}  Y==M_g(W): {c1}  "
          f"lambda_g(W)==sum(lambda*mu): {c2}  det Y == (sum lambda*mu)^d2: {c3}")
print("CONTRACTION LEMMA holds on all trials:", ok)
