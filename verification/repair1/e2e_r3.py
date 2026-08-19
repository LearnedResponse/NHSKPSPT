"""Theorem 5/7 end-to-end at r = n-2 = 3 (THREE polarization slots) -- a regime
no prior probe reached (prior max was r=2).  W_5 (x) F3_5 in Sym^5(C^2 (x) C^3),
n=5, d1=2, d2=3, D=6, slots U^(i) are 2x3 matrices of rank 2 => 2^3 = 8
multi-index terms in the Lemma-3 expansion."""
import sys, os, itertools, random
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klib import kron, flat_Z, polarize
from nspace import certificate, subs_tuple

n=5; r=3; d1,d2=2,3; D=d1*d2
xs=list(sp.symbols('x0:2')); ys=list(sp.symbols('y0:3'))
f=xs[0]**4*xs[1]                       # W_5
g=ys[0]**3*(ys[0]*ys[2]+ys[1]**2)      # F3_5
print(f"f={f}  g={g}  n={n} r={r} d1={d1} d2={d2} D={D}")
_,svf,Hf,cf,Qf=certificate(f,xs,n); _,svg,Hg,cg,Qg=certificate(g,ys,n)
print("P_f ~",Qf," c_f =",cf); print("P_g ~",Qg," c_g =",cg)
e=lambda i,d:[sp.Integer(1) if k==i else sp.Integer(0) for k in range(d)]
Bf=[e(0,d1)]*r; Bg=[e(0,d2)]*r
HfB=sp.Matrix(subs_tuple(Hf,svf,Bf)); HgB=sp.Matrix(subs_tuple(Hg,svg,Bg))
QfB=subs_tuple(Qf,svf,Bf); QgB=subs_tuple(Qg,svg,Bg)
print("det H_f(B)=",HfB.det()," det H_g(B')=",HgB.det())
Z=[[sp.Symbol(f'z{i}_{j}') for j in range(d2)] for i in range(d1)]
zs=flat_Z(Z); h=kron(f,xs,g,ys,n,Z)
print("h has",len(sp.Add.make_args(h)),"monomials in Sym^5(C^6)")
rng=random.Random(23)
for trial in range(4):
    while True:
        Us=[sp.Matrix(d1,d2,lambda i,j: sp.Integer(rng.randint(-3,3))) for _ in range(r)]
        if all(U.rank()==2 and U[0,0]!=0 for U in Us): break  # avoid the trivial P_h=0 locus
    Uflat=[[U[i,j] for i in range(d1) for j in range(d2)] for U in Us]
    hp=polarize(h,zs,Uflat)
    Hh=sp.Matrix(D,D,lambda a,b: sp.expand(sp.diff(hp,zs[a],zs[b])))
    lhs=sp.expand(Hh.det(method='berkowitz'))
    # rank decomposition U = P (P^-1 U)
    while True:
        P=sp.Matrix(d1,d1,lambda i,j: sp.Integer(rng.randint(-3,3)))
        if P.det()!=0: break
    decs=[]
    for U in Us:
        Qm=P.inv()*U
        decs.append([([P[i,k] for i in range(d1)],[Qm[k,j] for j in range(d2)])
                     for k in range(d1)])
    S=sp.zeros(D,D); Ph=sp.Integer(0); nt=0
    for idx in itertools.product(range(d1),repeat=r):
        vA=[decs[s][idx[s]][0] for s in range(r)]; wA=[decs[s][idx[s]][1] for s in range(r)]
        S += sp.Matrix(sp.kronecker_product(sp.Matrix(subs_tuple(Hf,svf,vA)),
                                            sp.Matrix(subs_tuple(Hg,svg,wA))))
        Ph += subs_tuple(Qf,svf,vA)*subs_tuple(Qg,svg,wA); nt+=1
    S=sp.expand(S/sp.factorial(n)); Ph=sp.expand(Ph)/sp.factorial(n)
    rhs=sp.expand(sp.Rational(cf**d2*cg**d1)*Ph**D)
    print(f"  trial {trial}: ranks={[U.rank() for U in Us]} terms={nt}  "
          f"L1(expansion)={sp.expand(Hh-S).is_zero_matrix}  "
          f"LHS={lhs} RHS={rhs} THEOREM={sp.expand(lhs-rhs)==0}")
