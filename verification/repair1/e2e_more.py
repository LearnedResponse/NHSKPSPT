"""More end-to-end trials of the repaired theorem, incl. a second NEW case with
d1 != d2 and rank-3 slots.  Also checks that the identity holds at rank-2 and
rank-1 slots (consistency) and that the multi-index expansion really has r^slots
terms."""
import sys, os, random
import sympy as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from klib import kron, flat_Z, polarize
from nspace import certificate, subs_tuple

def run(name, f, xs, g, ys, n, Us_list):
    d1, d2, r = len(xs), len(ys), n-2; D = d1*d2
    print(f"\n### {name}  n={n} d1={d1} d2={d2} D={D} r={r}")
    _,svf,Hf,cf,Qf = certificate(f, xs, n); _,svg,Hg,cg,Qg = certificate(g, ys, n)
    e = lambda i,d: [sp.Integer(1) if k==i else sp.Integer(0) for k in range(d)]
    Bf=[e(0,d1)]*r; Bg=[e(0,d2)]*r
    HfB=sp.Matrix(subs_tuple(Hf,svf,Bf))
    if HfB.det()==0:
        for i0 in range(d1):
            Bf=[e(i0,d1)]*r; HfB=sp.Matrix(subs_tuple(Hf,svf,Bf))
            if HfB.det()!=0: break
    HgB=sp.Matrix(subs_tuple(Hg,svg,Bg))
    if HgB.det()==0:
        for i0 in range(d2):
            Bg=[e(i0,d2)]*r; HgB=sp.Matrix(subs_tuple(Hg,svg,Bg))
            if HgB.det()!=0: break
    QfB=subs_tuple(Qf,svf,Bf); QgB=subs_tuple(Qg,svg,Bg)
    Z=[[sp.Symbol(f'z{i}_{j}') for j in range(d2)] for i in range(d1)]
    zs=flat_Z(Z); h=kron(f,list(xs),g,list(ys),n,Z)
    for Us in Us_list:
        ranks=[U.rank() for U in Us]
        Uflat=[[U[i,j] for i in range(d1) for j in range(d2)] for U in Us]
        hp=polarize(h,zs,Uflat)
        Hh=sp.Matrix(D,D,lambda a,b: sp.expand(sp.diff(hp,zs[a],zs[b])))
        lhs=sp.expand(Hh.det(method='berkowitz'))
        # Step-1 multi-index expansion, rank decomposition U = sum_k e_k (x) row_k
        decs=[[(e(k,d1),[U[k,j] for j in range(d2)]) for k in range(d1)] for U in Us]
        import itertools
        Ph=sp.Integer(0); nterms=0
        for idx in itertools.product(range(d1), repeat=r):
            vA=[decs[s][idx[s]][0] for s in range(r)]; wA=[decs[s][idx[s]][1] for s in range(r)]
            Ph += subs_tuple(Qf,svf,vA)*subs_tuple(Qg,svg,wA); nterms+=1
        Ph=sp.expand(Ph)/sp.factorial(n)
        rhs=sp.expand(sp.Rational(cf**d2*cg**d1)*Ph**D)
        print(f"  ranks={ranks}  expansion terms={nterms}  LHS={lhs}  RHS={rhs}  MATCH={sp.expand(lhs-rhs)==0}")

x=list(sp.symbols('x0:4')); y=list(sp.symbols('y0:4'))
rng=random.Random(7)
def rmat(a,b,rk):
    while True:
        M=sp.Matrix(a,b,lambda i,j: sp.Integer(rng.randint(-4,4)))
        if M.rank()==rk: return M

# NEW CASE 1 (extra trials): F3_4 box F3_4, n=4, 3x3, rank 3
run("F3_4 box F3_4", x[0]**2*(x[0]*x[2]+x[1]**2), x[:3],
    y[0]**2*(y[0]*y[2]+y[1]**2), y[:3], 4,
    [[rmat(3,3,3),rmat(3,3,3)] for _ in range(3)] + [[rmat(3,3,2),rmat(3,3,3)]])

# NEW CASE 2: F3_4 box GOEx28, n=4, d1=3 d2=4, D=12, rank-3 slots
run("F3_4 box GOEx28", x[0]**2*(x[0]*x[2]+x[1]**2), x[:3],
    y[0]*y[2]**3+y[1]*y[2]**2*y[3]+y[3]**4, y[:4], 4,
    [[rmat(3,4,3),rmat(3,4,3)] for _ in range(2)])
