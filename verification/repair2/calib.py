"""PART 1 -- CALIBRATION of the JC gate against Sol's striking hint:
      "in GO-Example-28's printed matrices,  N_1 N_4 + N_4 N_1 = N_3".

We (i) reproduce the four matrices exactly as PRINTED in the repair1 chain
(bundle B6, from repair1/out_T.txt), (ii) test the Jordan identity on them
verbatim, (iii) rebuild the generators from scratch with the repair2 DIAGONAL-base
pipeline and check the two generating sets span the same space and satisfy the
same Jordan closure, and (iv) print the FULL Jordan multiplication table.
"""
import os, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gates import nspace_diag, run_gates, in_span            # noqa: E402

print('=' * 78)
print('PART 1  CALIBRATION  --  GO-Example-28,  f = x0*x2^3 + x1*x2^2*x3 + x3^4')
print('=' * 78)

# ---- (i) the four matrices exactly as printed in repair1/out_T.txt (= bundle B6
#          lines 2172-2175), in the order printed.  1-indexed labels N_1..N_4.
P = [sp.Matrix([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0]]),
     sp.Matrix([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
     sp.Matrix([[0, 0, 0, 1], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
     sp.Matrix([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])]
print('\n(i) printed generators (out_T.txt order, labelled N_1..N_4):')
for k, M in enumerate(P, 1):
    print(f'    N_{k} = {M.tolist()}')

S14 = sp.expand(P[0] * P[3] + P[3] * P[0])
print('\n(ii) SOL\'S HINT, tested verbatim on the printed matrices:')
print(f'    N_1 N_4        = {sp.expand(P[0]*P[3]).tolist()}')
print(f'    N_4 N_1        = {sp.expand(P[3]*P[0]).tolist()}')
print(f'    N_1N_4+N_4N_1  = {S14.tolist()}')
print(f'    N_3            = {P[2].tolist()}')
print(f'    ==> N_1 N_4 + N_4 N_1 == N_3 :  {sp.expand(S14 - P[2]).is_zero_matrix}')

# ---- (iii) rebuild from the form with the repair2 diagonal-base pipeline
x = sp.symbols('x0:4')
f = x[0] * x[2]**3 + x[1] * x[2]**2 * x[3] + x[3]**4
info = nspace_diag(f, list(x), 4)
G = info['basis']
print('\n(iii) rebuilt by repair2/gates.py (diagonal base b = '
      f"{[str(t) for t in info['base'][0]]}, r={info['r']}, d={info['d']}):")
for k, M in enumerate(G, 1):
    print(f'    N_{k} = {M.tolist()}')
same_list = (len(G) == len(P)) and all(sp.expand(a - b).is_zero_matrix for a, b in zip(G, P))
print(f'    identical matrices, in the same order : {same_list}')
Bp = sp.Matrix([list(m) for m in P]).T
Bg = sp.Matrix([list(m) for m in G]).T
print(f'    same SPAN (rank test): rank(P)={Bp.rank()}  rank(G)={Bg.rank()}  '
      f'rank(P|G)={Bp.row_join(Bg).rank()}')
print(f'    lambda values at the 16 basis tuples all rational: '
      f"{all(v.is_rational for v in info['lams'].values())}")

# ---- (iv) full Jordan table on the rebuilt (= printed) generators
print('\n(iv) FULL Jordan multiplication table  N_i o N_j = N_i N_j + N_j N_i :')
d = info['d']
allin = True
for i in range(len(G)):
    for j in range(i, len(G)):
        S = sp.expand(G[i] * G[j] + G[j] * G[i])
        ok, co = in_span(S, G, d)
        allin = allin and ok
        if S.is_zero_matrix:
            txt = '0'
        else:
            terms = []
            for k, c in enumerate(list(co), 1):
                if c != 0:
                    terms.append(f'{c}*N_{k}' if c != 1 else f'N_{k}')
            txt = ' + '.join(terms) if ok else f'NOT IN SPAN: {S.tolist()}'
        print(f'    N_{i+1} o N_{j+1} = {txt}')
print(f'\n    JORDAN CLOSURE on GO-Ex.28 : {allin}')

g = run_gates(info)
print(f'    gates: TR={g["TR"]}  JC={g["JC"]}  CK={g["CK"]} (common kernel dim {g["ck_dim"]})')
print(f'    traces of the four generators: {[sp.expand(M.trace()) for M in G]}')

# ---- squares, since JC with i=j is the substantive nilpotency-flavoured part
print('\n    squares: ' + ', '.join(f'N_{k+1}^2 = {sp.expand(G[k]**2).tolist()}'
                                    for k in range(len(G))))
