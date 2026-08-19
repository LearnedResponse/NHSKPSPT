"""PART 3a support -- the three load-bearing steps of the Veronese contraction
lemma, checked exactly on the forms we have.

 (V1) SYMMETRY of P_f in its r slots (memo attempt-1 limit 8.7 leaves symmetric
      vs ALTERNATING open in general; the lemma needs symmetric).
 (V2) The CONTRACTION IDENTITY  P_{d_u f}(v^2..v^r) = zeta * P_f(u,v^2..v^r),
      zeta^d = 1, checked by computing the certificate of d_u f from scratch for
      random u and comparing with the first-slot contraction of P_f.
 (V3) The DENSITY step: {u : P_f(u,-) == 0} is the kernel of the linear map
      Phi: u -> iota_u P_f, hence a proper linear subspace, hence its complement
      is Zariski-dense -- computed explicitly per form.
 (V4) condition (a-multi): P_f = c * prod_i ell(v^(i)) with ONE linear form ell.
"""
import itertools, os, random, re, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gates import named_cases                                  # noqa: E402
from nspace import certificate, subs_tuple                     # noqa: E402

HUNT = os.path.join(os.path.dirname(HERE), 'repair1', 'out_hunt.txt')
random.seed(20260817)

print('=' * 78)
print('PART 3a SUPPORT -- symmetry of P_f, the contraction identity, density,')
print('                   and condition (a-multi)')
print('=' * 78)


def analyse(name, f, xs, n, verbose=True):
    d, r = len(xs), n - 2
    r_, sv, H, c, Q = certificate(f, xs, n)
    # (V1) symmetry of Q under slot permutations
    sym = True
    alt = (r >= 2)
    for perm in itertools.permutations(range(r)):
        sub = {}
        for k in range(r):
            for i in range(d):
                sub[sv[k][i]] = sp.Symbol(f's{perm[k]}_{i}')
        Qp = Q.subs(sub, simultaneous=True)
        Q0 = Q.subs({sv[k][i]: sp.Symbol(f's{k}_{i}')
                     for k in range(r) for i in range(d)}, simultaneous=True)
        sgn = sp.Integer(1)
        # parity of perm
        p = list(perm)
        inv = sum(1 for a in range(r) for b in range(a + 1, r) if p[a] > p[b])
        sgn = (-1)**inv
        if sp.expand(Qp - Q0) != 0:
            sym = False
        if sp.expand(Qp - sgn * Q0) != 0:
            alt = False
    # (V4) condition (a-multi): Q = c * prod_i ell(v^(i)) with the same ell
    cc, facs = sp.factor_list(sp.expand(Q))
    ells, cond_a = [], True
    for b, e in facs:
        slots = [k for k in range(r) if any(b.has(v) for v in sv[k])]
        if e != 1 or len(slots) != 1 or sp.Poly(b, *sv[slots[0]]).total_degree() != 1:
            cond_a = False
            break
        k = slots[0]
        ells.append([sp.Poly(b, *sv[k]).coeff_monomial(v) for v in sv[k]])
    if cond_a and len(ells) == r:
        L0 = sp.Matrix(ells[0])
        for L in ells[1:]:
            if sp.Matrix([list(L0), list(L)]).rank() != 1:
                cond_a = False
    elif len(ells) != r:
        cond_a = False
    # (V3) density: kernel of u -> iota_u P_f
    us = [sp.Symbol(f'u{i}') for i in range(d)]
    Q1 = Q.subs({sv[0][i]: us[i] for i in range(d)}, simultaneous=True)
    if r == 1:
        coeffs = [sp.expand(Q1)]          # Phi(u) = P_f(u) itself, a scalar
    else:
        rest = [v for k in range(1, r) for v in sv[k]]
        P = sp.Poly(sp.expand(Q1), *rest)
        coeffs = [sp.expand(co) for co in P.coeffs()]
    Mker = sp.Matrix([[sp.expand(sp.diff(co, u)) for u in us] for co in coeffs])
    kerdim = d - Mker.rank()
    # (V2) contraction identity, random u  (needs r >= 2 so that d_u f has r-1>=1)
    v2, nskip = None, 0
    if r >= 2:
        svg0 = [[sp.Symbol(f'v{k}_{i}') for i in range(d)] for k in range(r - 1)]
        for _ in range(30):
            uvec = [sp.Integer(random.randint(-3, 3)) for _ in range(d)]
            if all(t == 0 for t in uvec):
                continue
            # FIRST check u is outside ker(Phi): P_f(u,-) must not vanish
            sub = {sv[0][i]: uvec[i] for i in range(d)}
            for k in range(1, r):
                for i in range(d):
                    sub[sv[k][i]] = svg0[k - 1][i]
            Qc = sp.expand(Q.subs(sub, simultaneous=True))
            if Qc == 0:
                nskip += 1
                continue                      # u in ker(Phi): d_u f is NOT persistent
            g = sp.expand(sum(uvec[i] * sp.diff(f, xs[i]) for i in range(d)))
            try:
                rg, svg, Hg, cg, Qg = certificate(g, xs, n - 1)
            except Exception as e:
                v2 = f'certificate(d_u f) failed OFF ker(Phi) for u={uvec}: {type(e).__name__}: {e}'
                break
            Qg = Qg.subs({svg[k][i]: svg0[k][i] for k in range(r - 1) for i in range(d)},
                         simultaneous=True)
            ratio = sp.simplify(sp.cancel(sp.expand(Qg) / Qc))
            v2 = (uvec, ratio, ratio.is_number, nskip)
            break
    if verbose:
        print(f'  {name[:44]:44s} d={d} r={r}  P_f symmetric={sym} alternating={alt}  '
              f'cond(a-multi)={cond_a}  ker(Phi) dim={kerdim} (proper: {kerdim < d})')
        if v2 is not None:
            print(f'      contraction: u={v2[0] if not isinstance(v2, str) else v2}  '
                  + ('' if isinstance(v2, str) else
                     f"P_(d_u f) / P_f(u,-) = {v2[1]}  constant: {v2[2]}  (u in ker(Phi) skipped: {v2[3]})"))
    return dict(name=name, d=d, r=r, sym=sym, alt=alt, cond_a=cond_a,
                kerdim=kerdim, v2=v2)


print('\n--- ten named forms ---')
rows = [analyse(nm, f, xs, n) for nm, f, xs, n in named_cases()]

print('\n--- a sample of hunt forms (every 13th of the (4,4) block, all r=2) ---')
blk, forms = None, []
for L in open(HUNT):
    m = re.match(r'### scan d=(\d+) n=(\d+)', L)
    if m:
        blk = (int(m.group(1)), int(m.group(2)))
    m = re.match(r'\s+PERSISTENT: f = (.*?)\s\s+dimNsp', L)
    if m and blk == (4, 4):
        forms.append(m.group(1))
xs = list(sp.symbols('x0:4'))
env = {str(v): v for v in xs}
for s in forms[::13]:
    rows.append(analyse(s, sp.sympify(s, locals=env), xs, 4))

ok = [r for r in rows if r['r'] >= 1]
print('\n=========== TALLY (3a support) ===========')
print(f'  forms analysed              : {len(rows)}')
print(f'  P_f symmetric in its slots  : {sum(1 for r in rows if r["sym"])}/{len(rows)}')
print(f'  P_f alternating (r>=2 only) : {sum(1 for r in rows if r["alt"])}/{len(rows)}')
print(f'  condition (a-multi) holds   : {sum(1 for r in rows if r["cond_a"])}/{len(rows)}')
print(f'  ker(Phi) a PROPER subspace  : {sum(1 for r in rows if r["kerdim"] < r["d"])}/{len(rows)}')
c2 = [r for r in rows if r['r'] >= 2 and r['v2'] is not None and not isinstance(r['v2'], str)]
print(f'  contraction identity tested : {len(c2)};  ratio constant in every case: '
      f'{all(r["v2"][2] for r in c2)}')
print(f'  ratios seen                 : {sorted(set(str(r["v2"][1]) for r in c2))}')
