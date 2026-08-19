"""repair1: the normalized Hessian family N_f, and the two hypotheses

    Nil(f) : every element of the LINEAR SPAN  Nsp(f) = span{ N_f(A) : A a tuple }
             is nilpotent   (equivalently: det is the d-th power of the linear
             functional lambda_f on  L_f = C.I + Nsp(f) )
    T(f)   : Nsp(f) is SIMULTANEOUSLY STRICTLY TRIANGULARIZABLE
             (equivalently: the associative algebra it generates is nilpotent)

Definitions (see PROP10_REPAIR_ATTEMPT_1.md, section 2):
    r = n-2;  H_f(v^(1),..,v^(r)) = Hessian in x of  d_{v^(1)}..d_{v^(r)} f(x)
    (symmetric multilinear in the r slots, entries constant in x)
    certificate:  det H_f(A) = [P_f(A)]^{d}
    base tuple B with det H_f(B) != 0
    M_f(A) = H_f(B)^{-1} H_f(A),  lambda_f(A) = P_f(A)/P_f(B),
    N_f(A) = M_f(A) - lambda_f(A) I.

Because N_f is multilinear in the r slots, Nsp(f) = span{ N_f(A) : A ranges over
r-tuples of a fixed basis of V }, a FINITE and exactly computable generating set.
Everything below is exact over Q (or Q adjoin a root, handled by keeping
P_f only up to the d-th root of a constant -- ratios are all we ever use).
"""
import itertools, sys, os
import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def polarized_hessian(f, xs, r, slotvars):
    """H_f(v^(1),..,v^(r)) as a sympy Matrix with entries in C[slotvars]."""
    q = f
    for k in range(r):
        q = sp.expand(sum(slotvars[k][i] * sp.diff(q, xs[i]) for i in range(len(xs))))
    d = len(xs)
    return sp.Matrix(d, d, lambda i, j: sp.expand(sp.diff(q, xs[i], xs[j])))


def certificate(f, xs, n):
    """return (r, slotvars, H, cP, Q) with det H = cP * Q**d, Q multidegree (1,..,1).
       P_f = cP**(1/d) * Q ; only ratios P_f(A)/P_f(B) = Q(A)/Q(B) are used."""
    d, r = len(xs), n - 2
    slotvars = [[sp.Symbol(f'v{k}_{i}') for i in range(d)] for k in range(r)]
    H = polarized_hessian(f, xs, r, slotvars)
    D = sp.expand(H.det(method='berkowitz'))
    if D == 0:
        raise ValueError('polarized Hessian identically singular -> not persistent')
    c, facs = sp.factor_list(D)
    Q = sp.Integer(1)
    for b, e in facs:
        if e % d:
            raise ValueError(f'det H is NOT a perfect {d}-th power: exponent {e} on {b}')
        Q *= b ** (e // d)
    Q = sp.expand(Q)
    for k in range(r):
        for mono in sp.Poly(Q, *slotvars[k]).monoms():
            if sum(mono) != 1:
                raise ValueError(f'P_f not multilinear in slot {k}: {Q}')
    assert sp.expand(c * Q**d - D) == 0
    return r, slotvars, H, c, Q


def subs_tuple(expr_or_mat, slotvars, A):
    sub = {}
    for k, vec in enumerate(A):
        for i, val in enumerate(vec):
            sub[slotvars[k][i]] = val
    return expr_or_mat.subs(sub)


def find_base(H, Q, slotvars, d, r):
    """first r-tuple of standard basis vectors with det H != 0; else search small ints."""
    cands = []
    for combo in itertools.product(range(d), repeat=r):
        cands.append([[sp.Integer(1) if i == c else sp.Integer(0) for i in range(d)]
                      for c in combo])
    for combo in itertools.product([0, 1, -1, 2], repeat=d):
        if any(combo):
            cands.append([[sp.Integer(t) for t in combo]] * r)
    for A in cands:
        HB = sp.Matrix(subs_tuple(H, slotvars, A))
        if sp.expand(HB.det()) != 0:
            return A, HB
    raise ValueError('no base tuple found')


def nspace(f, xs, n, verbose=True):
    """returns dict with generators of Nsp(f) and the two verdicts."""
    d = len(xs)
    r, slotvars, H, c, Q = certificate(f, xs, n)
    A0, HB = find_base(H, Q, slotvars, d, r)
    QB = subs_tuple(Q, slotvars, A0)
    HBi = HB.inv()
    gens = []
    for combo in itertools.product(range(d), repeat=r):
        A = [[sp.Integer(1) if i == cc else sp.Integer(0) for i in range(d)]
             for cc in combo]
        HA = sp.Matrix(subs_tuple(H, slotvars, A))
        lam = sp.simplify(subs_tuple(Q, slotvars, A) / QB)
        N = sp.expand(HBi * HA - lam * sp.eye(d))
        if not N.is_zero_matrix:
            gens.append(sp.Matrix(N))
    gens = reduce_span(gens, d)
    if verbose:
        print(f'  r = n-2 = {r}, d = {d}')
        print(f'  det H_f = ({c}) * ({Q})**{d}   [certificate OK]')
        print(f'  base tuple = {[list(map(str,v)) for v in A0]}')
        print(f'  dim Nsp(f) = {len(gens)}')
        for N in gens:
            print(f'    N = {N.tolist()}')
    return dict(r=r, d=d, gens=gens, Q=Q, c=c, slotvars=slotvars, H=H, base=A0, HB=HB)


def reduce_span(mats, d):
    """row-reduce a list of dxd matrices to a basis of their span."""
    if not mats:
        return []
    M = sp.Matrix([list(m) for m in mats])
    rr, piv = M.rref()
    return [sp.Matrix(d, d, list(rr.row(i))) for i in range(len(piv))]


def check_nil(gens, d):
    """Nil: is EVERY element of the span nilpotent?  Exact: generic combination."""
    if not gens:
        return True, 'Nsp = 0'
    cs = sp.symbols(f'c0:{len(gens)}')
    Ngen = sp.zeros(d, d)
    for ci, N in zip(cs, gens):
        Ngen += ci * N
    Ngen = sp.expand(Ngen)
    t = sp.Symbol('t')
    cp = sp.expand(sp.factor(Ngen.charpoly(t).as_expr()))
    return sp.expand(cp - t**d) == 0, sp.factor(cp)


def check_T(gens, d):
    """T: is the associative algebra generated by span(gens) nilpotent?
       (<=> simultaneous STRICT triangularizability, over an alg. closed field)"""
    if not gens:
        return True, 0
    cur = list(gens)
    for step in range(1, d + 2):
        if all(sp.expand(M).is_zero_matrix for M in cur):
            return True, step
        prods = [sp.expand(Aa * Bb) for Aa in cur for Bb in gens]
        prods = [P for P in prods if not P.is_zero_matrix]
        cur = reduce_span(prods, d)
        if not cur:
            return True, step + 1
    return False, None


def report(name, f, xs, n):
    print(f'\n### {name}   (n={n}, d={len(xs)})')
    try:
        info = nspace(f, xs, n)
    except Exception as e:
        print(f'  !! {type(e).__name__}: {e}')
        return None
    nil, cp = check_nil(info['gens'], info['d'])
    T, step = check_T(info['gens'], info['d'])
    print(f'  charpoly of generic element of Nsp: {cp}')
    print(f'  Nil(f)  [every element of span nilpotent] : {nil}')
    print(f'  T(f)    [simultaneously strictly triangularizable] : {T}'
          + (f' (algebra dies at word length {step})' if T else ''))
    return dict(name=name, n=n, d=len(xs), r=info['r'],
                dimN=len(info['gens']), nil=nil, T=T)
