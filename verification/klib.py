"""Kronecker product of symmetric tensors (Gharahi, arXiv:2608.11182, Remark 5)
   plus polarization / Hessian helpers.  Pure sympy, exact rationals."""
import itertools, sympy as sp
from sympy import factorial, prod


def nth_derivs(f, xs, n):
    """dict: index-tuple -> (1/n!) d^n f / dx_{i1}...dx_{in}  (a constant for deg-n f)."""
    d = len(xs)
    out = {}
    for it in itertools.product(range(d), repeat=n):
        e = [0] * d
        for i in it:
            e[i] += 1
        key = tuple(e)
        if key not in out:
            g = f
            for i in it:
                g = sp.diff(g, xs[i])
            out[key] = sp.simplify(g) / factorial(n)
    # re-key by index tuple
    res = {}
    for it in itertools.product(range(d), repeat=n):
        e = [0] * d
        for i in it:
            e[i] += 1
        res[it] = out[tuple(e)]
    return res


def kron(f, xs, g, ys, n, Z):
    """f box g  in Sym^n(V (x) W).  Z[i][j] are the z_{ij} coordinates.
       Remark 5:  f|X|g = (1/(n!)^2) sum (d^n f)(d^n g) z_{i1j1}...z_{injn}."""
    A = nth_derivs(f, xs, n)
    B = nth_derivs(g, ys, n)
    d1, d2 = len(xs), len(ys)
    tot = 0
    for it in itertools.product(range(d1), repeat=n):
        a = A[it]
        if a == 0:
            continue
        for jt in itertools.product(range(d2), repeat=n):
            b = B[jt]
            if b == 0:
                continue
            tot += a * b * prod([Z[i][j] for i, j in zip(it, jt)])
    return sp.expand(tot)


def dir_deriv(p, vars_, coeffs):
    return sp.expand(sum(c * sp.diff(p, v) for c, v in zip(coeffs, vars_)))


def polarize(p, vars_, dirs):
    """r-fold partial polarization p_{U^(1),...,U^(r)}."""
    q = p
    for c in dirs:
        q = dir_deriv(q, vars_, c)
    return q


def hess_det(p, vars_):
    H = sp.Matrix(len(vars_), len(vars_),
                  lambda i, j: sp.diff(p, vars_[i], vars_[j]))
    return H, sp.expand(H.det(method='berkowitz'))


def flat_Z(Z):
    return [Z[i][j] for i in range(len(Z)) for j in range(len(Z[0]))]
