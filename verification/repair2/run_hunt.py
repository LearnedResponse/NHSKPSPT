"""PART 2(b),(c) -- the three gates on every certificate-passing form the repair1
hunt found.

PROVENANCE: the forms are READ from repair1/out_hunt.txt (the hunt printed each
one on a `PERSISTENT: f = ...` line), not re-enumerated.  This is safe because
gates.py re-runs `certificate()` on every form from scratch, so any misparse or
non-persistent form raises and is reported as an error rather than silently
passing.  The (4,4) block is the completed exhaustive sweep of coefficient-one
sums of at most three distinct quartic monomials in four variables (7175 forms
scanned, 156 certificate-passing); the (4,5) block is the partial accumulation of
quintics in four variables at the time the hunt process was frozen.

usage:  python3 run_hunt.py <d> <n>
"""
import os, re, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gates import report_case                                 # noqa: E402

HUNT = os.path.join(os.path.dirname(HERE), 'repair1', 'out_hunt.txt')

want_d, want_n = int(sys.argv[1]), int(sys.argv[2])

forms, blk = [], None
for L in open(HUNT):
    m = re.match(r'### scan d=(\d+) n=(\d+)', L)
    if m:
        blk = (int(m.group(1)), int(m.group(2)))
    m = re.match(r'\s+PERSISTENT: f = (.*?)\s\s+dimNsp', L)
    if m and blk == (want_d, want_n):
        forms.append(m.group(1))

print('=' * 78)
print(f'PART 2  TR / JC / CK on the hunt block (d,n) = ({want_d},{want_n}):  '
      f'{len(forms)} certificate-passing forms read from out_hunt.txt')
print('=' * 78)

xs = list(sp.symbols(f'x0:{want_d}'))
env = {str(v): v for v in xs}
rows = []
for idx, s in enumerate(forms, 1):
    f = sp.sympify(s, locals=env)
    r = report_case(f'{s}', f, xs, want_n, verbose=False, do_nilT=True)
    rows.append(r)
    if 'error' in r:
        print(f'[{idx:3d}] {s:52s}  ERROR {r["error"]}')
    else:
        flag = ''
        if not r['JC']:
            flag = '   <<<<<< JC FAILS'
        elif not r['TR']:
            flag = '   <<<<<< TR FAILS'
        elif not r['CK']:
            flag = '   <<<<<< CK FAILS'
        print(f'[{idx:3d}] {s:52s} dimN={r["dimN"]} TR={r["TR"]} JC={r["JC"]} '
              f'CK={r["CK"]}({r["ckdim"]}) Nil={r["Nil"]} T={r["T"]}{flag}')
        if not r['JC']:
            for pair, S, res in r['_gates']['jc_fail']:
                print(f'      pair {pair}: S = {S.tolist()}')
                print(f'      residual   = {None if res is None else res.T.tolist()}')
                for k, M in enumerate(r['_info']['basis'], 1):
                    print(f'      N_{k} = {M.tolist()}')
    sys.stdout.flush()

ok = [r for r in rows if 'error' not in r]
print(f'\n=========== TALLY  (d,n)=({want_d},{want_n}) ===========')
print(f'forms processed        : {len(rows)}')
print(f'errors                 : {len(rows) - len(ok)}')
print(f'TR pass                : {sum(1 for r in ok if r["TR"])}/{len(ok)}')
print(f'JC pass                : {sum(1 for r in ok if r["JC"])}/{len(ok)}')
print(f'CK pass                : {sum(1 for r in ok if r["CK"])}/{len(ok)}')
print(f'Nil pass               : {sum(1 for r in ok if r["Nil"])}/{len(ok)}')
print(f'T pass                 : {sum(1 for r in ok if r["T"])}/{len(ok)}')
from collections import Counter
print(f'dim N_f distribution   : {dict(sorted(Counter(r["dimN"] for r in ok).items()))}')
print(f'common-kernel dim distr: {dict(sorted(Counter(r["ckdim"] for r in ok).items()))}')
print(f'diagonal base found    : {len(ok)}/{len(rows)}')
