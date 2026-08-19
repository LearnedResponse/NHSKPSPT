"""PART 2(a) -- the three gates on the ten named normal forms of repair1's test set."""
import os, sys
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gates import named_cases, report_case                    # noqa: E402

print('=' * 78)
print('PART 2(a)  TR / JC / CK on the ten named normal forms  (diagonal base)')
print('=' * 78)
rows = []
for name, f, xs, n in named_cases():
    print(f'\n### {name}   (n={n}, d={len(xs)})')
    r = report_case(name, f, xs, n)
    rows.append(r)
    if '_gates' in r:
        for k, M in enumerate(r['_info']['basis'], 1):
            print(f'    N_{k} = {M.tolist()}')

print('\n\n=========== SUMMARY (a) ===========')
hdr = f'{"case":38s} {"n":>2s} {"d":>2s} {"r":>2s} {"dimN":>4s} {"TR":>5s} {"JC":>5s} {"CK":>5s} {"ckdim":>5s} {"Nil":>5s} {"T":>5s}'
print(hdr)
for r in rows:
    if 'error' in r:
        print(f'{r["name"][:38]:38s}  ERROR {r["error"]}')
        continue
    print(f'{r["name"][:38]:38s} {r["n"]:2d} {r["d"]:2d} {r["r"]:2d} {r["dimN"]:4d} '
          f'{str(r["TR"]):>5s} {str(r["JC"]):>5s} {str(r["CK"]):>5s} {r["ckdim"]:5d} '
          f'{str(r.get("Nil")):>5s} {str(r.get("T")):>5s}')
ok = [r for r in rows if 'error' not in r]
print(f'\nTALLY (a): {len(ok)} cases; '
      f'TR pass {sum(1 for r in ok if r["TR"])}/{len(ok)}; '
      f'JC pass {sum(1 for r in ok if r["JC"])}/{len(ok)}; '
      f'CK pass {sum(1 for r in ok if r["CK"])}/{len(ok)}')
