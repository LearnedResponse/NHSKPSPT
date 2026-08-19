#!/usr/bin/env python3
"""Derive the paper's published deduplicated census (320 distinct inputs,
2204 Jordan-pair tests) from the committed raw execution census in
out_jstats.txt.  jstats.py counts EXECUTIONS: 321 gate executions and 2214
pairs.  One literal input appears twice -- the named form GOEx28
(x0*x2^3 + x1*x2^2*x3 + x3^4, GO Example 28) is also one of the 156
certificate-passing forms of the (d=4, n=4) hunt block -- so the census of
DISTINCT literal polynomial inputs removes one execution and its 10 pairs.
Deterministic; reads only the committed snapshot beside this script."""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
text = open(os.path.join(HERE, "out_jstats.txt")).read()

named_sec = text.split("--- ten named forms ---")[1].split("--- equation count")[0]
named = re.findall(r"^  (\S+) .*?pairs=(\d+) equations=\d+$", named_sec, re.M)
assert len(named) == 10, f"expected 10 named forms, parsed {len(named)}"
named_pairs = sum(int(p) for _, p in named)
dup = dict(named).get("GOEx28")
assert dup is not None, "GOEx28 not found among named forms"
dup_pairs = int(dup)

hunt = re.findall(
    r"\(d,n\)=\(4, (\d)\): (\d+) forms, dim N distribution \{3: (\d+), 4: (\d+)\}", text)
assert len(hunt) == 2, f"expected 2 hunt blocks, parsed {len(hunt)}"
hunt_forms, hunt_pairs = 0, 0
for _n, forms, d3, d4 in hunt:
    forms, d3, d4 = int(forms), int(d3), int(d4)
    assert d3 + d4 == forms
    hunt_forms += forms
    hunt_pairs += d3 * 6 + d4 * 10   # dimN=3 -> C(3,2)+3 = 6 pairs; dimN=4 -> 10

kron_sec = text.split("--- Kronecker products")[1]
kron = re.findall(r"pairs=(\d+) equations=\d+$", kron_sec, re.M)
assert len(kron) == 4, f"expected 4 Kronecker products, parsed {len(kron)}"
kron_pairs = sum(int(p) for p in kron)

exec_inputs = 10 + hunt_forms + 4
exec_pairs = named_pairs + hunt_pairs + kron_pairs
print(f"raw execution census (as in jstats.py): "
      f"{exec_inputs} gate executions, {exec_pairs} Jordan-pair tests")
print(f"  named {named_pairs} + hunt {hunt_pairs} + kron {kron_pairs} pairs")
assert exec_inputs == 321 and exec_pairs == 2214, "raw census mismatch"

print(f"literal duplicate: GOEx28 appears in the named forms AND the (4,4) "
      f"hunt block; {dup_pairs} pairs counted twice")
distinct_inputs = exec_inputs - 1
dedup_pairs = exec_pairs - dup_pairs
print(f"deduplicated census (as published): "
      f"{distinct_inputs} distinct literal inputs, {dedup_pairs} Jordan-pair tests")
assert distinct_inputs == 320 and dedup_pairs == 2204, "published census mismatch"
print("OK: published 320 / 2204 derived from the committed raw census")
