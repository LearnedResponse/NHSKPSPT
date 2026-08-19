#!/usr/bin/env python3
"""Fast-tier reproducibility check: rerun the quick deterministic scripts
and byte-compare against the committed out_*.txt snapshots.
Usage: verify_all.py [--list]"""
import subprocess, sys, os, pathlib

HERE = pathlib.Path(__file__).resolve().parent
FAST = [  # (script, snapshot) - each finishes in seconds to ~2 min
    ("check_a.py", "out_a.txt"),
    ("check_b.py", "out_b.txt"),
    ("check_cauchy.py", "out_cauchy.txt"),
    ("check_ex13.py", "out_ex13.txt"),
    ("check_tri.py", "out_tri.txt"),
    ("repair1/e2e.py", "repair1/out_e2e.txt"),
    ("repair2/jstats.py", "repair2/out_jstats.txt"),
    ("repair2/dedup_census.py", "repair2/out_dedup_census.txt"),
]
LONG = ["run_probes.py", "repair1/e2e_more.py", "repair1/e2e_r3.py",
        "repair1/contract.py", "repair1/closure_oneslot.py", "repair1/run_T.py",
        "repair1/toy_sep.py", "repair2/run_named.py", "repair2/controls.py",
        "repair2/veronese.py", "repair2/run_kron.py", "repair2/run_hunt.py",
        "repair1/hunt.py"]

def main():
    if "--list" in sys.argv:
        print("fast tier (run + byte-compare):");  [print("  ", s) for s, _ in FAST]
        print("long tier (run manually; deterministic, hours for the sweeps):")
        [print("  ", s) for s in LONG]
        return 0
    fails = 0
    for script, snap in FAST:
        r = subprocess.run([sys.executable, str(HERE / script)],
                           capture_output=True, text=True, cwd=HERE)
        want = (HERE / snap).read_text()
        ok = (r.returncode == 0) and (r.stdout == want)
        print(f"{'OK  ' if ok else 'FAIL'} {script}  vs {snap}")
        fails += (not ok)
    print(f"{len(FAST)-fails}/{len(FAST)} fast-tier scripts reproduce their snapshots byte-for-byte")
    return 1 if fails else 0

if __name__ == "__main__":
    sys.exit(main())
