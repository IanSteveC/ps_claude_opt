#!/usr/bin/env python3
"""Compare a period_search_out against a reference, CR-stripped.

Reports, per file: total lines, identical lines, and for every differing
line the per-field deltas. Fields: period rms chisq darkarea lambda beta.
Correctness bar: darkarea/lambda/beta must match EXACTLY on every line;
period/rms/chisq differences are reported with worst-case magnitudes
(BOINC validator tolerance is 0.1/0.1/0.5 but we track raw deltas).
Exit 1 if any pole/darkarea field differs or line counts mismatch.
"""
import sys

def load(p):
    with open(p, 'rb') as f:
        return [l.decode().replace('\r', '').rstrip('\n') for l in f.read().splitlines()]

def main(test, ref):
    t, r = load(test), load(ref)
    if len(t) != len(r):
        print(f"LINE COUNT MISMATCH {len(t)} vs {len(r)}")
        return 1
    ident = 0
    worst = [0.0, 0.0, 0.0]
    pole_bad = 0
    diffs = []
    for i, (a, b) in enumerate(zip(t, r)):
        if a == b:
            ident += 1
            continue
        fa, fb = a.split(), b.split()
        if len(fa) != 6 or len(fb) != 6:
            print(f"  line {i+1}: field count off: {a!r} vs {b!r}")
            pole_bad += 1
            continue
        # pole + dark area: exact string match required
        if fa[3:] != fb[3:]:
            pole_bad += 1
            diffs.append((i + 1, a, b, True))
        else:
            diffs.append((i + 1, a, b, False))
        for k in range(3):
            worst[k] = max(worst[k], abs(float(fa[k]) - float(fb[k])))
    n = len(t)
    print(f"{n} lines: {ident} identical, {n-ident} differ; "
          f"worst |dP|={worst[0]:.3g} |dRMS|={worst[1]:.3g} |dchi2|={worst[2]:.3g}; "
          f"pole/darkarea mismatches: {pole_bad}")
    for ln, a, b, bad in diffs[:12]:
        tag = " POLE-DIFF" if bad else ""
        print(f"  line {ln}{tag}:\n    test {a}\n    ref  {b}")
    if len(diffs) > 12:
        print(f"  ... and {len(diffs)-12} more differing lines")
    return 1 if pole_bad else 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
