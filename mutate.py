#!/usr/bin/env python3
"""Mutation testing: plant one realistic bug at a time in loopstab.py and check
that tests.py catches it. A surviving mutation would mean an untested behaviour.

Run:  python3 mutate.py [MUTATION_REPORT.md]
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

MUTATIONS = [
    ("kappa off by one", "(2 * D + 1)", "(2 * D + 2)"),
    ("companion sign flipped", "A[0, d] -= g", "A[0, d] += g"),
    ("mean-square: same-agent term dropped", "+ N * EE))", "+ 0 * EE))"),
    ("mean-square: cross term N^2 instead of N(N-1)", "N * (N - 1) * np.kron(Ebar, Ebar)",
     "N * N * np.kron(Ebar, Ebar)"),
    ("regime matrix transposed", "P[i, j] * blocks[i]", "P[j, i] * blocks[i]"),
    ("latency rounded instead of floored", "np.floor(np.asarray(lat_s) / period_s)",
     "np.round(np.asarray(lat_s) / period_s)"),
    ("robust fit replaced by plain mean of ratios",
     "K = _wmedian(-y[m] / x[m], np.abs(x[m]) / scale[m])", "K = float(np.mean(-y[m] / x[m]))"),
    ("pre-trace history padded with zeros", "np.full(d_max, e[0])", "np.zeros(d_max)"),
    ("verdict ignores the confidence interval", "if hi < kap_lo * (1 - band):",
     "if fit.K < kap * (1 - band):"),
    ("bootstrap does not re-select the delay",
     "b, _ = select(X[idx], y[idx], scale[idx], usable)",
     "b, _ = select(X[idx], y[idx], scale[idx], [D])"),
    ("boundary search returns the first unstable gain", "            hi = mid\n    return lo",
     "            hi = mid\n    return hi + 0.01"),
]


def main() -> int:
    src = (HERE / "loopstab.py").read_text()
    rows, all_killed = [], True
    for name, old, new in MUTATIONS:
        if src.count(old) != 1:
            print(f"!! mutation '{name}' does not apply ({src.count(old)} matches)")
            return 2
        with tempfile.TemporaryDirectory() as td:
            for f in ("tests.py", "latency_traces.csv"):
                shutil.copy(HERE / f, td)
            (Path(td) / "loopstab.py").write_text(src.replace(old, new))
            r = subprocess.run([sys.executable, "tests.py"], cwd=td, capture_output=True,
                               text=True, timeout=900)
        failed = re.findall(r"^(T\d+)\s+FAIL", r.stdout, re.M)
        killed = bool(failed) or r.returncode != 0
        all_killed &= killed
        rows.append((name, killed, failed))
        print(f"{name:52s} {'KILLED' if killed else 'SURVIVED'}  by {', '.join(failed) or '-'}",
              flush=True)
    lines = ["# Mutation testing", "",
             "Each row plants one realistic bug in `loopstab.py` and reruns the full test suite.", "",
             "| Planted bug | Result | Caught by |", "|---|---|---|"]
    lines += [f"| {n} | **{'KILLED' if k else 'SURVIVED'}** | {', '.join(f) or '-'} |" for n, k, f in rows]
    lines += ["", f"All mutations killed: **{all_killed}**"]
    out = sys.argv[1] if len(sys.argv) > 1 else "MUTATION_REPORT.md"
    Path(out).write_text("\n".join(lines) + "\n")
    print(f"\nall killed: {all_killed}")
    return 0 if all_killed else 1


if __name__ == "__main__":
    sys.exit(main())
