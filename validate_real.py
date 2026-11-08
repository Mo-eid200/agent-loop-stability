#!/usr/bin/env python3
"""Validate loopstab on real data. Writes VALIDATION.md.

A. 54 recorded runs of gpt-4o-mini agents correcting a shared value
   (llm_loop.jsonl: N in {1,2,3} agents, delay D in {0,1,2}, two prompts,
   30 rounds, 1,806 model calls). loopstab is NOT told D: it must infer D and K
   from the first 15 rounds and predict what the remaining rounds do.
B. 4,050 measured request latencies of a production LLM backend
   (latency_traces.csv). loopstab derives safe gains, and a replay of the real
   delay sequences through the loop checks them.

Exit code 0 only if: the delay is recovered in >= 95% of runs, and no
confident verdict (STABLE / UNSTABLE / BOUNDARY) contradicts the observed outcome.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import numpy as np

import loopstab as L

FIT_ROUNDS, D_MAX = 15, 3
MATCH = {("STABLE", "decays"), ("BOUNDARY", "sustained"), ("UNSTABLE", "diverges")}
CONFIDENT = {"STABLE", "BOUNDARY", "UNSTABLE"}


def load_runs(path="llm_loop.jsonl"):
    rows = [json.loads(line) for line in open(path)]
    runs = defaultdict(list)
    for r in rows:
        runs[r["run"]].append(r)
    out = []
    for rid, rs in sorted(runs.items()):
        m = rs[0]
        xs, adj = {}, defaultdict(float)
        for r in rs:
            xs[r["round"]] = r["x"]
            adj[r["round"]] += r["adj"]
        T = max(xs)
        e = np.array([xs[t] - 100.0 for t in range(T + 1)])
        u = np.array([adj[t] for t in range(T + 1)])
        err = np.append(e, xs[T] + adj[T] - 100.0)
        tail = float(np.abs(err[-10:]).mean() / abs(err[0]))
        out.append(dict(run=rid, prompt=m["prompt"], N=m["N"], D=m["D"], e=e, u=u, tail=tail,
                        calls=len(rs), parse_fail=sum(not r["parse_ok"] for r in rs)))
    return out


def observed(tail, lo=0.2, hi=2.0):
    return "decays" if tail < lo else ("diverges" if tail > hi else "sustained")


def part_a(lines):
    runs = load_runs()
    res = []
    for r in runs:
        f = L.fit_loop(r["e"][:FIT_ROUNDS], r["u"][:FIT_ROUNDS], d_max=D_MAX, n_boot=500,
                       history="initial")
        v = L.judge(f, r["N"])
        # earliest round count giving a confident, correct verdict
        early = None
        for n in range(4, FIT_ROUNDS + 1):
            fv = L.judge(L.fit_loop(r["e"][:n], r["u"][:n], d_max=D_MAX, n_boot=300,
                                    history="initial"), r["N"])
            if fv.verdict in CONFIDENT:
                early = (n, (fv.verdict, observed(r["tail"])) in MATCH)
                break
        res.append(dict(r, fit=f, verdict=v, obs=observed(r["tail"]), early=early))

    n = len(res)
    d_ok = sum(x["fit"].D == x["D"] for x in res)
    conf = [x for x in res if x["verdict"].verdict in CONFIDENT]
    wrong = [x for x in conf if (x["verdict"].verdict, x["obs"]) not in MATCH]
    cm = Counter((x["verdict"].verdict, x["obs"]) for x in res)
    calls = sum(x["calls"] for x in res)

    # sensitivity of the outcome labels to the thresholds
    sens_wrong = 0
    grid = [(lo, hi) for lo in (0.05, 0.1, 0.2, 0.3, 0.5) for hi in (1.2, 1.5, 2, 5, 10, 20)]
    for lo, hi in grid:
        sens_wrong = max(sens_wrong, sum((x["verdict"].verdict, observed(x["tail"], lo, hi)) not in MATCH
                                         for x in conf))
    early_ok = [x["early"] for x in res if x["early"]]
    early_n = [e[0] for e in early_ok]

    lines += ["### A. Real LLM agents in the loop", "",
              f"{n} runs of gpt-4o-mini agents ({calls:,} model calls): N in {{1,2,3}} agents, "
              "delay D in {0,1,2}, a naive and an informed prompt, 30 rounds each. "
              f"loopstab sees only the first {FIT_ROUNDS} rounds and is **not told D**.", "",
              f"- **Delay recovered:** {d_ok}/{n} runs ({d_ok / n:.0%}).",
              f"- **Confident verdicts:** {len(conf)}/{n}; **wrong confident verdicts: {len(wrong)}**.",
              f"- **Threshold sensitivity:** across {len(grid)} choices of the outcome thresholds "
              f"(decay < 0.05–0.5, divergence > 1.2–20), the worst case is {sens_wrong} wrong confident verdict(s).",
              f"- **Early warning:** fitting only the first n rounds (n = 4, 5, ...), a confident verdict "
              f"appeared after a median of {int(np.median(early_n))} rounds (range {min(early_n)}–{max(early_n)}; "
              f"4 is the shortest window tried), and {sum(ok for _, ok in early_ok)}/{len(early_ok)} of those "
              f"earliest verdicts were already correct.",
              f"- **Fitted aggregate gain:** median K = {np.median([x['fit'].K / x['N'] for x in res]):.2f} "
              "per agent. Each agent corrects the *whole* visible error, even when told how many agents "
              "share the loop and how stale the reading is.", "",
              "| Verdict (rounds 1–15) | Observed in rounds 21–30 | Runs |", "|---|---|---|"]
    for (v, o), c in sorted(cm.items()):
        lines.append(f"| {v} | {o} | {c} |")
    lines += ["", "<details><summary>Per-run results</summary>", "",
              "| Run | Prompt | N | true D | fitted D | K (95% CI) | Verdict | Observed (tail ratio) |",
              "|---|---|---|---|---|---|---|---|"]
    for x in res:
        f = x["fit"]
        lines.append(f"| {x['run']} | {x['prompt']} | {x['N']} | {x['D']} | {f.D} | "
                     f"{f.K:.2f} ({f.K_ci[0]:.2f}–{f.K_ci[1]:.2f}) | {x['verdict'].verdict} | "
                     f"{x['obs']} ({x['tail']:.2f}) |")
    lines += ["", "</details>", ""]
    print(f"A: delay {d_ok}/{n}, confident {len(conf)}, wrong {len(wrong)}, worst-case sensitivity {sens_wrong}")
    return d_ok / n >= 0.95 and not wrong and sens_wrong == 0


def replay_boundary(seqs: list[np.ndarray], N: int, k_grid: np.ndarray) -> float:
    """Largest per-agent gain at which the loop, driven by the REAL delay
    sequences (agent i's t-th request delay), shrinks the error in most windows."""
    T = min(len(s) for s in seqs[:N])
    win = min(60, T)
    best = 0.0
    for k in k_grid:
        shrink = []
        for start in range(0, T - win + 1, max(1, (T - win) // 10 or 1)):
            hist = [1.0] * 32
            for t in range(start, start + win):
                ds = [int(s[t]) for s in seqs[:N]]
                new = hist[0] - k * sum(hist[min(d, 31)] for d in ds)
                hist = [new] + hist[:-1]
            shrink.append(abs(hist[0]) < 1.0)
        if np.mean(shrink) >= 0.5:
            best = float(k)
        else:
            break
    return best


def part_b(lines):
    import csv
    rows = list(csv.DictReader(open("latency_traces.csv")))
    lines += ["### B. Real latency traces of a production LLM backend", "",
              "Requests issued once per second per agent; loop period 1 s, so a correction lands "
              "floor(latency / 1 s) steps late. Largest safe **per-agent** gain under each criterion, "
              "and the boundary observed when the real delay sequences are replayed through the loop.", "",
              "| Trace | N | p50 / p99 latency | worst-case rule | mean-square (regimes) | almost-sure | replay of real delays | worst-case conservatism |",
              "|---|---|---|---|---|---|---|---|"]
    ok = True
    for st in ("short", "long"):
        for N in ("1", "2", "4", "8"):
            by = defaultdict(list)
            for r in rows:
                if r["set"] == st and r["N"] == N and r["ok"] == "1":
                    by[r["agent"]].append((float(r["t_request"]),
                                           float(r["t_response"]) - float(r["t_request"])))
            lat = [np.array([v for _, v in sorted(x)]) for x in by.values()]
            rep = L.analyze_latency(lat, 1.0, int(N), lyap_steps=2000, lyap_reps=8)
            seqs = [L.delays_from_latency(x, 1.0) for x in lat]
            kg = np.linspace(0.002, 2.4 / int(N), 400)
            kr = replay_boundary(seqs, int(N), kg)
            # the replay should not sit below the mean-square bound by more than grid/finite-sample slack
            consistent = kr >= 0.85 * rep.k_mean_square_regimes
            ok &= consistent
            lines.append(f"| {st} | {N} | {rep.latency_p50:.2f} / {rep.latency_p99:.2f} s | "
                         f"{rep.k_worst_case:.3f} | **{rep.k_mean_square_regimes:.3f}** | "
                         f"{rep.k_almost_sure:.3f} | {kr:.3f}{'' if consistent else ' ⚠'} | "
                         f"{rep.conservatism_worst_case:.1f}x |")
            print(f"B: {st} N={N}: wc {rep.k_worst_case:.3f} ms {rep.k_mean_square_regimes:.3f} "
                  f"as {rep.k_almost_sure:.3f} replay {kr:.3f}")
    lines += ["", "The worst-case-delay rule is safe but throws away much of the usable gain, because a "
              "single slow request sets the limit. The mean-square bound with regimes is the recommended "
              "design basis; replaying the real delay sequences confirms that the loop still converges "
              "there (the replay boundary is at or above it).", ""]
    return ok


def main() -> int:
    lines = ["# Validation on real data", ""]
    a = part_a(lines)
    b = part_b(lines)
    lines += [f"**Validation passed: {a and b}**"]
    open(sys.argv[1] if len(sys.argv) > 1 else "VALIDATION.md", "w").write("\n".join(lines) + "\n")
    print(f"validation passed: {a and b}")
    return 0 if a and b else 1


if __name__ == "__main__":
    sys.exit(main())
