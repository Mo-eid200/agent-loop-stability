#!/usr/bin/env python3
"""Verification suite for loopstab (no test framework needed).

Run:  python3 tests.py            all tests
      python3 tests.py T4 T8      selected tests
Exit code 0 only if every selected test passes.
"""
from __future__ import annotations

import csv
import itertools
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import loopstab as L

TESTS = {}


def test(fn):
    TESTS[fn.__name__.split("_")[0]] = fn
    return fn


# ----------------------------------------------------------------- exact theory
@test
def T1_kappa_matches_eigenvalues():
    """kappa(D) is exactly where the spectral radius crosses 1, D = 0..40."""
    for D in range(41):
        k = L.kappa(D)
        assert L.spectral_radius(L.companion({D: k * (1 - 1e-6)})) < 1, D
        assert L.spectral_radius(L.companion({D: k * (1 + 1e-6)})) > 1, D
    return "41 delays"


@test
def T2_analytic_test_equals_eigen_test():
    """3,000 random (K, D): the analytic verdict equals the eigenvalue verdict."""
    rng = np.random.default_rng(1)
    n = 0
    for _ in range(3000):
        D, K = int(rng.integers(0, 16)), float(rng.uniform(0.001, 3.0))
        if abs(K / L.kappa(D) - 1) < 1e-6:
            continue
        assert L.stable_fixed(K, D) == (L.spectral_radius(L.companion({D: K})) < 1), (K, D)
        n += 1
    return f"{n} cases"


@test
def T3_heterogeneous_rule_sufficient_and_tight():
    """sum k_i < kappa(max d_i) implies stability (2,000 random teams); equal
    delays make the rule tight."""
    rng = np.random.default_rng(2)
    for _ in range(2000):
        N = int(rng.integers(2, 6))
        ds = [int(x) for x in rng.integers(0, 9, N)]
        w = rng.dirichlet(np.ones(N))
        ks = list(w * L.kappa(max(ds)) * rng.uniform(0.01, 0.999))
        g = {}
        for d, k in zip(ds, ks):
            g[d] = g.get(d, 0) + k
        assert L.stable_heterogeneous(ks, ds)
        assert L.spectral_radius(L.companion(g, max(ds) + 1)) < 1, (ds, ks)
    for D in range(8):
        assert L.spectral_radius(L.companion({D: L.kappa(D) * 1.001})) > 1
    return "2000 teams"


@test
def T4_closed_form_ms_operator_equals_enumeration():
    """The closed-form E[A (x) A] equals brute-force enumeration over all joint delays."""
    rng = np.random.default_rng(3)
    for N in (1, 2, 3):
        for n in (1, 2, 3, 4):
            p = rng.dirichlet(np.ones(n))
            k = float(rng.uniform(0.05, 0.6))
            S = L.shift_matrix(n)
            brute = np.zeros((n * n, n * n))
            brute_mean = np.zeros((n, n))
            for ds in itertools.product(range(n), repeat=N):
                pr = float(np.prod([p[d] for d in ds]))
                A = S.copy()
                for d in ds:
                    A[0, d] -= k
                brute += pr * np.kron(A, A)
                brute_mean += pr * A
            assert np.allclose(brute, L.ms_operator(k, N, p), atol=1e-12)
            assert np.allclose(brute_mean, L.mean_operator(k, N, p), atol=1e-12)
    return "N = 1..3, n = 1..4"


def _mc_second_moment(k, N, p, steps, paths, rng, modes=None, P=None):
    n = len(p) if modes is None else max(len(q) for q in modes)
    hist = np.ones((paths, n))
    m = np.zeros(paths, dtype=int)
    for _ in range(steps):
        if modes is None:
            ds = rng.choice(len(p), size=(paths, N), p=p)
        else:
            m = (rng.random(paths) < P[m, 1]).astype(int)
            ds = np.empty((paths, N), dtype=int)
            for j in range(len(modes)):
                sel = m == j
                ds[sel] = rng.choice(len(modes[j]), size=(sel.sum(), N), p=modes[j])
        corr = np.take_along_axis(hist, ds, axis=1).sum(axis=1)
        new = hist[:, 0] - k * corr
        hist = np.concatenate([new[:, None], hist[:, :-1]], axis=1)
    return float((hist[:, 0] ** 2).mean()), float(hist[:, 0].mean())


@test
def T5_ms_and_mean_operators_match_monte_carlo():
    """E[e^2] and E[e] after 10 steps: operator prediction vs 400,000 simulated paths."""
    rng = np.random.default_rng(4)
    for N, p, k in ((1, np.array([0.5, 0.3, 0.2]), 0.30), (2, np.array([0.6, 0.3, 0.1]), 0.18),
                    (3, np.array([0.2, 0.5, 0.3]), 0.08)):
        n, steps = len(p), 10
        M, A = L.ms_operator(k, N, p), L.mean_operator(k, N, p)
        z = np.ones(n)
        Q = np.outer(z, z).reshape(-1)
        mu = z.copy()
        for _ in range(steps):
            Q, mu = M @ Q, A @ mu
        mc2, mc1 = _mc_second_moment(k, N, p, steps, 400_000, rng)
        assert abs(mc2 / Q[0] - 1) < 0.03, (N, mc2, Q[0])
        assert abs(mc1 - mu[0]) < 5 * np.sqrt(mc2 / 400_000) + 1e-12, (N, mc1, mu[0])
    return "3 configurations"


@test
def T6_markov_regime_operator():
    """(a) identical regimes reduce to the independent case; (b) persistent,
    asymmetric regimes match Monte Carlo (checks the P[i,j] orientation)."""
    p = np.array([0.7, 0.2, 0.1])
    for k in (0.2, 0.4, 0.6):
        P = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert abs(L.spectral_radius(L.mjls_ms_operator(k, 2, P, [p, p]))
                   - L.spectral_radius(L.ms_operator(k, 2, p))) < 1e-9
    rng = np.random.default_rng(5)
    fast, slow = np.array([0.9, 0.1, 0.0, 0.0]), np.array([0.0, 0.1, 0.3, 0.6])
    P = np.array([[0.95, 0.05], [0.30, 0.70]])
    k, N, steps = 0.18, 2, 10
    big = L.mjls_ms_operator(k, N, P, [fast, slow])
    n2 = 16
    z = np.outer(np.ones(4), np.ones(4)).reshape(-1)
    # the simulation starts in the fast regime and draws the regime of step 0 from P[0]
    Q = np.concatenate([P[0, 0] * z, P[0, 1] * z])
    for _ in range(steps):
        Q = big @ Q
    pred = Q[0] + Q[n2]
    mc2, _ = _mc_second_moment(k, N, None, steps, 400_000, rng, modes=[fast, slow], P=P)
    assert abs(mc2 / pred - 1) < 0.04, (mc2, pred)
    # the transposed operator (wrong orientation) must disagree
    bad = L.mjls_ms_operator(k, N, P.T.copy() / P.T.sum(axis=1, keepdims=True), [fast, slow])
    Qb = np.concatenate([P[0, 0] * z, P[0, 1] * z])
    for _ in range(steps):
        Qb = bad @ Qb
    assert abs(mc2 / (Qb[0] + Qb[n2]) - 1) > 0.10
    pred2 = pred
    return f"MC {mc2:.4g} vs operator {pred2:.4g}"


@test
def T7_boundary_search():
    """boundary() finds a known threshold."""
    for thr in (0.013, 0.5, 1.7, 3.99):
        b = L.boundary(lambda g: g < thr, 4.0, 1e-6)
        assert abs(b - thr) < 1e-5, (thr, b)
    return "4 thresholds"


@test
def T8_latency_to_delay():
    """A correction lands floor(latency / period) steps late."""
    d = L.delays_from_latency(np.array([0.0, 0.99, 1.0, 1.99, 2.0, 7.5]), 1.0)
    assert list(d) == [0, 0, 1, 1, 2, 7], d
    return "ok"


# ----------------------------------------------------------------- fitting
def _synthetic(K, D, T, rng, noise=0.02, outlier_rate=0.05, e0=40.0):
    e = np.zeros(T + 1)
    e[0] = e0
    u = np.zeros(T)
    for t in range(T):
        ed = e[t - D] if t - D >= 0 else e0
        u[t] = -K * ed * (1 + noise * rng.standard_normal()) + noise * e0 * rng.standard_normal()
        if rng.random() < outlier_rate:                       # LLM-style wild reply
            u[t] *= float(rng.choice([10.0, -1.0, 0.0]))
        e[t + 1] = e[t] + u[t]
        if not np.isfinite(e[t + 1]) or abs(e[t + 1]) > 1e12:
            return e[: t + 1], u[: t + 1]
    return e[:T], u


@test
def T9_fit_recovers_delay_and_gain_with_outliers():
    """300 synthetic loops with 5% wild corrections: when loopstab says the delay
    is identified it is right >= 98% of the time, overall >= 90%, wrong delays are
    flagged as uncertain >= 85% of the time, and the 95% CI covers the true K in
    88-99.5% of cases."""
    rng = np.random.default_rng(6)
    hit = cover = n = n_id = hit_id = wrong = wrong_flagged = 0
    for _ in range(300):
        D = int(rng.integers(0, 5))
        K = float(L.kappa(D) * rng.uniform(0.3, 1.6))
        e, u = _synthetic(K, D, 40, rng)
        f = L.fit_loop(e, u, d_max=5, n_boot=300, history="initial", seed=int(rng.integers(1e9)))
        if f.K is None:
            continue
        n += 1
        hit += f.D == D
        cover += f.K_ci[0] <= K <= f.K_ci[1]
        if f.identifiable:
            n_id += 1
            hit_id += f.D == D
        if f.D != D:
            wrong += 1
            wrong_flagged += not f.identifiable
    assert hit_id / n_id >= 0.98, (hit_id, n_id)
    assert hit / n >= 0.90, (hit, n)
    assert wrong == 0 or wrong_flagged / wrong >= 0.85, (wrong_flagged, wrong)
    assert 0.88 <= cover / n <= 0.995, (cover, n)
    return (f"identified {hit_id}/{n_id} right, overall {hit}/{n}, "
            f"{wrong_flagged}/{wrong} misses flagged, CI coverage {cover / n:.1%}")


@test
def T10_verdicts_never_on_the_wrong_side():
    """Synthetic loops at 0.6x, 1.0x and 1.5x the bound: no STABLE verdict for an
    unstable loop or UNSTABLE for a stable one, and >= 90% get the right verdict."""
    rng = np.random.default_rng(7)
    right = wrong_side = n = 0
    want = {0.6: "STABLE", 1.0: "BOUNDARY", 1.5: "UNSTABLE"}
    for _ in range(80):
        for m, exp in want.items():
            D = int(rng.integers(0, 4))
            e, u = _synthetic(L.kappa(D) * m, D, 40, rng, noise=0.01, outlier_rate=0.03)
            v = L.judge(L.fit_loop(e, u, d_max=4, n_boot=300, history="initial",
                                   seed=int(rng.integers(1e9))))
            n += 1
            right += v.verdict == exp
            wrong_side += (m < 1 and v.verdict == "UNSTABLE") or (m > 1 and v.verdict == "STABLE")
    assert wrong_side == 0, wrong_side
    assert right / n >= 0.90, (right, n)
    return f"{right}/{n} exact, 0 on the wrong side"


@test
def T11_insufficient_settled_and_short_traces():
    """(a) an all-zero trace is INSUFFICIENT_DATA; (b) a loop that settles in one
    step is fitted as D = 0, K = 1, but judged INCONCLUSIVE (one informative step
    cannot rule out a longer delay), never STABLE; (c) short step responses with
    delay 3 are identified only if the pre-trace history is used correctly."""
    v = L.judge(L.fit_loop(np.zeros(30), np.zeros(30)))
    assert v.verdict == "INSUFFICIENT_DATA", v.verdict
    e = np.array([40.0] + [0.0] * 14)
    u = np.array([-40.0] + [0.0] * 14)
    f = L.fit_loop(e, u, d_max=3, history="initial")
    assert f.D == 0 and abs(f.K - 1) < 1e-9, f
    v = L.judge(f)
    assert v.verdict == "INCONCLUSIVE", v.verdict
    # short step responses (4-5 steps) with delay 3, rested at e0 = 40 before the start
    for K, T in ((1.0, 4), (1.0, 5), (0.5, 5), (1.5, 4)):
        e, u = [40.0], []
        for t in range(T):
            u.append(-K * (e[t - 3] if t >= 3 else 40.0))
            e.append(e[t] + u[t])
        f = L.fit_loop(np.array(e[:T]), np.array(u), d_max=3, history="initial")
        assert f.D == 3 and abs(f.K - K) < 1e-9, (K, T, f.D, f.K)
    return "ok"


@test
def T12_cli_end_to_end():
    """The CLI runs on a CSV trace and on the real latency data, with correct exit codes."""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "trace.csv")
        rng = np.random.default_rng(8)
        e, u = _synthetic(L.kappa(1) * 1.5, 1, 30, rng, outlier_rate=0.0)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "x", "adj"])
            for t in range(len(u)):
                w.writerow([t, 100 + e[t], u[t]])
        run = lambda *a: subprocess.run([sys.executable, os.path.join(here, "loopstab.py"), *a],
                                        capture_output=True, text=True)
        r = run("fit", path, "--error-col", "x", "--adjust-col", "adj", "--target", "100",
                "--history", "initial", "--d-max", "3")
        assert r.returncode == 2 and "UNSTABLE" in r.stdout, r.stdout + r.stderr
        r = run("margin", "--delay", "2", "--agents", "3", "--gain", "0.1")
        assert r.returncode == 0 and "STABLE" in r.stdout, r.stdout
        r = run("latency", os.path.join(here, "latency_traces.csv"), "--period", "1",
                "--agents", "1", "--where", "set=long", "--where", "N=1", "--json")
        assert r.returncode == 0 and '"k_mean_square_regimes"' in r.stdout, r.stdout + r.stderr
    return "fit, margin, latency"


def main() -> int:
    sel = sys.argv[1:] or list(TESTS)
    ok_all = True
    for name in sel:
        t0 = time.time()
        try:
            info = TESTS[name]()
            ok = True
        except AssertionError as e:
            ok, info = False, f"assertion failed: {e}"
        except Exception as e:
            ok, info = False, f"{type(e).__name__}: {e}"
        ok_all &= ok
        print(f"{name:4s} {'PASS' if ok else 'FAIL'}  {TESTS[name].__doc__.strip().splitlines()[0]}"
              f"  [{info}; {time.time() - t0:.1f}s]", flush=True)
    print(f"\nall tests pass: {ok_all}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
