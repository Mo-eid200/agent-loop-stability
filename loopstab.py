#!/usr/bin/env python3
"""loopstab: stability analysis for multi-agent feedback loops with delay.

Model
    N agents read a shared error e and each applies a correction. A reading
    used at step t is d steps old (d = 0: the correction lands on the next
    step), so
        e[t+1] = e[t] - k * sum_i e[t - d_i(t)],        K = N k (aggregate gain).
    This is the loop formed by LLM agents, tool calls, controllers or async
    workers acting on shared state with latency.

What it answers
    margin   exact stability boundary for a fixed delay: K < kappa(D), with
             kappa(D) = 2 sin(pi / (2 (2D + 1)))  (Levin and May, 1976).
    fit      estimate K and D from a recorded loop trace, with bootstrap
             confidence intervals, and give a verdict with recommendations.
    latency  from measured request latencies and a loop period, the largest
             safe per-agent gain under four criteria: worst-case delay,
             mean, mean-square (independent delays), and mean-square with
             persistent fast/slow regimes (Markov jump), plus a Monte Carlo
             almost-sure estimate.

Only NumPy is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__version__ = "1.0.0"
D_CAP = 20                      # largest delay (in steps) handled by the operators


# =============================================================================
# 1. Exact deterministic results
# =============================================================================
def kappa(D: int) -> float:
    """Exact stability bound on the aggregate gain for a constant delay D >= 0."""
    if D < 0:
        raise ValueError("delay must be >= 0")
    return 2.0 * math.sin(math.pi / (2.0 * (2 * D + 1)))


def shift_matrix(n: int) -> np.ndarray:
    """S: e[t+1] = e[t] with the history shifted down (no correction)."""
    S = np.zeros((n, n))
    S[0, 0] = 1.0
    for i in range(1, n):
        S[i, i - 1] = 1.0
    return S


def companion(gains_by_delay: dict[int, float], n: Optional[int] = None) -> np.ndarray:
    """State matrix for e[t+1] = e[t] - sum_d g_d e[t-d]; state (e[t], ..., e[t-n+1])."""
    dmax = max(gains_by_delay) if gains_by_delay else 0
    n = n or dmax + 1
    A = shift_matrix(n)
    for d, g in gains_by_delay.items():
        A[0, d] -= g
    return A


def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0


def stable_fixed(K: float, D: int) -> bool:
    """Analytic test, exact: stable iff 0 < K < kappa(D)."""
    return 0.0 < K < kappa(D)


def stable_heterogeneous(gains: Sequence[float], delays: Sequence[int]) -> bool:
    """Sufficient test for agents with different constant delays:
    sum(k_i) < kappa(max d_i) guarantees stability, and is tight when all delays
    are equal (worst-case delay rule)."""
    return 0.0 < sum(gains) < kappa(max(delays))


def max_delay_for_gain(K: float) -> int:
    """Largest constant delay D with K < kappa(D); -1 if none."""
    if K <= 0:
        return D_CAP
    D = -1
    while D + 1 <= 10_000 and K < kappa(D + 1):
        D += 1
    return D


# =============================================================================
# 2. Random delays: mean and mean-square operators (exact, closed form)
# =============================================================================
def _E(d: int, n: int) -> np.ndarray:
    E = np.zeros((n, n))
    E[0, d] = 1.0
    return E


def _moments(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(p)
    Ebar = sum(p[d] * _E(d, n) for d in range(n))
    EE = sum(p[d] * np.kron(_E(d, n), _E(d, n)) for d in range(n))
    return Ebar, EE, n


def mean_operator(k: float, N: int, p: np.ndarray) -> np.ndarray:
    """E[A] for N agents with i.i.d. delays ~ p: governs E[e]."""
    Ebar, _, n = _moments(p)
    return shift_matrix(n) - k * N * Ebar


def ms_operator(k: float, N: int, p: np.ndarray) -> np.ndarray:
    """E[A (x) A] for N agents with independent delays ~ p: governs E[e e^T].

    With A = S - k sum_i E_{d_i} and independent d_i:
      E[A(x)A] = S(x)S - kN (Ebar(x)S + S(x)Ebar)
                 + k^2 ( N(N-1) Ebar(x)Ebar + N E[E_d (x) E_d] ).
    """
    Ebar, EE, n = _moments(p)
    S = shift_matrix(n)
    return (np.kron(S, S) - k * N * (np.kron(Ebar, S) + np.kron(S, Ebar))
            + k * k * (N * (N - 1) * np.kron(Ebar, Ebar) + N * EE))


def mjls_ms_operator(k: float, N: int, P: np.ndarray, p_modes: Sequence[np.ndarray]) -> np.ndarray:
    """Mean-square operator when a shared regime (e.g. fast/slow backend) follows
    a Markov chain with transition matrix P, and delays within regime i are
    i.i.d. ~ p_modes[i] (Costa, Fragoso and Marques, 2005).
    Q_j(t+1) = sum_i P[i, j] M_i Q_i(t);  stable iff spectral radius < 1."""
    m = len(p_modes)
    n = max(len(p) for p in p_modes)
    blocks = [ms_operator(k, N, np.pad(p, (0, n - len(p)))) for p in p_modes]
    n2 = n * n
    big = np.zeros((m * n2, m * n2))
    for j in range(m):
        for i in range(m):
            big[j * n2:(j + 1) * n2, i * n2:(i + 1) * n2] = P[i, j] * blocks[i]
    return big


def lyapunov_exponent(k: float, N: int, sampler: Callable[[np.random.Generator], list],
                      n: int, steps: int = 4000, reps: int = 16, seed: int = 0) -> tuple[float, float]:
    """Monte Carlo top Lyapunov exponent (almost-sure growth rate per step).
    sampler(rng) returns the N delays for one step. Returns (mean, 95% half-width)."""
    return _lyap_vectorised(k, N, sampler, n, steps, reps, seed)


def _lyap_vectorised(k, N, sampler, n, steps, reps, seed):
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(reps):
        hist = rng.standard_normal(n)            # hist[j] = e[t - j]
        hist /= np.linalg.norm(hist)
        acc = 0.0
        for _ in range(steps):
            ds = sampler(rng)
            new = hist[0] - k * float(sum(hist[d] for d in ds))
            hist = np.concatenate(([new], hist[:-1]))
            nrm = np.linalg.norm(hist)
            if nrm == 0.0:
                acc = -np.inf
                break
            acc += math.log(nrm)
            hist /= nrm
        rates.append(acc / steps)
    r = np.array(rates)
    if np.isinf(r).any():
        return -np.inf, 0.0
    return float(r.mean()), float(1.96 * r.std(ddof=1) / math.sqrt(len(r)))


def boundary(is_stable: Callable[[float], bool], k_max: float = 4.0, tol: float = 1e-4) -> float:
    """Largest gain g in (0, k_max] such that every gain in (0, g] is stable.
    Scans for the first unstable gain, then bisects."""
    grid = np.linspace(k_max / 60, k_max, 60)
    lo = 0.0
    for g in grid:
        if is_stable(float(g)):
            lo = float(g)
        else:
            hi = float(g)
            break
    else:
        return k_max
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if is_stable(mid):
            lo = mid
        else:
            hi = mid
    return lo


# =============================================================================
# 3. Fitting a recorded loop
# =============================================================================
@dataclass
class Fit:
    D: Optional[int]
    K: Optional[float]
    K_ci: Optional[tuple]
    r2: Optional[float]
    sse_by_D: dict
    D_support: dict                 # bootstrap frequency of each selected D
    n_points: int
    identifiable: bool
    note: str = ""


def _wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, 0.5 * c[-1])])


def _robust_select(X: np.ndarray, y: np.ndarray, scale: np.ndarray, usable: list):
    """For each candidate delay d: K_d = weighted median of -y/X[:,d] over steps
    where that lag is non-zero; loss = mean over ALL steps of |y + K_d X[:,d]| / scale.
    Each step's scaled residual is bounded by 1 + |K_d|, so one wild correction
    cannot dominate, and steps where nothing happened still count as evidence."""
    best, losses = None, {}
    for d in usable:
        x = X[:, d]
        m = np.abs(x) > 1e-12 * scale
        if m.sum() < 1:
            continue
        K = _wmedian(-y[m] / x[m], np.abs(x[m]) / scale[m])
        loss = float(np.mean(np.abs(y + K * x) / scale))
        losses[d] = loss
        if best is None or loss < best[2] - 1e-12:
            best = (d, K, loss)
    return best, losses


def _ols_select(X: np.ndarray, y: np.ndarray, scale: np.ndarray, usable: list):
    best, losses = None, {}
    for d in usable:
        xx = X[:, d] @ X[:, d]
        if xx <= 0:
            continue
        K = -(X[:, d] @ y) / xx
        loss = float(((y + K * X[:, d]) ** 2).sum())
        losses[d] = loss
        if best is None or loss < best[2] - 1e-12 * max(1.0, best[2]):
            best = (d, K, loss)
    return best, losses


def fit_loop(e: Sequence[float], u: Optional[Sequence[float]] = None, d_max: int = 6,
             n_boot: int = 2000, seed: int = 0, method: str = "robust",
             history: str = "unknown") -> Fit:
    """Estimate (K, D) in u[t] = -K e[t-D] + noise.

    e        error per step (state minus target)
    u        total correction applied at each step; if omitted, u[t] = e[t+1] - e[t]
    method   "robust" (default): scale-invariant weighted-median estimator, robust
             to occasional wild corrections (LLM typos, sign slips) and to the
             geometric growth of an unstable loop. "ols": least squares.
    history  "unknown" (default): steps before the trace are unknown, so the
             first d_max steps are used only as history. "initial": the system
             rested at e[0] before the trace started (step-response experiments).

    D is selected over 0..d_max. The bootstrap resamples steps and re-selects D
    each time, so the confidence interval for K includes the uncertainty in D."""
    e = np.asarray(e, float)
    u = np.diff(e) if u is None else np.asarray(u, float)
    T = min(len(u), len(e))
    if history == "initial":
        e = np.concatenate([np.full(d_max, e[0]), e[:T]])
        u = np.concatenate([np.zeros(d_max), u[:T]])
        t_first = d_max
    else:
        e, u, t_first = e[:T], u[:T], d_max
    T = len(u)
    y = u[t_first:T]
    if len(y) < 3:
        return Fit(None, None, None, None, {}, {}, len(y), False, "trace too short for the requested d_max")
    X = np.stack([e[t_first - d:T - d] for d in range(d_max + 1)], axis=1)   # X[:, d] = e[t-d]
    scale = np.maximum(np.abs(X).max(axis=1), np.abs(y))
    keep = scale > 0
    X, y, scale = X[keep], y[keep], scale[keep]
    usable = [d for d in range(d_max + 1) if (np.abs(X[:, d]) > 0).any()] if len(y) else []
    if len(y) < 3 or not usable:
        return Fit(None, None, None, None, {}, {}, int(len(y)), False,
                   "the error is zero for (almost) the whole trace: the loop settled "
                   "before its dynamics could be observed")
    select = _robust_select if method == "robust" else _ols_select
    best, losses = select(X, y, scale, usable)
    if best is None:
        return Fit(None, None, None, None, {}, {}, int(len(y)), False, "no delay could be fitted")
    D, K, loss = best
    resid = y + K * X[:, D]
    r2 = 1.0 - float((resid ** 2).sum()) / max(float(((y - y.mean()) ** 2).sum()), 1e-300)

    rng = np.random.default_rng(seed)
    Ks, Ds = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        b, _ = select(X[idx], y[idx], scale[idx], usable)
        if b:
            Ds.append(b[0]); Ks.append(b[1])
    Ks = np.array(Ks)
    ci = (float(np.quantile(Ks, 0.025)), float(np.quantile(Ks, 0.975)))
    support = {int(d): round(Ds.count(d) / len(Ds), 3) for d in sorted(set(Ds))}
    ident = len(usable) >= 2 and support.get(int(D), 0) >= 0.9
    note = "" if ident else ("only one candidate delay could be tested" if len(usable) < 2 else
                             f"the delay is uncertain: bootstrap support {support}")
    return Fit(int(D), float(K), ci, float(r2), {int(k): v for k, v in losses.items()}, support,
               int(len(y)), bool(ident), note)


@dataclass
class Verdict:
    verdict: str                    # STABLE / UNSTABLE / BOUNDARY / INCONCLUSIVE / INSUFFICIENT_DATA
    K: Optional[float]
    K_ci: Optional[tuple]
    D: Optional[int]
    kappa: Optional[float]
    margin: Optional[float]         # K / kappa(D)
    reasons: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)


def judge(fit: Fit, N: int = 1, safety: float = 0.8, band: float = 0.05) -> Verdict:
    """Turn a fit into a verdict.

    STABLE    the whole 95% CI of K lies below kappa(D)(1 - band): errors decay
    UNSTABLE  the whole CI lies above kappa(D)(1 + band): errors grow
    BOUNDARY  the estimate itself is within the band around kappa(D): expect a
              sustained oscillation
    INCONCLUSIVE  the CI straddles the bound only because the trace carries too
              little information (the point estimate is clearly on one side)
    using, for every delay with >= 5% bootstrap support, the least favourable kappa."""
    if fit.K is None:
        return Verdict("INSUFFICIENT_DATA", None, None, None, None, None, [fit.note],
                       ["record a longer trace, or start from a non-zero error"])
    Ds = [d for d, f in fit.D_support.items() if f >= 0.05] or [fit.D]
    kap_lo, kap_hi = min(kappa(d) for d in Ds), max(kappa(d) for d in Ds)
    lo, hi = fit.K_ci
    kap = kappa(fit.D)
    margin = fit.K / kap
    reasons = [f"fitted aggregate gain K = {fit.K:.3f} (95% CI {lo:.3f} to {hi:.3f})",
               f"delay D = {fit.D} step(s); stability requires K < kappa(D) = {kap:.3f}"]
    if len(Ds) > 1:
        reasons.append(f"bootstrap supports delays {Ds}: kappa between {kap_lo:.3f} and {kap_hi:.3f}")
    if hi < kap_lo * (1 - band):
        v = "STABLE"
    elif lo > kap_hi * (1 + band):
        v = "UNSTABLE"
    elif abs(margin - 1.0) <= band:
        v = "BOUNDARY"
        reasons.append("the gain sits on the stability bound: expect a sustained oscillation "
                       "that neither decays nor grows")
    else:
        v = "INCONCLUSIVE"
        side = "below" if margin < 1 else "above"
        reasons.append(f"the best estimate is {side} the bound ({margin:.0%} of it), but the "
                       f"trace is too short or too quiet to be sure")
    recs = []
    k_agent = fit.K / N
    safe_k = safety * kap / N
    if v != "STABLE" or margin > safety:
        recs.append(f"reduce each agent's correction to <= {safe_k:.3f} x the visible error "
                    f"({safety:.0%} of the bound with N = {N}); currently {k_agent:.3f}")
        n_max = int(math.floor(kap / k_agent - 1e-12)) if k_agent > 0 else N
        if n_max < 1:
            recs.append("even a single agent at this per-agent gain is at or beyond the bound")
        elif n_max < N:
            recs.append(f"or run at most {n_max} agent(s) at the current per-agent gain")
        dmax = max_delay_for_gain(fit.K)
        if dmax < fit.D:
            recs.append(f"or cut the loop delay to <= {dmax} step(s)" if dmax >= 0 else
                        "cutting the delay alone cannot fix this: the aggregate gain is too "
                        "high even with zero delay")
    else:
        recs.append(f"margin is healthy ({margin:.0%} of the bound); the loop tolerates "
                    f"delays up to {max_delay_for_gain(fit.K)} step(s) at this gain")
    if not fit.identifiable:
        reasons.append(fit.note)
    return Verdict(v, fit.K, fit.K_ci, fit.D, kap, margin, reasons, recs)


# =============================================================================
# 4. Measured latencies -> safe gains
# =============================================================================
def delays_from_latency(lat_s: np.ndarray, period_s: float) -> np.ndarray:
    """A correction whose request takes L seconds lands floor(L / period) steps late."""
    return np.minimum(np.floor(np.asarray(lat_s) / period_s).astype(int), D_CAP)


def two_regimes(lat_by_agent: list[np.ndarray], period_s: float):
    """Split latencies into fast/slow regimes (1-D 2-means on log latency) and
    estimate regime persistence from consecutive requests of each agent."""
    allv = np.concatenate(lat_by_agent)
    x = np.log(np.maximum(allv, 1e-6))
    c = np.quantile(x, [0.25, 0.75])
    for _ in range(100):
        lab = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        newc = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(2)])
        if np.allclose(newc, c):
            break
        c = newc
    thr = math.exp(c.mean())
    P = np.ones((2, 2))                                   # Laplace prior
    for seq in lat_by_agent:
        s = (np.asarray(seq) > thr).astype(int)
        for a, b in zip(s[:-1], s[1:]):
            P[a, b] += 1
    P /= P.sum(axis=1, keepdims=True)
    d = delays_from_latency(allv, period_s)
    n = int(d.max()) + 1
    p_modes = []
    for j in range(2):
        sel = d[(allv > thr) == bool(j)]
        p = np.bincount(sel, minlength=n).astype(float) if len(sel) else np.eye(n)[0]
        p_modes.append(p / p.sum())
    return thr, P, p_modes


@dataclass
class LatencyReport:
    period_s: float
    N: int
    n_requests: int
    latency_p50: float
    latency_p99: float
    latency_max: float
    delay_distribution: dict
    regime_threshold_s: float
    regime_persistence: list
    k_worst_case: float
    k_mean: float
    k_mean_square: float
    k_mean_square_regimes: float
    k_almost_sure: float
    conservatism_worst_case: float
    recommendation: str


def analyze_latency(lat_by_agent: list[np.ndarray], period_s: float, N: int,
                    lyap_steps: int = 3000, lyap_reps: int = 12, seed: int = 0) -> LatencyReport:
    allv = np.concatenate(lat_by_agent)
    d = delays_from_latency(allv, period_s)
    n = int(d.max()) + 1
    p = np.bincount(d, minlength=n).astype(float)
    p /= p.sum()
    thr, P, p_modes = two_regimes(lat_by_agent, period_s)
    k_max = 2.5 / N

    k_wc = kappa(int(d.max())) / N
    k_mean = boundary(lambda k: spectral_radius(mean_operator(k, N, p)) < 1, k_max)
    k_ms = boundary(lambda k: spectral_radius(ms_operator(k, N, p)) < 1, k_max)
    k_mj = boundary(lambda k: spectral_radius(mjls_ms_operator(k, N, P, p_modes)) < 1, k_max)

    # almost-sure: Monte Carlo with the same Markov regime model
    def make_sampler():
        state = {"m": 0}
        cum = [np.cumsum(pm) for pm in p_modes]

        def sampler(rng):
            state["m"] = int(rng.random() < P[state["m"], 1])
            c = cum[state["m"]]
            return [int(np.searchsorted(c, rng.random(), side="right")) for _ in range(N)]
        return sampler

    lo, hi = 0.0, k_max                      # bisection (exponent increases with gain here)
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if _lyap_vectorised(mid, N, make_sampler(), n, lyap_steps, lyap_reps, seed)[0] < 0:
            lo = mid
        else:
            hi = mid
    k_as = lo
    rec = (f"keep each agent's correction below {0.8 * k_mj:.3f} x the visible error "
           f"(80% of the mean-square bound with regimes). The worst-case-delay rule "
           f"(<= {k_wc:.3f}) is {k_mj / k_wc:.1f}x more conservative than needed.")
    if k_as > 1.2 * k_mj:
        rec += (f" Do not design to the almost-sure bound ({k_as:.3f}): between it and the "
                f"mean-square bound the loop converges eventually but with rare, very large excursions.")
    else:
        rec += " The mean-square and almost-sure bounds agree closely for this latency profile."
    return LatencyReport(
        period_s, N, int(len(allv)), float(np.quantile(allv, 0.5)), float(np.quantile(allv, 0.99)),
        float(allv.max()), {int(i): round(float(v), 4) for i, v in enumerate(p) if v > 0},
        round(thr, 3), [[round(float(x), 3) for x in row] for row in P],
        round(k_wc, 4), round(k_mean, 4), round(k_ms, 4), round(k_mj, 4), round(k_as, 4),
        round(k_mj / k_wc, 2), rec)


# =============================================================================
# 5. I/O and CLI
# =============================================================================
def read_loop_csv(path: str, error_col: str, adjust_col: Optional[str], target: Optional[float],
                  step_col: Optional[str]) -> tuple[np.ndarray, Optional[np.ndarray], int]:
    """Rows: one per step, or one per agent per step (adjustments are summed per
    step). Returns (error per step, total adjustment per step or None, N)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("empty trace")
    if step_col and step_col in rows[0]:
        steps: dict = {}
        agents_per_step: dict = {}
        for r in rows:
            s = int(float(r[step_col]))
            steps.setdefault(s, {"x": float(r[error_col]), "u": 0.0})
            if adjust_col:
                steps[s]["u"] += float(r[adjust_col])
            agents_per_step[s] = agents_per_step.get(s, 0) + 1
        order = sorted(steps)
        x = np.array([steps[s]["x"] for s in order])
        u = np.array([steps[s]["u"] for s in order]) if adjust_col else None
        N = int(np.median(list(agents_per_step.values())))
    else:
        x = np.array([float(r[error_col]) for r in rows])
        u = np.array([float(r[adjust_col]) for r in rows]) if adjust_col else None
        N = 1
    e = x - target if target is not None else x
    return e, u, N


def read_latency_csv(path: str, where: Optional[dict] = None) -> list[np.ndarray]:
    """Columns: agent, t_request, t_response (seconds); ok (optional, 1 = success).
    where: keep only rows whose columns equal these values, e.g. {"set": "long"}."""
    by: dict = {}
    where = where or {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("ok", "1") not in ("1", "true", "True"):
                continue
            if any(r.get(k) != v for k, v in where.items()):
                continue
            by.setdefault(r.get("agent", "0"), []).append(
                (float(r["t_request"]), float(r["t_response"]) - float(r["t_request"])))
    return [np.array([lat for _, lat in sorted(v)]) for v in by.values()]


def _print_verdict(v: Verdict, fit: Fit) -> None:
    print(f"\n  VERDICT: {v.verdict}")
    for r in v.reasons:
        print(f"   - {r}")
    if fit.D_support:
        print(f"   - bootstrap support for D: {fit.D_support}   R^2 = {fit.r2:.3f}   points = {fit.n_points}")
    print("  Recommendations:")
    for r in v.recommendations:
        print(f"   * {r}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="loopstab", description=__doc__.split("\n")[0])
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("margin", help="exact bound for a fixed delay")
    m.add_argument("--delay", type=int, required=True)
    m.add_argument("--agents", type=int, default=1)
    m.add_argument("--gain", type=float, help="per-agent gain to check")

    f = sub.add_parser("fit", help="fit K and D from a loop trace and judge stability")
    f.add_argument("trace")
    f.add_argument("--error-col", default="x")
    f.add_argument("--adjust-col", default=None)
    f.add_argument("--step-col", default=None)
    f.add_argument("--target", type=float, default=None)
    f.add_argument("--d-max", type=int, default=6)
    f.add_argument("--agents", type=int, default=None, help="override N")
    f.add_argument("--method", choices=["robust", "ols"], default="robust")
    f.add_argument("--history", choices=["unknown", "initial"], default="unknown",
                   help="'initial': the system rested at the first value before the trace")
    f.add_argument("--json", action="store_true")

    la = sub.add_parser("latency", help="safe gains from measured request latencies")
    la.add_argument("trace")
    la.add_argument("--period", type=float, required=True, help="loop period in seconds")
    la.add_argument("--agents", type=int, required=True)
    la.add_argument("--where", action="append", default=[], metavar="COL=VALUE",
                    help="filter rows, e.g. --where set=long --where N=8")
    la.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "margin":
        k = kappa(a.delay)
        print(f"delay D = {a.delay}: aggregate gain must satisfy K < kappa(D) = {k:.4f}")
        print(f"with N = {a.agents} agents: per-agent gain < {k / a.agents:.4f}")
        if a.gain is not None:
            K = a.gain * a.agents
            ok = stable_fixed(K, a.delay)
            print(f"K = {K:.4f}: {'STABLE' if ok else 'UNSTABLE'} (margin {K / k:.0%} of the bound)")
            return 0 if ok else 2
        return 0

    if a.cmd == "fit":
        e, u, N = read_loop_csv(a.trace, a.error_col, a.adjust_col, a.target, a.step_col)
        N = a.agents or N
        fit = fit_loop(e, u, d_max=a.d_max, method=a.method, history=a.history)
        v = judge(fit, N)
        if a.json:
            print(json.dumps({"fit": asdict(fit), "verdict": asdict(v), "N": N}, indent=1, default=str))
        else:
            print(f"loopstab fit: {a.trace}  ({len(e)} steps, N = {N})")
            _print_verdict(v, fit)
        return {"STABLE": 0, "BOUNDARY": 1, "UNSTABLE": 2, "INCONCLUSIVE": 3}.get(v.verdict, 4)

    if a.cmd == "latency":
        lat = read_latency_csv(a.trace, dict(w.split("=", 1) for w in a.where))
        if not lat:
            raise SystemExit("no rows match")
        r = analyze_latency(lat, a.period, a.agents)
        if a.json:
            print(json.dumps(asdict(r), indent=1))
        else:
            print(f"loopstab latency: {a.trace}  ({r.n_requests} requests, period {r.period_s}s, N = {r.N})")
            print(f"  latency p50 {r.latency_p50:.2f}s  p99 {r.latency_p99:.2f}s  max {r.latency_max:.2f}s")
            print(f"  delay distribution (steps): {r.delay_distribution}")
            print(f"  regimes: slow if > {r.regime_threshold_s}s, persistence P = {r.regime_persistence}")
            print("  largest safe per-agent gain:")
            print(f"    worst-case delay rule      {r.k_worst_case:.4f}")
            print(f"    mean (first moment)        {r.k_mean:.4f}   (not sufficient on its own)")
            print(f"    mean-square, independent   {r.k_mean_square:.4f}")
            print(f"    mean-square, with regimes  {r.k_mean_square_regimes:.4f}   <- recommended basis")
            print(f"    almost-sure (Monte Carlo)  {r.k_almost_sure:.4f}   (converges, but with huge excursions)")
            print(f"  {r.recommendation}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
