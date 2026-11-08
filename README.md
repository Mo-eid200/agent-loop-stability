# loopstab: will your agent loop converge, oscillate, or blow up?

When several agents (LLM agents, tool-calling workers, controllers, async jobs) read a shared state, compute a correction and write it back, the loop has **latency**. Latency turns a well-behaved correction into oscillation and then divergence, and adding agents makes it worse. The problem doesn't show up in unit tests or in averages.

`loopstab` answers three questions from data you already have:

| Command | Input | Answer |
|---|---|---|
| `fit` | a recorded loop (state per step, corrections) | the loop's effective gain and delay, with confidence intervals; **STABLE / BOUNDARY / UNSTABLE**; and what to change |
| `latency` | measured request latencies | the largest safe correction per agent, under four stability criteria |
| `margin` | a delay and a team size | the exact stability bound |

```
pip install numpy
python3 loopstab.py fit example_trace.csv --step-col round --error-col x --adjust-col adj --target 100 --history initial --d-max 3
```

```
loopstab fit: example_trace.csv  (15 steps, N = 2)

  VERDICT: UNSTABLE
   - fitted aggregate gain K = 2.000 (95% CI 2.000 to 2.000)
   - delay D = 1 step(s); stability requires K < kappa(D) = 1.000
   - bootstrap support for D: {1: 1.0}   R^2 = 1.000   points = 15
  Recommendations:
   * reduce each agent's correction to <= 0.400 x the visible error (80% of the bound with N = 2); currently 1.000
   * even a single agent at this per-agent gain is at or beyond the bound
   * cutting the delay alone cannot fix this: the aggregate gain is too high even with zero delay
```

`example_trace.csv` holds the first 15 rounds of a real run: two gpt-4o-mini agents, each told that another agent shares the loop and that its reading is one round old. Both still corrected the full error.

## Validated on real LLM agents

On **54 recorded runs of gpt-4o-mini agents** (1,806 model calls, 1–3 agents, delay 0–2 steps), `loopstab` saw only the first 15 rounds of each run and was **not told the delay**:

- **Delay recovered in 54 of 54 runs.**
- **48 confident verdicts, 0 wrong:**
  - every run called UNSTABLE diverged (36/36);
  - every run called BOUNDARY settled into a sustained oscillation (12/12).

  This holds for every one of 30 alternative definitions of "decays" and "diverges".
- **Early warning:** the first confident verdict came after **4–8 rounds**, and all 48 of those earliest verdicts were already correct.
- **What it learned about the agents:** each agent corrects the **whole** visible error (median gain 1.00 per agent), even when the prompt states how many agents share the loop and how stale the reading is.
- **The other 6 runs** settled in a single step. That gives too little information to rule out a longer delay, so `loopstab` reports them as INCONCLUSIVE instead of guessing.

On **4,050 measured request latencies** of a production LLM backend, the recommended bound is confirmed by replaying the real delay sequences through the loop. The common worst-case-delay rule turns out **1.7–5.3× more conservative** than necessary, because a single slow request sets its limit.

Full tables: [`RESULTS.md`](RESULTS.md).

## How it works

**Model.** With N agents, per-agent gain k and readings that are d steps old:

```
e[t+1] = e[t] - k * sum_i e[t - d_i(t)]          aggregate gain K = N k
```

**Constant delay D.** The loop is stable if and only if `K < kappa(D) = 2 sin(pi / (2(2D+1)))`. That gives 2, 1, 0.618, 0.445, … for D = 0, 1, 2, 3. For agents with different delays, `sum k_i < kappa(max d_i)` is sufficient, and it is tight when all the delays are equal.

**Fitting.** `fit` estimates D and K with a **scale-invariant, robust estimator**:
- For each candidate delay, a weighted median of correction / lagged error.
- The delay is chosen by the mean bounded residual.
- A single wild reply (real LLMs occasionally answer 31860 instead of 3180, or flip a sign) cannot dominate it, and neither can the geometric growth of an unstable loop.
- A bootstrap re-selects the delay in every resample, so the confidence interval on K also carries the uncertainty about D.

**Verdicts.**

| Verdict | Meaning |
|---|---|
| STABLE | the whole 95% CI lies below the bound (with a 5% band), for every delay the data support |
| UNSTABLE | the whole CI lies above the bound |
| BOUNDARY | the estimate sits on the bound: expect a sustained oscillation |
| INCONCLUSIVE | the estimate is clearly on one side, but the trace is too short or too quiet to be sure |
| INSUFFICIENT_DATA | nothing happened in the trace |

**Random latency.** `latency` turns latencies into delays with `floor(latency / period)`, then computes the largest safe per-agent gain four ways:

| Criterion | What it guarantees |
|---|---|
| worst-case delay | safe, usually very conservative |
| mean | the average converges: **not sufficient** on its own |
| mean-square, independent delays | the variance converges too (exact closed-form operator) |
| **mean-square with fast/slow regimes** | same, when slow periods persist (Markov jump linear system). **The recommended design basis.** |
| almost-sure (Monte Carlo Lyapunov exponent) | every run converges *eventually*, possibly after huge excursions. Shown so you know **not** to design to it. |

## Input formats

- **Loop trace (`fit`):** a CSV with one row per step, or one row per agent per step; corrections are summed per step.
  - `--error-col`: the state (use `--target` to subtract a setpoint).
  - `--adjust-col`: the correction. If omitted, the corrections are taken from the state differences.
  - `--step-col`: the step number, when there is one row per agent.
  - `--history initial`: use this if the system rested at its first value before the trace started.
- **Latencies (`latency`):** a CSV with columns `agent, t_request, t_response` (seconds) and optionally `ok`. Filter rows with `--where col=value`.

## Verification

```
python3 run_all.py      # about 5 minutes; writes RESULTS.md
```

| Layer | File | What it checks |
|---|---|---|
| 1 | `tests.py` | 12 tests (listed below) |
| 2 | `validate_real.py` | the real-data results above, including sensitivity to the outcome thresholds |
| 3 | `mutate.py` | 11 realistic bugs planted one at a time (an off-by-one in the bound, a transposed regime matrix, a non-robust fit, a verdict that ignores the confidence interval, and others). **Every one is caught.** |

The 12 tests in layer 1 check:
- the bound against eigenvalues for D = 0–40 and 3,000 random cases;
- the heterogeneous-delay rule on 2,000 random teams;
- the closed-form mean-square operator against brute-force enumeration and against 400,000 simulated paths;
- the regime operator against simulation;
- recovery of the delay and gain on 300 synthetic loops with 5% wild corrections:
  - when it says the delay is identified, it is right 241 times out of 242;
  - 95% confidence intervals cover the truth 95.7% of the time;
- that no verdict ever lands on the wrong side of the bound;
- the CLI end to end.

`verify.yml` is a GitHub Actions workflow that runs all of this on every push.

## Limitations

- **The model is linear, with a constant gain per agent.** Real agents can saturate, refuse, or change strategy as errors grow. The robust fit tolerates occasional wild replies, but not a systematic change of behaviour during the trace.
- **A delay can only be identified if the error actually moves.** In slow, heavily damped loops several delays fit about equally well. `loopstab` then says so (bootstrap support, `identifiable: false`) and takes the least favourable bound.
- **Latency analysis assumes the latency profile is stationary,** and models persistence with two regimes. The almost-sure bound is a Monte Carlo estimate.
- **The real-data validation covers one model (gpt-4o-mini) and one task type.** It shows the method works on real LLM agents. It does not show that every agent behaves like these.

## Background

The theory and the LLM experiments come from:

- M. Eid, *Delay-Induced Instability in Multi-Agent LLM Orchestration: Exact Stability Boundaries and a Delay-Aware Extension of UCCF*, preprint, doi:[10.5281/zenodo.23697430](https://doi.org/10.5281/zenodo.23697430).
- M. Eid, *Artificial Intelligence Needs Control Theory: Towards Stable, Adaptive, and Autonomous AI Systems*, preprint, doi:[10.5281/zenodo.23378509](https://doi.org/10.5281/zenodo.23378509).
- S. A. Levin and R. M. May, *A note on difference-delay equations*, Theoretical Population Biology 9 (1976).
- O. L. V. Costa, M. D. Fragoso and R. P. Marques, *Discrete-Time Markov Jump Linear Systems*, Springer (2005).

**Data.** `llm_loop.jsonl` holds the recorded gpt-4o-mini runs, and `latency_traces.csv` holds the request latencies from a production LLM orchestration backend. Both were collected by the author.

License: MIT. Author: Mohamed Eid (ORCID [0009-0008-8404-978X](https://orcid.org/0009-0008-8404-978X)).
