# pump.fun top-5 forecast

Entry for [pumpfun-top5-forecast-01](https://make.fun/bounty/pumpfun-top5-forecast-01)
on make.fun: predict the top 5 new Solana coins by market cap, 24 hours out.

## How it works

`main.py` (stdlib only, offline, deterministic) reads
`/input/history.ndjson` + `task.json` and writes `/output/forecast.json`.

It is a Monte Carlo over log10 market caps (in SOL), simulated in 30-minute
steps from the last snapshot to each of the 7 answer reads:

- **Core diffusion** — small log-cap diffusion estimated from the history's
  own lagged (3-mark) returns, plus a dump/rocket mixture (dump/rocket
  rates and sizes estimated the same way).
- **Shrunk, fast-decaying velocity** — each coin's recent log-velocity
  (recency-weighted) is shrunk x0.25, capped, and decays x0.8 per step:
  pumps exhaust.
- **Dump risk by coin type** — young non-graduated coins dump far more
  often than graduated (complete/pool) ones. Measured from the history:
  every coin that ever reached the top 5 debuted there within minutes of
  launch, and the non-graduated ones dumped to the ~28 SOL floor within
  10–20 minutes and never came back.
- **Newcomer "ghosts"** — the main source of top-5 churn. Debutants arrive
  at the measured debut rate, enter at the known leader's current cap plus
  a sampled debut offset (debutants debut at the top of the chart; the
  reference is known coins only, so ghosts never feed on each other), and
  are stayers (30%, low dump rate) or flashers (70%, dump fast). Ghost
  mass stays unnamed in the forecast — the scorer shares it among the
  answer chart's coins the forecast does not name — which is exactly what
  covers real newcomers at scoring time. Above the cut leader + 0.6, caps
  are pulled back toward it (pumps saturate), so the size distribution
  keeps a realistic upper tail.

Per-place frequencies over 2000 paths x 7 reads are the forecast
probabilities (additively smoothed, shrunk 0.95); per-place simulated caps
give the size-bin probabilities. `top5` is ordered by expected place.

## Backtest (practice history, rolling windows)

| window | this model | persistence baseline |
|--------|-----------|---------------------|
| k=2    | -47.87    | -47.78              |
| k=3    | -21.51    | -29.65              |
| k=4    | -23.62    | -29.87              |
| mean   | **-31.00**| -35.76              |

Mean log score +4.76 over the house persistence baseline on the three
rolling windows (1 read each). Even spread scores -45.06.

## Run

```sh
MAKEFUN_INPUT_DIR=<dir with history.ndjson + task.json> \
MAKEFUN_OUTPUT_DIR=<dir> python3 main.py
```

Deterministic (fixed seed): two runs are byte-identical. ~10 s on a
66-mark history, well inside the 300 s / 1 CPU / 1024 MB limits.
