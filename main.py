#!/usr/bin/env python3
"""pump.fun top-5 forecast (make.fun pumpfun-top5-forecast-01).

Reads /input/history.ndjson + task.json, writes /output/forecast.json.

Method: Monte Carlo over log10 market caps (in SOL). Each coin's log-cap is
simulated step by step (30 min steps) to each answer read as a
persistence-centred process: a small core diffusion (estimated from the
history's own lagged returns), a shrunk recent velocity that decays fast
(pumps exhaust), and a dump/rocket mixture (estimated dump/rocket rates and
sizes). Dump risk depends on the coin: young non-graduated coins dump far
more often than graduated (complete/pool) ones, which matches the history —
every coin that ever reached the top 5 debuted there within minutes of
launch, and the non-graduated ones dumped to the ~28 SOL floor within
10-20 minutes and never came back.

Newcomers are the main source of top-5 churn, so they are simulated as
"ghosts": debutants that arrive at the measured debut rate, enter at the
known leader's current cap plus a sampled debut offset (debutants debut at
the top of the chart; the reference is known coins only, so ghosts never
feed on each other), and are either stayers (30%, low dump rate) or
flashers (70%, dump fast). Above the cut leader + 0.6, caps are pulled
back toward it (pumps saturate) — a gradient, not a wall, so the size
distribution keeps a realistic upper tail. Ghost mass stays unnamed in the
forecast — the scorer shares it among the answer chart's coins the forecast
does not name — which is exactly what covers real newcomers at scoring
time. The per-place frequencies over all paths and reads are the forecast
probabilities (additively smoothed, shrunk); the per-place simulated caps
give the size-bin probabilities. Deterministic: fixed seed.

Stdlib only. No network. Reads MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR when set.
"""
import json
import math
import os
import random
from datetime import datetime, timedelta

SEED = 20261112
N_PATHS = 2000
# Additive smoothing (pseudo-counts) on per-place frequencies.
ALPHA = 2.0
# Shrink frequencies toward the unnamed share (1.0 = honest frequencies).
SHRINK = 0.95
# Name at most this many coins per place.
MAX_NAMED = 60
# Ghost debutants: fraction that are stayers (low dump rate).
STAYER_FRAC = 0.30
# Per-step dump probability for ghost flashers / stayers.
FLASH_DUMP = 0.5
STAYER_DUMP = 0.08
# Velocity decay per step (momentum exhausts).
VEL_DECAY = 0.8
# Cap on shrunk recent velocity (log10/hour).
VEL_CAP = 0.15


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_marks(input_dir):
    """Load ok marks with rows, sorted by time."""
    marks = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if obj.get("ok") and isinstance(obj.get("rows"), list) and obj["rows"]:
                marks.append(obj)
    marks.sort(key=lambda m: parse_time(m["at"]))
    return marks


def _mean_sd(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0, 0.1
    mean = sum(vals) / n
    sd = max(0.05, (sum((v - mean) ** 2 for v in vals) / n) ** 0.5)
    return mean, sd


def logcap(row):
    """log10 of a row's market cap in SOL."""
    return math.log10(int(row["market_cap_lamports"]) / 1e9)


def estimate(marks):
    """Estimate the dynamics from the history itself.

    Lagged returns (3 marks apart, i.e. 30 min at the 10-min cadence) give
    the core diffusion and the dump/rocket mixture at one step. Debutants
    (coins first reaching the top 5) give the newcomer rate and entry caps.
    """
    series = {}
    created = {}
    for m in marks:
        t = parse_time(m["at"])
        for r in m["rows"]:
            mint = r["mint"]
            series.setdefault(mint, []).append((t, logcap(r)))
            if r.get("created_at") and mint not in created:
                created[mint] = parse_time(r["created_at"])
    lag = 3
    if len(marks) <= lag + 1:
        lag = 1
    lag_rets = []
    lag_dts = []
    for mint, s in series.items():
        for i in range(len(s) - lag):
            (t1, x1), (t2, x2) = s[i], s[i + lag]
            dt = (t2 - t1).total_seconds() / 3600.0
            if dt <= 0:
                continue
            lag_rets.append(x2 - x1)
            lag_dts.append(dt)
    lag_dts.sort()
    step_h = lag_dts[len(lag_dts) // 2] if lag_dts else 0.5
    core = [r for r in lag_rets if abs(r) < 0.3]
    core_mean, core_sd = _mean_sd(core)
    dumps = [r for r in lag_rets if r < -0.3]
    rockets = [r for r in lag_rets if r > 0.3]
    dump_mean, dump_sd = _mean_sd(dumps) if dumps else (-0.7, 0.3)
    n = len(lag_rets)
    if rockets:
        rock_mean, rock_sd = _mean_sd(rockets)
        p_rocket = len(rockets) / max(1, n)
    else:
        # Fall back to consecutive-pair big gains, scaled to one step.
        r1 = []
        for mint, s in series.items():
            for (t1, x1), (t2, x2) in zip(s, s[1:]):
                dt = (t2 - t1).total_seconds() / 3600.0
                if 0 < dt < 2.0:
                    r1.append((x2 - x1) / dt * step_h)
        rk = [r for r in r1 if r > 0.3]
        if rk:
            rock_mean, rock_sd = _mean_sd(rk)
            p_rocket = len(rk) / max(1, len(r1))
        else:
            rock_mean, rock_sd, p_rocket = 0.5, 0.3, 0.02
    # Debutants: first time a coin reaches the top 5, and the cap it debuted
    # at — as an OFFSET against the previous mark's leader. A debutant
    # debuts at the top of the chart, so its entry size is measured relative
    # to the leader of the moment, not as an absolute cap.
    rank_series = {}
    for m in marks:
        t = parse_time(m["at"])
        for r in m["rows"]:
            rank_series.setdefault(r["mint"], []).append((t, r["rank"], logcap(r)))
    top1_caps = []
    for m in marks:
        rows = sorted(m["rows"], key=lambda r: r["rank"])
        top1_caps.append(logcap(rows[0]) if rows else 3.0)
    debut_offsets = []
    for mint, s in rank_series.items():
        for (t, rk, x) in s:
            if rk <= 5:
                idx = next(i for i, m in enumerate(marks)
                           if parse_time(m["at"]) == t)
                if idx > 0:
                    leader = top1_caps[idx - 1]
                else:
                    rows = sorted(marks[0]["rows"], key=lambda r: r["rank"])
                    leader = logcap(rows[1]) if len(rows) > 1 else x
                debut_offsets.append(x - leader)
                break
    # Trim the offset tail: the +0.91 outlier is an artefact of the chaotic
    # opening (the "previous leader" was a coin that had just dumped). A
    # debutant may debut at most 0.3 above the leader.
    debut_offsets = [max(-1.0, min(0.3, o)) for o in debut_offsets]
    t0 = parse_time(marks[0]["at"])
    t1 = parse_time(marks[-1]["at"])
    span_h = max((t1 - t0).total_seconds() / 3600.0, 0.5)
    lam_deb = len(debut_offsets) / span_h
    return {
        "step_h": step_h,
        "core_mean": core_mean, "core_sd": core_sd,
        "p_dump": len(dumps) / n if n else 0.05,
        "dump_mean": dump_mean, "dump_sd": dump_sd,
        "p_rocket": p_rocket, "rock_mean": rock_mean, "rock_sd": rock_sd,
        "lam_deb": lam_deb,
        "debut_offsets": sorted(debut_offsets) or [0.0],
    }


def recent_stats(marks, mint, now):
    """EWMA log-velocity (log10/hour) and stability for one coin.

    Velocity is recency-weighted; stability is the recent mean |log-return|
    per sqrt-hour relative to 1.0 (low = sticky coin).
    """
    s = []
    for m in marks:
        for r in m["rows"]:
            if r["mint"] == mint:
                s.append((parse_time(m["at"]), logcap(r)))
    if len(s) < 2:
        return 0.0, 1.0
    num = den = 0.0
    asum = aden = 0.0
    for (t1, x1), (t2, x2) in zip(s, s[1:]):
        dt = (t2 - t1).total_seconds() / 3600.0
        if dt <= 0:
            continue
        age = (now - t2).total_seconds() / 3600.0
        w = math.exp(-age / 3.0)
        num += w * (x2 - x1) / dt
        den += w
        asum += w * abs(x2 - x1) / math.sqrt(dt)
        aden += w
    v = num / den if den else 0.0
    stab = asum / aden if aden else 1.0
    return v, stab


def size_bin(x):
    """Size bin of a log10 cap in SOL: 0 below 2.0, 81 at/above 6.0."""
    if x < 2.0:
        return 0
    if x >= 6.0:
        return 81
    return int((x - 2.0) / 0.05) + 1


def simulate(marks, read_times, params):
    """Monte Carlo over the window. Returns freq, ghost, sizec, total.

    freq[place][mint] = fraction of samples where the mint holds that place;
    ghost[place] = fraction held by a newcomer (unnamed); sizec[place][bin]
    = counts of the place-k cap falling in each size bin.
    """
    rng = random.Random(SEED)
    now = parse_time(marks[-1]["at"])
    last = sorted(marks[-1]["rows"], key=lambda r: r["rank"])
    created = {}
    for m in marks:
        for r in m["rows"]:
            if r.get("created_at") and r["mint"] not in created:
                created[r["mint"]] = parse_time(r["created_at"])
    step_h = params["step_h"]
    core_sd = params["core_sd"]
    core_mean = params["core_mean"]
    coins = []
    for r in last:
        mint = r["mint"]
        v, stab = recent_stats(marks, mint, now)
        age = (now - created.get(mint, now)).total_seconds() / 3600.0
        # Velocity shrunk hard and capped; stability scales diffusion.
        v_eff = max(-VEL_CAP, min(VEL_CAP, v * 0.25))
        s_mult = max(0.5, min(1.6, 0.5 + 0.5 * stab / max(core_sd, 0.05)))
        # Dump risk: young non-graduated coins dump far more often.
        if not r.get("complete") and age < 1.0:
            pd = min(0.6, params["p_dump"] * 4)
        elif not r.get("complete"):
            pd = min(0.4, params["p_dump"] * 2)
        else:
            pd = params["p_dump"]
        coins.append({"mint": mint, "x": logcap(r), "v": v_eff,
                      "sm": s_mult, "age": age, "pd": pd})
    end = max(read_times)
    total_h = (end - now).total_seconds() / 3600.0
    steps = max(1, int(math.ceil(total_h / step_h)))
    sample_steps = sorted({min(steps, max(1, int(round(
        (rt - now).total_seconds() / 3600.0 / step_h)))) for rt in read_times})
    freq = {k: {} for k in range(1, 6)}
    ghost = {k: 0 for k in range(1, 6)}
    sizec = {k: {} for k in range(1, 6)}
    total = 0
    lam_deb = params["lam_deb"] * step_h
    doffs = params["debut_offsets"]
    leader0 = max(c["x"] for c in coins) if coins else 3.0
    # Soft ceiling: above leader0 + 0.6, caps are pulled back toward it
    # (pumps saturate). A gradient, not a wall — a hard cap piles every
    # path onto one bin.
    soft_x = leader0 + 0.6
    for _ in range(N_PATHS):
        # state: [logcap, velocity, sigma_mult, age, mint, dump_prob]
        state = [[c["x"], c["v"], c["sm"], c["age"], c["mint"], c["pd"]]
                 for c in coins]
        ghosts = []  # [logcap, is_stayer]
        for si in range(1, steps + 1):
            # Debutants arrive as a Poisson process at the measured rate.
            lam = lam_deb
            thresh = math.exp(-lam)
            p = 1.0
            n_new = 0
            while True:
                p *= rng.random()
                if p <= thresh:
                    break
                n_new += 1
            for _ in range(n_new):
                # Debut at the KNOWN leader's current cap plus a sampled
                # debut offset. The reference is known coins only, so
                # ghosts never feed on each other (no ratchet).
                ref = max(s[0] for s in state)
                x = ref + doffs[int(rng.random() * len(doffs))]
                x = min(max(x, 1.5), soft_x)
                ghosts.append([x, rng.random() < STAYER_FRAC])
            for s in state:
                s[3] += step_h
                s[1] *= VEL_DECAY  # momentum exhausts
                u = rng.random()
                if u < s[5]:
                    dx = rng.gauss(params["dump_mean"], params["dump_sd"])
                elif u < s[5] + params["p_rocket"]:
                    dx = rng.gauss(params["rock_mean"], params["rock_sd"])
                else:
                    dx = (core_mean + s[1] * step_h) \
                        + core_sd * s[2] * rng.gauss(0, 1)
                s[0] += dx
                if s[0] > soft_x:
                    s[0] -= (s[0] - soft_x) * 0.15
            alive = []
            for g in ghosts:
                if rng.random() < (STAYER_DUMP if g[1] else FLASH_DUMP):
                    continue  # dumped to the floor, gone for good
                g[0] += core_mean + core_sd * 1.2 * rng.gauss(0, 1)
                if g[0] > soft_x:
                    g[0] -= (g[0] - soft_x) * 0.15
                alive.append(g)
            ghosts = alive
            if si in sample_steps:
                cand = [(s[0], 0, s[4]) for s in state]
                cand += [(g[0], 1, -1) for g in ghosts]
                cand.sort(key=lambda t: -t[0])
                total += 1
                for k in range(1, 6):
                    if k - 1 >= len(cand):
                        break
                    x, is_g, mint = cand[k - 1]
                    if is_g:
                        ghost[k] += 1
                    else:
                        freq[k][mint] = freq[k].get(mint, 0) + 1
                    b = size_bin(x)
                    sizec[k][b] = sizec[k].get(b, 0) + 1
    for k in range(1, 6):
        for mint in freq[k]:
            freq[k][mint] /= total
        ghost[k] /= total
    return freq, ghost, sizec, total


def build_forecast(freq, ghost, sizec, total, places):
    """Turn Monte Carlo frequencies into the forecast object."""
    probabilities = []
    for k in range(1, places + 1):
        fk = freq.get(k, {})
        ranked = sorted(fk.items(), key=lambda kv: -kv[1])[:MAX_NAMED]
        n_named = len(ranked)
        rows = []
        s = 0.0
        for mint, f in ranked:
            p = (f * total + ALPHA) / (total + ALPHA * max(n_named, 1)) * SHRINK
            if p < 0.001:
                continue
            rows.append((mint, p))
            s += p
        # Leave the ghost share plus the unnamed tail unnamed: that share
        # is what covers real newcomers and unlisted coins at scoring time.
        tail = 1.0 - ghost.get(k, 0.0) - sum(fk.values())
        cap = max(0.05, 1.0 - ghost.get(k, 0.0) - max(0.0, tail))
        if s > cap and s > 0:
            rows = [(mint, p * cap / s) for mint, p in rows]
            s = cap
        for mint, p in rows:
            probabilities.append({"mint": mint, "place": k, "p": round(p, 6)})
    sizes = []
    for k in range(1, places + 1):
        sc = sizec.get(k, {})
        tot = sum(sc.values())
        if tot == 0:
            continue
        for b, c in sorted(sc.items()):
            p = (c + 0.5) / (tot + 0.5 * 82)
            u = math.floor(p * 1e6)
            if u > 0:
                sizes.append({"place": k, "bin": b, "p": u / 1e6})
    # top5: order by expected place (6 if never in the top 5).
    exp_rank = {}
    for k in range(1, places + 1):
        for mint, f in freq.get(k, {}).items():
            exp_rank[mint] = exp_rank.get(mint, 0.0) + k * f
    for mint in exp_rank:
        tot = sum(freq.get(k, {}).get(mint, 0.0) for k in range(1, places + 1))
        exp_rank[mint] = exp_rank[mint] + 6.0 * (1.0 - tot)
    order = sorted(exp_rank.items(), key=lambda kv: (kv[1], kv[0]))
    top5 = [mint for mint, _ in order[:places]]
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json")) as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    marks = load_marks(input_dir)
    if not marks:
        raise SystemExit("no history marks")
    reads = task.get("reads") or []
    if reads:
        read_times = [parse_time(t) for t in reads]
    else:
        # Fall back to seven hourly reads starting 7 h after the cut.
        cut = parse_time(marks[-1]["at"])
        read_times = [cut + timedelta(hours=h) for h in range(7, 14)]
    params = estimate(marks)
    freq, ghost, sizec, total = simulate(marks, read_times, params)
    fc = build_forecast(freq, ghost, sizec, total, places)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, task.get("output", "forecast.json")), "w") as f:
        f.write(json.dumps(fc) + "\n")


if __name__ == "__main__":
    main()
