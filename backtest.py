#!/usr/bin/env python3
"""
NRFI model backtest / calibration harness
------------------------------------------
Answers the only question that makes the EV column trustworthy: when the model
says 65% NRFI, does it actually happen ~65% of the time?

Design:
  * No look-ahead. A game on date D is scored using ONLY data from before D —
    the starter's 1st-inning Statcast history and each offense's 1st-inning
    scoring, both cut off at D (exclusive).
  * Efficient. Each starter's Statcast is pulled once for the whole window and
    sliced by game_date, instead of re-querying per date.
  * Faithful. Reuses the production math (shrink, log5, the same priors) from
    nrfi_scanner, so we're calibrating the real model, not a copy.

Outputs a reliability table, Brier score (vs. always-predict-base-rate baseline),
log-loss, and a Platt recalibration fit (logistic outcome ~ logit(p)). The Platt
slope < 1 means overconfident; the fitted (a, b) are the recalibration you can
paste back into the model. Optionally writes a per-game CSV and a reliability PNG.

Usage:
  python backtest.py --start 2026-07-01 --end 2026-08-07 --min-starts 3
"""

import argparse
import csv
import datetime as dt
import math
import sys
from collections import defaultdict

import pandas as pd
import requests

import nrfi_scanner as n  # reuse SEASON_START, shrink, log5, priors, statcast_pitcher

MLB_API = n.MLB_API


def _d(s):
    return dt.date.fromisoformat(s)


# ----------------------------------------------------------------------------
# Data: one schedule pull gives outcomes, starters, and offense history
# ----------------------------------------------------------------------------
def fetch_full_schedule(start_str, end_str):
    """Every regular-season game start..end with starters + 1st-inning runs."""
    r = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "startDate": start_str, "endDate": end_str,
                "gameType": "R", "hydrate": "probablePitcher,linescore"},
        timeout=60,
    )
    r.raise_for_status()
    games = []
    for d in r.json().get("dates", []):
        gd = _d(d["date"])
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            innings = (g.get("linescore") or {}).get("innings") or []
            if not innings:
                continue
            first = innings[0]
            aw = first.get("away", {}).get("runs")
            hm = first.get("home", {}).get("runs")
            if aw is None or hm is None:
                continue
            home, away = g["teams"]["home"], g["teams"]["away"]
            games.append({
                "date": gd,
                "home_team": home["team"]["name"], "away_team": away["team"]["name"],
                "home_id": home["team"]["id"], "away_id": away["team"]["id"],
                "home_pid": (home.get("probablePitcher") or {}).get("id"),
                "away_pid": (away.get("probablePitcher") or {}).get("id"),
                "away_1st": aw, "home_1st": hm,
                "nrfi": 1 if (aw == 0 and hm == 0) else 0,
            })
    return games


def build_offense_history(games):
    """team_id -> sorted [(date, scored_in_1st_bool)] from every final."""
    hist = defaultdict(list)
    for g in games:
        hist[g["home_id"]].append((g["date"], g["home_1st"] > 0))
        hist[g["away_id"]].append((g["date"], g["away_1st"] > 0))
    for k in hist:
        hist[k].sort()
    return hist


def offense_scoreless_asof(hist, team_id, D):
    """Mirror nrfi_scanner.team_offense_scoreless but as-of D (exclusive)."""
    rows = [s for (dte, s) in hist.get(team_id, []) if dte < D]
    total = len(rows)
    if total == 0:
        return n.LEAGUE_OFF_SCORELESS
    scored = sum(rows)
    score_rate = (scored + n.TEAM_PRIOR_K * (1 - n.LEAGUE_OFF_SCORELESS)) / (total + n.TEAM_PRIOR_K)
    return 1 - score_rate


# ----------------------------------------------------------------------------
# Pitcher 1st-inning history (pulled once per pitcher, sliced by date)
# ----------------------------------------------------------------------------
def pitcher_history(pid, start_str, end_str, _cache={}):
    """Sorted [(date, runs_allowed_in_1st)] per start, from Statcast."""
    if pid in _cache:
        return _cache[pid]
    out = []
    try:
        df = n.statcast_pitcher(start_str, end_str, pid)
        if df is not None and not df.empty and "inning" in df.columns:
            d1 = df[df["inning"] == 1].copy()
            if not d1.empty:
                d1["runs"] = (d1["post_bat_score"] - d1["bat_score"]).clip(lower=0)
                d1["gd"] = pd.to_datetime(d1["game_date"]).dt.date
                by = d1.groupby(["game_pk", "gd"])["runs"].sum().reset_index()
                out = sorted((row.gd, int(row.runs)) for row in by.itertuples())
    except Exception:
        out = []
    _cache[pid] = out
    return out


def pitcher_rate_asof(hist, D):
    games = [r for (dte, r) in hist if dte < D]
    starts = len(games)
    clean = sum(1 for r in games if r == 0)
    return n.shrink(clean, starts), starts


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def _clip(p, lo=1e-6, hi=1 - 1e-6):
    return max(lo, min(hi, p))


def brier(preds, acts):
    return sum((p - a) ** 2 for p, a in zip(preds, acts)) / len(preds)


def log_loss(preds, acts):
    return -sum(a * math.log(_clip(p)) + (1 - a) * math.log(1 - _clip(p))
                for p, a in zip(preds, acts)) / len(preds)


def platt_fit(preds, acts):
    """Fit sigmoid(a + b*logit(p)) to outcomes via scipy; returns (a, b)."""
    from scipy.optimize import minimize
    xs = [math.log(_clip(p) / (1 - _clip(p))) for p in preds]

    def nll(params):
        a, b = params
        loss = 0.0
        for x, y in zip(xs, acts):
            z = a + b * x
            # numerically stable log-loss
            loss += math.log1p(math.exp(-abs(z))) + (max(z, 0) - z * y)
        return loss / len(xs)

    res = minimize(nll, [0.0, 1.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def recalibrate(p, a, b):
    x = math.log(_clip(p) / (1 - _clip(p)))
    z = a + b * x
    return 1 / (1 + math.exp(-z))


def reliability_table(preds, acts, edges):
    """Return rows [(lo, hi, n, mean_pred, actual_rate)]."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, p in enumerate(preds) if (lo <= p < hi) or (hi == edges[-1] and p == hi)]
        if not idx:
            continue
        mp = sum(preds[i] for i in idx) / len(idx)
        ar = sum(acts[i] for i in idx) / len(idx)
        rows.append((lo, hi, len(idx), mp, ar))
    return rows


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-start", default=n.SEASON_START,
                    help="start of the season used to build prior history")
    ap.add_argument("--start", required=True, help="first date to PREDICT (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last date to PREDICT (YYYY-MM-DD)")
    ap.add_argument("--min-starts", type=int, default=3,
                    help="require both starters to have >= this many prior starts")
    ap.add_argument("--csv", default="backtest_games.csv", help="per-game output CSV")
    ap.add_argument("--chart", default="calibration.png", help="reliability PNG (\"\" to skip)")
    args = ap.parse_args()

    season_start, start, end = args.season_start, _d(args.start), _d(args.end)
    if n.statcast_pitcher is None:
        print(f"pybaseball unavailable: {n._PYB_ERR}", file=sys.stderr)
        sys.exit(1)

    print(f"Backtest: predict {start}..{end}, history from {season_start}")
    full = fetch_full_schedule(season_start, args.end)
    print(f"  {len(full)} final games pulled (history + outcomes)")
    off_hist = build_offense_history(full)

    pred_games = [g for g in full if start <= g["date"] <= end
                  and g["home_pid"] and g["away_pid"]]
    print(f"  {len(pred_games)} games in the prediction window")

    # Pre-pull Statcast for every starter in the window (once each).
    pids = sorted({g["home_pid"] for g in pred_games} | {g["away_pid"] for g in pred_games})
    print(f"  pulling Statcast for {len(pids)} starters (cached after first run)...")
    for i, pid in enumerate(pids, 1):
        pitcher_history(pid, season_start, args.end)
        if i % 25 == 0:
            print(f"    {i}/{len(pids)}")

    preds, acts, records = [], [], []
    skipped = 0
    for g in pred_games:
        hh = pitcher_history(g["home_pid"], season_start, args.end)
        ah = pitcher_history(g["away_pid"], season_start, args.end)
        ph, hs = pitcher_rate_asof(hh, g["date"])
        pa, as_ = pitcher_rate_asof(ah, g["date"])
        if min(hs, as_) < args.min_starts:
            skipped += 1
            continue
        away_off = offense_scoreless_asof(off_hist, g["away_id"], g["date"])
        home_off = offense_scoreless_asof(off_hist, g["home_id"], g["date"])
        p_nrfi = n.log5(ph, away_off) * n.log5(pa, home_off)
        preds.append(p_nrfi)
        acts.append(g["nrfi"])
        records.append({
            "date": g["date"], "match": f'{g["away_team"]} @ {g["home_team"]}',
            "p_nrfi": round(p_nrfi, 4), "actual": g["nrfi"],
            "min_starts": min(hs, as_),
        })

    if not preds:
        print("No games met the criteria. Widen the window or lower --min-starts.")
        return

    ngames = len(preds)
    base = sum(acts) / ngames
    b_raw = brier(preds, acts)
    b_base = base * (1 - base)
    ll = log_loss(preds, acts)
    a, b = platt_fit(preds, acts)
    recal = [recalibrate(p, a, b) for p in preds]
    b_recal = brier(recal, acts)

    print(f"\n{'='*58}\nCALIBRATION — {ngames} games "
          f"({skipped} skipped for <{args.min_starts} starts)\n{'='*58}")
    print(f"Actual NRFI base rate : {base:.1%}")
    print(f"Mean model prediction : {sum(preds)/ngames:.1%}")
    print(f"Brier (model)         : {b_raw:.4f}")
    print(f"Brier (base-rate)     : {b_base:.4f}   <- beat this to add value")
    print(f"Brier (recalibrated)  : {b_recal:.4f}")
    print(f"Log-loss (model)      : {ll:.4f}")
    print(f"Platt fit             : a={a:+.3f}  b={b:.3f}"
          f"   ({'OVERCONFIDENT' if b < 0.9 else 'ok/underconfident'}; "
          "recal p = sigmoid(a + b·logit(p)))")

    edges = [0, .40, .45, .50, .55, .60, .65, .70, 1.0]
    print(f"\n{'bin':>13} {'n':>4} {'pred':>7} {'actual':>7} {'gap':>7}")
    for lo, hi, cnt, mp, ar in reliability_table(preds, acts, edges):
        print(f"  {lo:.2f}-{hi:.2f} {cnt:>4} {mp:>6.1%} {ar:>7.1%} {mp-ar:>+7.1%}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        print(f"\nPer-game predictions -> {args.csv}")

    if args.chart:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rows = reliability_table(preds, acts, edges)
            xs = [mp for _, _, _, mp, _ in rows]
            ys = [ar for _, _, _, _, ar in rows]
            ns = [cnt for _, _, cnt, _, _ in rows]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "--", color="#7a8699", label="perfect")
            ax.scatter(xs, ys, s=[max(30, c * 4) for c in ns], color="#4aa3ff",
                       alpha=.8, label="model (bubble = n)")
            ax.set_xlabel("Predicted NRFI probability")
            ax.set_ylabel("Actual NRFI rate")
            ax.set_title(f"NRFI reliability — {start}..{end} (n={ngames})")
            ax.set_xlim(.3, .8); ax.set_ylim(.3, .8)
            ax.legend(); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(args.chart, dpi=120)
            print(f"Reliability chart    -> {args.chart}")
        except Exception as e:
            print(f"(chart skipped: {e})")


if __name__ == "__main__":
    main()
