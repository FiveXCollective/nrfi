#!/usr/bin/env python3
"""
HR model backtest / calibration
-------------------------------
Same discipline as the NRFI backtest, harder data problem: to score a batter on
date D we need his HR/PA and the opposing starter's HR-allowed/BF as of BEFORE D,
plus whether he actually homered that day.

We make ONE chronological pass over boxscores. Accumulators hold season-to-date
batter/pitcher rates; when we reach date D we predict every starter using the
accumulator (which contains only games before D), record the actual outcome from
that day's boxscores, and only THEN fold the day's games into the accumulators.
No look-ahead, and each boxscore supplies the lineup, the starter, and outcomes.

Reuses the model math from hr_scanner and the metrics from backtest.

Usage (validate short, then real):
  python backtest_hr.py --prior-start 2026-07-15 --start 2026-08-01 --end 2026-08-05
  python backtest_hr.py --prior-start 2026-03-01 --start 2026-05-01 --end 2026-08-07
"""

import argparse
import csv
import datetime as dt
import sys
from collections import defaultdict

import requests

import hr_scanner as h
import backtest as bt  # reuse brier, log_loss, platt_fit, recalibrate, reliability_table

MLB_API = h.MLB_API


def _d(s):
    return dt.date.fromisoformat(s)


def final_games(prior_start, end):
    """[(date, gamePk, home_team_name)] for finals in the range, date-sorted."""
    r = requests.get(
        f"{MLB_API}/schedule",
        params={"sportId": 1, "startDate": prior_start, "endDate": end,
                "gameType": "R"},
        timeout=60,
    )
    r.raise_for_status()
    out = []
    for d in r.json().get("dates", []):
        gd = _d(d["date"])
        for g in d.get("games", []):
            if g.get("status", {}).get("abstractGameState") == "Final":
                out.append((gd, g["gamePk"], g["teams"]["home"]["team"]["name"]))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def boxscore(pk):
    try:
        return requests.get(f"{MLB_API}/game/{pk}/boxscore", timeout=30).json()
    except Exception:
        return None


def _bat(players, pid):
    return (players.get(f"ID{pid}", {}).get("stats", {}).get("batting", {}) or {})


def _pit(players, pid):
    return (players.get(f"ID{pid}", {}).get("stats", {}).get("pitching", {}) or {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior-start", default=h.TODAY[:4] + "-03-01",
                    help="date to start accumulating priors (season start)")
    ap.add_argument("--start", required=True, help="first date to PREDICT")
    ap.add_argument("--end", required=True, help="last date to PREDICT")
    ap.add_argument("--min-pa", type=int, default=50, help="min prior PA for a batter")
    ap.add_argument("--min-bf", type=int, default=80, help="min prior BF for a starter")
    ap.add_argument("--csv", default="backtest_hr_games.csv")
    ap.add_argument("--chart", default="calibration_hr.png")
    args = ap.parse_args()

    start, end = _d(args.start), _d(args.end)
    games = final_games(args.prior_start, args.end)
    print(f"HR backtest: predict {start}..{end}, priors from {args.prior_start}")
    print(f"  {len(games)} final games to walk")

    # Accumulators (as-of, updated day by day)
    bat = defaultdict(lambda: [0, 0])   # pid -> [hr, pa]
    pit = defaultdict(lambda: [0, 0])   # pid -> [hr_allowed, bf]
    tot = [0, 0]                        # league [hr, pa]

    by_date = defaultdict(list)
    for gd, pk, home in games:
        by_date[gd].append((pk, home))

    preds, acts, records = [], [], []
    skipped = 0

    for gd in sorted(by_date):
        day_boxes = []
        for pk, home_team in by_date[gd]:
            bs = boxscore(pk)
            if not bs:
                continue
            day_boxes.append((pk, home_team, bs))

            # --- Predict (only if in window), using accumulators BEFORE today ---
            if start <= gd <= end:
                league = (tot[0] / tot[1]) if tot[1] else 0.030
                park = h.PARK_HR.get(home_team, h.LEAGUE_PARK)
                teams = bs.get("teams", {})
                for side, opp in (("home", "away"), ("away", "home")):
                    t = teams.get(side, {})
                    o = teams.get(opp, {})
                    order = t.get("battingOrder") or []
                    opp_pitchers = o.get("pitchers") or []
                    if len(order) < 9 or not opp_pitchers:
                        continue
                    sp = opp_pitchers[0]
                    phr, pbf = pit[sp]
                    if pbf < args.min_bf:
                        continue
                    pit_r = (phr + h.PIT_PRIOR_BF * league) / (pbf + h.PIT_PRIOR_BF)
                    for slot, bid in enumerate(order[:9]):
                        bhr, bpa = bat[bid]
                        if bpa < args.min_pa:
                            skipped += 1
                            continue
                        bat_r = (bhr + h.BAT_PRIOR_PA * league) / (bpa + h.BAT_PRIOR_PA)
                        p_hr, _ = h.p_hr_game(bat_r, pit_r, league, park, slot)
                        game_bat = _bat(t.get("players", {}), bid)
                        actual = 1 if (game_bat.get("homeRuns") or 0) >= 1 else 0
                        preds.append(p_hr)
                        acts.append(actual)
                        records.append({
                            "date": gd, "batter_id": bid,
                            "name": t.get("players", {}).get(f"ID{bid}", {}).get("person", {}).get("fullName", ""),
                            "slot": slot + 1, "park": park,
                            "p_hr": round(p_hr, 4), "actual": actual,
                        })

        # --- Fold today's games into accumulators (after predicting) ---
        for pk, home_team, bs in day_boxes:
            for side in ("home", "away"):
                players = bs.get("teams", {}).get(side, {}).get("players", {})
                for pdata in players.values():
                    st = pdata.get("stats", {})
                    b = st.get("batting", {}) or {}
                    pa = b.get("plateAppearances")
                    if pa:
                        hr = b.get("homeRuns") or 0
                        bat[pdata["person"]["id"]][0] += hr
                        bat[pdata["person"]["id"]][1] += pa
                        tot[0] += hr
                        tot[1] += pa
                    p = st.get("pitching", {}) or {}
                    bf = p.get("battersFaced")
                    if bf:
                        pit[pdata["person"]["id"]][0] += (p.get("homeRuns") or 0)
                        pit[pdata["person"]["id"]][1] += bf

    if not preds:
        print("No predictions — widen the window or lower --min-pa/--min-bf.")
        return

    ngames = len(preds)
    base = sum(acts) / ngames
    b_raw = bt.brier(preds, acts)
    b_base = base * (1 - base)
    ll = bt.log_loss(preds, acts)
    a, b = bt.platt_fit(preds, acts)
    recal = [bt.recalibrate(p, a, b) for p in preds]
    b_recal = bt.brier(recal, acts)

    print(f"\n{'='*58}\nHR CALIBRATION — {ngames} batter-games "
          f"({skipped} skipped for thin priors)\n{'='*58}")
    print(f"Actual HR base rate   : {base:.1%}")
    print(f"Mean model prediction : {sum(preds)/ngames:.1%}")
    print(f"Brier (model)         : {b_raw:.4f}")
    print(f"Brier (base-rate)     : {b_base:.4f}   <- beat this to add value")
    print(f"Brier (recalibrated)  : {b_recal:.4f}")
    print(f"Log-loss (model)      : {ll:.4f}")
    print(f"Platt fit             : a={a:+.3f}  b={b:.3f}"
          f"   ({'OVERCONFIDENT' if b < 0.9 else 'ok/underconfident'}; "
          "recal p = sigmoid(a + b·logit(p)))")

    edges = [0, .05, .08, .10, .12, .14, .16, .18, .22, .28, 1.0]
    print(f"\n{'bin':>13} {'n':>5} {'pred':>7} {'actual':>7} {'gap':>7}")
    for lo, hi, cnt, mp, ar in bt.reliability_table(preds, acts, edges):
        print(f"  {lo:.2f}-{hi:.2f} {cnt:>5} {mp:>6.1%} {ar:>7.1%} {mp-ar:>+7.1%}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader(); w.writerows(records)
        print(f"\nPer batter-game -> {args.csv}")

    if args.chart:
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rows = bt.reliability_table(preds, acts, edges)
            xs = [mp for *_, mp, _ in rows]; ys = [ar for *_, ar in rows]
            ns = [c for _, _, c, _, _ in rows]
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot([0, 1], [0, 1], "--", color="#7a8699", label="perfect")
            ax.scatter(xs, ys, s=[max(30, c / 3) for c in ns], color="#e0663a",
                       alpha=.8, label="model (bubble = n)")
            ax.set_xlabel("Predicted P(HR)"); ax.set_ylabel("Actual HR rate")
            ax.set_title(f"HR reliability — {start}..{end} (n={ngames})")
            ax.set_xlim(0, .4); ax.set_ylim(0, .4)
            ax.legend(); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(args.chart, dpi=120)
            print(f"Reliability chart -> {args.chart}")
        except Exception as e:
            print(f"(chart skipped: {e})")


if __name__ == "__main__":
    main()
