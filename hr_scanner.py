#!/usr/bin/env python3
"""
HR Scanner — v0 (model core)
----------------------------
Ranks each hitter in today's posted lineups by P(hits >= 1 HR), then (later)
prices that against anytime-HR props for edge/EV and emails a digest.

Model (per batter-game):
  p_pa = clamp( batter_HR/PA  ×  pitcher_HR-allowed/PA  /  league_HR/PA  ×  park )
  P(>=1 HR) = 1 - (1 - p_pa) ** expected_PA
where batter and pitcher rates are shrunk toward the league rate, `park` is the
home park's HR factor, and expected_PA comes from the batting-order slot.

Why HR and not NRFI: power is a stable, high-signal skill with strong known
covariates (park, weather, pitcher, handedness) — unlike single-inning run
suppression, which the backtest showed was noise. We still calibrate before
trusting any edge (see backtest, to come).

This file is the model core: it prints a ranked board. Odds/EV, email, weather,
handedness, and the backtest are layered on next.
"""

import os
import sys
import math
import statistics
import unicodedata
import datetime as dt
from collections import defaultdict

import requests

import nrfi_scanner as n  # reuse fetch_slate (slate + lineups + probables) and MLB_API

MLB_API = n.MLB_API
TODAY = os.getenv("SCAN_DATE") or dt.date.today().isoformat()
SEASON = int(os.getenv("SEASON", TODAY[:4]))

# Shrinkage strength (pseudo-observations toward league rate).
BAT_PRIOR_PA = float(os.getenv("BAT_PRIOR_PA", "170"))   # HR/PA stabilizes ~170 PA
PIT_PRIOR_BF = float(os.getenv("PIT_PRIOR_BF", "200"))   # HR/BF stabilizes slowly
P_PA_CAP = float(os.getenv("P_PA_CAP", "0.15"))          # sanity clamp on per-PA HR prob

# Anytime-HR odds via The Odds API (market key batter_home_runs, Over 0.5).
ODDS_API = "https://api.the-odds-api.com/v4"
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_REGIONS = os.getenv("ODDS_REGIONS", "us")
# A value play must appear at >= this many books (consensus guard). A lone soft
# book — common early in the day — produces fake longshot "edges", so gate them out.
MIN_BOOKS_VALUE = int(os.getenv("MIN_BOOKS_VALUE", "2"))

# Platt recalibration of the raw model probability, fitted by backtest_hr.py over
# ~18.8k batter-games (May-Aug 2026). The raw model is well-calibrated in the bulk
# but overconfident on longshots; this pulls the extremes back to reality.
#   p_cal = sigmoid(RECAL_A + RECAL_B * logit(p_raw))
# Refit periodically (re-run backtest_hr.py, paste the new a/b, or override via env).
RECAL_A = float(os.getenv("RECAL_A", "-0.559"))
RECAL_B = float(os.getenv("RECAL_B", "0.742"))

# Expected plate appearances by batting-order slot (index 0 = leadoff).
PA_BY_SLOT = [4.65, 4.55, 4.45, 4.35, 4.25, 4.13, 4.02, 3.92, 3.80]

# Approximate HR park factors (1.00 = league average). Keyed by home team name.
# Rough, static — replace with a maintained source later. Coors/GABP high; the
# pitcher's-park set (Oracle, loanDepot, Comerica) low.
PARK_HR = {
    "Colorado Rockies": 1.18, "Cincinnati Reds": 1.12, "New York Yankees": 1.10,
    "Milwaukee Brewers": 1.08, "Philadelphia Phillies": 1.07, "Boston Red Sox": 1.02,
    "Chicago Cubs": 1.02, "Arizona Diamondbacks": 1.03, "Atlanta Braves": 1.02,
    "Texas Rangers": 1.03, "Houston Astros": 1.02, "Toronto Blue Jays": 1.03,
    "Washington Nationals": 1.01, "Chicago White Sox": 1.03, "Baltimore Orioles": 1.01,
    "New York Mets": 0.97, "Los Angeles Dodgers": 1.01, "Los Angeles Angels": 1.02,
    "Minnesota Twins": 1.01, "St. Louis Cardinals": 0.95, "San Diego Padres": 0.94,
    "Seattle Mariners": 0.93, "Miami Marlins": 0.90, "Detroit Tigers": 0.93,
    "Pittsburgh Pirates": 0.92, "Cleveland Guardians": 0.98, "Kansas City Royals": 0.97,
    "Athletics": 1.05, "San Francisco Giants": 0.90, "Tampa Bay Rays": 1.00,
}
LEAGUE_PARK = 1.00


def _season_rate_table(group, keys):
    """player_id -> {key: value} for every player (playerPool=all)."""
    r = requests.get(
        f"{MLB_API}/stats",
        params={"stats": "season", "group": group, "season": SEASON,
                "sportId": 1, "limit": 3000, "playerPool": "all"},
        timeout=60,
    )
    r.raise_for_status()
    out = {}
    for s in r.json().get("stats", [{}])[0].get("splits", []):
        st = s.get("stat", {})
        out[s["player"]["id"]] = {k: st.get(k) for k in keys}
    return out


def load_rate_tables():
    bats = _season_rate_table("hitting", ("homeRuns", "plateAppearances"))
    pits = _season_rate_table("pitching", ("homeRuns", "battersFaced"))
    # League HR/PA from the batter pool (the anchor for the odds-ratio model).
    tot_hr = sum((v.get("homeRuns") or 0) for v in bats.values())
    tot_pa = sum((v.get("plateAppearances") or 0) for v in bats.values())
    league = (tot_hr / tot_pa) if tot_pa else 0.030
    return bats, pits, league


def batter_rate(bats, pid, league):
    v = bats.get(pid) or {}
    hr, pa = (v.get("homeRuns") or 0), (v.get("plateAppearances") or 0)
    return (hr + BAT_PRIOR_PA * league) / (pa + BAT_PRIOR_PA)


def pitcher_rate(pits, pid, league):
    v = pits.get(pid) or {}
    hr, bf = (v.get("homeRuns") or 0), (v.get("battersFaced") or 0)
    return (hr + PIT_PRIOR_BF * league) / (bf + PIT_PRIOR_BF)


def p_hr_game(bat_r, pit_r, league, park, slot):
    """Raw P(>=1 HR) for one batter vs one starter in one park, given lineup slot."""
    p_pa = bat_r * pit_r / league * park
    p_pa = max(0.0005, min(P_PA_CAP, p_pa))
    pa = PA_BY_SLOT[slot] if 0 <= slot < len(PA_BY_SLOT) else 4.0
    return 1 - (1 - p_pa) ** pa, p_pa


def recalibrate(p):
    """Apply the fitted Platt correction to a raw model probability."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    z = RECAL_A + RECAL_B * math.log(p / (1 - p))
    return 1 / (1 + math.exp(-z))


def _norm(s):
    """Accent- and punctuation-insensitive name key (García Jr. -> garciajr)."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c.lower() for c in s if c.isalnum())


def load_hr_odds(games):
    """(match, norm_name) -> {price, book, fair, n_books} for anytime HR (Over 0.5).
    price/book = best US payout; fair = median per-book de-vig where the Under side
    exists (anytime HR is often one-sided, in which case fair is None)."""
    ev = requests.get(f"{ODDS_API}/sports/baseball_mlb/events",
                      params={"apiKey": ODDS_API_KEY}, timeout=30)
    ev.raise_for_status()
    by_pair = {}
    for e in ev.json():
        by_pair.setdefault((n._norm_team(e.get("away_team")),
                            n._norm_team(e.get("home_team"))), []).append(e)

    out, calls = {}, 0
    for g in games:
        cands = by_pair.get((n._norm_team(g["away_team"]), n._norm_team(g["home_team"])))
        if not cands:
            continue
        eid = cands[0]["id"]
        r = requests.get(
            f"{ODDS_API}/sports/baseball_mlb/events/{eid}/odds",
            params={"apiKey": ODDS_API_KEY, "regions": ODDS_REGIONS,
                    "markets": "batter_home_runs", "oddsFormat": "american"},
            timeout=30)
        calls += 1
        if r.status_code != 200:
            continue
        match = f'{g["away_team"]} @ {g["home_team"]}'
        players = defaultdict(lambda: {"over": [], "under": {}})
        for bm in r.json().get("bookmakers", []):
            for mk in bm.get("markets", []):
                if mk.get("key") != "batter_home_runs":
                    continue
                for oc in mk.get("outcomes", []):
                    if abs(float(oc.get("point", 0)) - 0.5) > 1e-9:
                        continue  # only the 0.5 line is "anytime HR"
                    nm = _norm(oc.get("description"))
                    if oc.get("name") == "Over":
                        players[nm]["over"].append((bm.get("key"), oc.get("price")))
                    elif oc.get("name") == "Under":
                        players[nm]["under"][bm.get("key")] = oc.get("price")
        for nm, d in players.items():
            if not d["over"]:
                continue
            best_book, best_price = max(d["over"], key=lambda x: n.american_to_decimal(x[1]))
            fairs = [n.devig_two_way(pr, d["under"][bk]) for bk, pr in d["over"] if bk in d["under"]]
            out[(match, nm)] = {
                "price": best_price, "book": best_book,
                "fair": statistics.median(fairs) if fairs else None,
                "n_books": len(d["over"]),
            }
    print(f"  odds: {calls} event calls (~{calls + 1} credits), {len(out)} player prices")
    return out


def build_board():
    games = n.fetch_slate(TODAY)
    bats, pits, league = load_rate_tables()
    print(f"  league HR/PA = {league:.4f}  ({len(bats)} batters, {len(pits)} pitchers)")
    odds = load_hr_odds(games) if ODDS_API_KEY else {}

    board = []
    for g in games:
        park = PARK_HR.get(g["home_team"], LEAGUE_PARK)
        # Home lineup faces the AWAY starter; away lineup faces the HOME starter.
        sides = [
            ("home", g["home_lineup"], g["away_pid"], g["away_pitcher"]),
            ("away", g["away_lineup"], g["home_pid"], g["home_pitcher"]),
        ]
        for side, lineup, opp_pid, opp_name in sides:
            if len(lineup) < 9 or not opp_pid:
                continue  # need posted lineup + known opposing starter
            pit_r = pitcher_rate(pits, opp_pid, league)
            match = f'{g["away_team"]} @ {g["home_team"]}'
            for slot, (bid, bname) in enumerate(lineup):
                bat_r = batter_rate(bats, bid, league)
                p_raw, p_pa = p_hr_game(bat_r, pit_r, league, park, slot)
                p_hr = recalibrate(p_raw)
                row = {
                    "batter": bname, "team": g[f"{side}_team"],
                    "opp_p": opp_name, "park": park, "slot": slot + 1,
                    "p_hr": p_hr, "p_raw": p_raw, "fair": n.fair_american(p_hr),
                    "bat_hr_pa": bat_r, "pit_hr_pa": pit_r, "match": match,
                    "price": None, "book": None, "mkt_fair": None,
                    "edge": None, "ev": None, "n_books": None,
                }
                o = odds.get((match, _norm(bname)))
                if o:
                    row["price"], row["book"] = o["price"], o["book"]
                    row["mkt_fair"], row["n_books"] = o["fair"], o["n_books"]
                    row["ev"] = n.ev_per_unit(p_hr, o["price"])
                    if o["fair"] is not None:
                        row["edge"] = p_hr - o["fair"]
                board.append(row)
    board.sort(key=lambda r: r["p_hr"], reverse=True)
    return board


def main():
    print(f"HR scan {TODAY} (season {SEASON})...")
    board = build_board()
    if not board:
        print("No posted lineups yet — HR scanner runs best mid-afternoon.")
        return
    print(f"  {len(board)} batters in posted lineups\n")
    print("MODEL BOARD (top P(HR))")
    print(f"{'Batter':22} {'Team':20} {'Sl':>2} {'Pk':>5} {'raw':>5} {'P(HR)':>6} {'Fair':>6} {'Book':>6}")
    for r in board[:20]:
        bk = f'{r["price"]:+d}' if r["price"] is not None else "—"
        print(f"{r['batter'][:22]:22} {r['team'][:20]:20} {r['slot']:>2} "
              f"{r['park']:>5.2f} {r['p_raw']*100:4.1f}% {r['p_hr']*100:5.1f}% "
              f"{r['fair']:>+6d} {bk:>6}")

    priced = [r for r in board if r["ev"] is not None]
    trusted = [r for r in priced if (r["n_books"] or 0) >= MIN_BOOKS_VALUE]
    if trusted:
        trusted.sort(key=lambda r: r["ev"], reverse=True)
        print(f"\nVALUE BOARD (EV at best price, >={MIN_BOOKS_VALUE} books) — "
              f"{len(trusted)} of {len(priced)} priced")
        print(f"{'Batter':22} {'P(HR)':>6} {'Book':>6} {'Src':>10} {'#bk':>3} {'Edge':>7} {'EV/u':>7}")
        for r in trusted[:15]:
            edge = f'{r["edge"]*100:+.1f}pp' if r["edge"] is not None else "—"
            print(f"{r['batter'][:22]:22} {r['p_hr']*100:5.1f}% {r['price']:>+6d} "
                  f"{str(r['book'])[:10]:>10} {r['n_books']:>3} {edge:>7} {r['ev']*100:>+6.1f}%")
    elif priced:
        print(f"\nVALUE BOARD: {len(priced)} prices, but all single-book "
              f"(< {MIN_BOOKS_VALUE}) — no consensus, edges not trustworthy yet. "
              "Run mid-afternoon when DK/FD/etc. post.")


if __name__ == "__main__":
    main()
