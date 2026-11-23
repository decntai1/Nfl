#!/usr/bin/env python3
"""
nfl_matchup_card.py — the weekly matchup step: one side-by-side card per game, a matchup verdict on every priced
leg, and a running test of whether that verdict adds anything to the book's price.

    python nfl_matchup_card.py card                                   next week's games -> data/matchups_w<N>.md
    python nfl_matchup_card.py card --week 12                         every week-12 game
    python nfl_matchup_card.py card --week 11 --games CIN             one game; --priced adds the book's lines
    python nfl_matchup_card.py tag priced_week12/priced_*.csv         adds mx_exp / mx_p / mx_tag to each priced file
    python nfl_matchup_card.py backtest                               every graded ledger leg, walk-forward

What "matchup" means here. Every team gets an offence rating and a defence rating per metric, from this season's
games played BEFORE the game's date only (no peeking), shrunk toward the league with 3 average games, then
adjusted for schedule (a defence that faced weak offences is marked down, and the reverse, iterated):
  scoring  points, 1st-half points, Q1 points, TDs, FGs made, points of the last 3 games (form, not adjusted)
  efficiency  pass EPA/dropback, rush EPA/run, success rate, yards/play, explosive-play rate, 3rd-down %, red-zone TD %
  volume  plays, pass yards, rush yards
  trenches + ball  pressure rate (sack + half a QB hit per dropback), sacks, INTs, turnovers
  by position  receiving yards allowed to RB / WR / TE, rush yards allowed to RBs, passing yards allowed to QBs
Matchup expectation for A's offence against B's defence = league + (A offence - league) + (B defence - league),
home side +0.75 points. Player lines: the player's own per-game average this season x what the opposing defence
allows to his position group against the league (shrunk, clipped 0.7-1.4).

mx_p = that expectation turned into the chance the leg wins (normal for points and yards, Poisson for counts).
mx_tag = "+" when mx_p is 5+ points above the book's margin-free p, "-" when 5+ below, "0" otherwise.

The earlier tests (nfl_ratings.py, nfl_matchups.py) found team ratings lose to the book and defence-vs-position
adds under 1% to player projections. So mx_p is NOT a price. The question `backtest` answers is narrower: among
legs the book already rates 60-88% (the only ones the slip rules allow), do "-" legs lose more often than the book
says? If yes, "-" is a veto. If no, the card is reading material, and the numbers say so.

Results (23 Nov 2026):
  * vegas.hu ledger, weeks 8-11, 18,364 graded legs / 51 games: the matchup is less accurate than the book in every
    family (Brier: game points 0.2445 vs 0.2154, player props 0.1790 vs 0.1621), and a 50/50 mix is worse than the
    book alone. On game lines the tag ran BACKWARDS: book 60-88% legs tagged "-" won 83% (book said 70%), "+" won 60%
    (book said 69%). Only 42 games and legs inside one game move together (t ~1.9): not trusted on its own.
  * Closing lines 2018-2025, weeks 5+, 1,618 games: margin MAE matchup 10.53 vs market 9.92, total 11.12 vs 10.38.
    Siding with the matchup where it disagrees with the line by 3+ points: spread 46.7-51.7%, total 47.6-47.9%.
    Fading it lands near 52% — break-even at 1.91, not profit. 8 seasons out of 8, no stable edge either way.
  => The book already prices the matchup. mx_tag is NOT a veto and NOT a pick; read "+" as "the stats like this leg
     more than the book does", which history says is slightly more often wrong than right. Logged so the record
     keeps growing; re-run `backtest` after each graded week.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import pandas as pd

import unicodedata

import nfl_ingest as ing

# `card` needs only nfl_ingest.py (so it runs in the repo's GitHub Action); `tag` and `backtest` read vegas.hu
# markets and need nfl_odds_pricer.py next to this file.
BOOK_ABBR = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}


def _norm_name(x: str) -> str:
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower()
    x = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", x)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", "", x)).strip()


def classify(market, sel, teams):
    try:
        from nfl_odds_pricer import classify as c
    except ImportError:
        sys.exit("`tag` / `backtest` need nfl_odds_pricer.py in the same folder (it reads the vegas.hu market names)")
    return c(market, sel, teams)

OUT = os.environ.get("NFL_OUT", "./data")
K = 3            # pseudo-games of league average in every team rating
K_OPP = 4        # pseudo-games in the defence-vs-position factor
HOME = 0.75      # home points, each side of the margin
SD = {("pts", None): (9.5, 13.5, 13.5), ("pts", "h1"): (7.0, 10.0, 9.5), ("pts", "q1"): (4.8, 6.5, 6.0)}  # team, total, margin
CV = {"passing_yards": 0.30, "rushing_yards": 0.60, "receiving_yards": 0.70}
SDABS = {"completions": 5.0, "attempts": 6.0}
POISSON = {"receptions", "passing_tds", "passing_interceptions", "rush_td", "anytime"}
PSTAT = {"rush_td": "rushing_tds"}

# metric -> (higher is better for the offence?, label)
METRICS = {
    "pts": (True, "Points"), "pts_h1": (True, "1st-half points"), "pts_q1": (True, "Q1 points"),
    "tds": (True, "Touchdowns"), "fgm": (True, "Field goals made"),
    "pass_epa": (True, "Pass EPA / dropback"), "rush_epa": (True, "Rush EPA / run"), "success": (True, "Success rate"),
    "ypp": (True, "Yards / play"), "explosive": (True, "Explosive-play rate"), "third": (True, "3rd-down conversion"),
    "rz_td": (True, "Red-zone TD rate"), "plays": (True, "Plays"), "pass_yds": (True, "Pass yards"),
    "rush_yds": (True, "Rush yards"), "pressure": (False, "Pressure rate (allowed / made)"),
    "sacks": (False, "Sacks (allowed / made)"), "ints": (False, "INTs (thrown / caught)"),
    "tov": (False, "Turnovers (lost / forced)"),
}
POS_ROWS = [("RB", "receiving_yards", "Rec yards to RBs"), ("WR", "receiving_yards", "Rec yards to WRs"),
            ("TE", "receiving_yards", "Rec yards to TEs"), ("RB", "rushing_yards", "Rush yards to RBs"),
            ("QB", "passing_yards", "Pass yards to QBs")]


def code(x):
    x = str(x).upper()
    return BOOK_ABBR.get(x, {"ARZ": "ARI", "NOS": "NO", "PS": "PIT", "LAR": "LA"}.get(x, x))


# ===================================================================== data
class Data:
    def __init__(self, season=None):
        sch = ing.fetch("schedule")
        self.season = season or int(sch.loc[sch["result"].notna(), "season"].max())
        self.sch = sch[(sch["season"] == self.season) & (sch["game_type"] == "REG")].copy()
        self.sch["gameday"] = pd.to_datetime(self.sch["gameday"])
        p = ing.fetch("pbp", self.season)
        self.pbp = p[p["season_type"] == "REG"].copy()
        pw = ing.fetch("player_week", self.season)
        pw = pw[pw["season_type"] == "REG"].copy()
        pw["grp"] = pw["position"].replace({"FB": "RB"})
        pw["skill_tds"] = pw[["rushing_tds", "receiving_tds"]].fillna(0).sum(axis=1)
        self.pw = pw
        self.game_rows = self._team_games()
        self._cache = {}

    def _team_games(self) -> pd.DataFrame:
        """one row per (game, offence team, defence team) with every metric's per-game value."""
        p = self.pbp[self.pbp["posteam"].notna()].copy()
        play = p[((p["qb_dropback"] == 1) | (p["rush"] == 1)) & p["epa"].notna()
                 & (p["two_point_attempt"].fillna(0) == 0) & (p["qb_kneel"].fillna(0) == 0) & (p["qb_spike"].fillna(0) == 0)].copy()
        play["db"] = (play["qb_dropback"] == 1).astype(int)
        play["run"] = ((play["rush"] == 1) & (play["db"] == 0)).astype(int)
        play["sk"] = play["sack"].fillna(0)
        play["press"] = play["db"] * (play["sk"] + 0.5 * play["qb_hit"].fillna(0) * (1 - play["sk"]))
        play["y"] = play["yards_gained"].fillna(0)
        play["expl"] = ((play["db"] == 1) & (play["y"] >= 20)) | ((play["run"] == 1) & (play["y"] >= 10))
        play["tv"] = play["interception"].fillna(0) + play["fumble_lost"].fillna(0)
        play["third_n"] = (play["down"] == 3).astype(int)
        play["third_y"] = ((play["down"] == 3) & (play["third_down_converted"] == 1)).astype(int)
        k = ["game_id", "posteam", "defteam"]
        g = play.groupby(k).agg(
            plays=("epa", "size"), dbs=("db", "sum"), runs=("run", "sum"),
            pass_epa_s=("epa", lambda s: s[play.loc[s.index, "db"] == 1].sum()),
            rush_epa_s=("epa", lambda s: s[play.loc[s.index, "run"] == 1].sum()),
            success=("success", "mean"), ypp=("y", "mean"), explosive=("expl", "mean"),
            pressure_s=("press", "sum"), sacks=("sk", "sum"), ints=("interception", "sum"), tov=("tv", "sum"),
            third_n=("third_n", "sum"), third_y=("third_y", "sum"),
            pass_yds=("y", lambda s: s[play.loc[s.index, "db"] == 1].sum()),
            rush_yds=("y", lambda s: s[play.loc[s.index, "run"] == 1].sum())).reset_index()
        g["pass_epa"] = g["pass_epa_s"] / g["dbs"].replace(0, np.nan)
        g["rush_epa"] = g["rush_epa_s"] / g["runs"].replace(0, np.nan)
        g["pressure"] = g["pressure_s"] / g["dbs"].replace(0, np.nan)
        g["third"] = g["third_y"] / g["third_n"].replace(0, np.nan)
        # red zone: drives that reached the 20, share that ended in a TD
        d = p[p["fixed_drive"].notna()].groupby(["game_id", "posteam", "fixed_drive"]).agg(
            rz=("drive_inside20", "max"), res=("fixed_drive_result", "last")).reset_index()
        d = d[d["rz"] == 1]
        rz = d.assign(td=(d["res"] == "Touchdown").astype(float)).groupby(["game_id", "posteam"])["td"].mean().rename("rz_td")
        g = g.merge(rz.reset_index(), on=["game_id", "posteam"], how="left")
        # TDs (any kind credited to the team) and FGs made
        td = p[(p["touchdown"] == 1) & p["td_team"].notna()].groupby(["game_id", "td_team"]).size().rename("tds")
        td.index.names = ["game_id", "posteam"]
        fg = p[p["field_goal_result"] == "made"].groupby(["game_id", "posteam"]).size().rename("fgm")
        g = g.merge(td.reset_index(), on=["game_id", "posteam"], how="left").merge(fg.reset_index(), on=["game_id", "posteam"], how="left")
        g[["tds", "fgm"]] = g[["tds", "fgm"]].fillna(0)
        # points: final from the schedule, half and Q1 from the score at the start of Q3 / Q2
        s = self.sch[self.sch["result"].notna()].set_index("game_id")
        pts = []
        for gid, x in p.groupby("game_id"):
            if gid not in s.index:
                continue
            r = s.loc[gid]
            def at(q):
                y = x[x["qtr"] >= q]
                return (y["total_home_score"].min(), y["total_away_score"].min()) if len(y) else (np.nan, np.nan)
            h2, a2 = at(2)
            h3, a3 = at(3)
            pts += [(gid, r.home_team, r.home_score, h3, h2), (gid, r.away_team, r.away_score, a3, a2)]
        P = pd.DataFrame(pts, columns=["game_id", "posteam", "pts", "pts_h1", "pts_q1"])
        g = g.merge(P, on=["game_id", "posteam"], how="left")
        g = g.merge(self.sch[["game_id", "gameday", "week", "home_team"]], on="game_id", how="left")
        g["is_home"] = g["posteam"] == g["home_team"]
        return g

    # ----------------------------------------------------------------- ratings as of a date
    def ratings(self, before: pd.Timestamp) -> dict:
        key = pd.Timestamp(before).normalize()
        if key in self._cache:
            return self._cache[key]
        g = self.game_rows[self.game_rows["gameday"] < key]
        R = {"lg": {}, "off": {}, "def": {}, "games": g.groupby("posteam")["game_id"].nunique()}
        opps_of = g.groupby("posteam")["defteam"].apply(list)
        opps_def = g.groupby("defteam")["posteam"].apply(list)
        teams = sorted(set(g["posteam"]) | set(g["defteam"]))
        for m in METRICS:
            v = g[["posteam", "defteam", m]].dropna()
            lg = float(v[m].mean()) if len(v) else np.nan
            o = v.groupby("posteam")[m].agg(["sum", "count"])
            d = v.groupby("defteam")[m].agg(["sum", "count"])
            o = ((o["sum"] + K * lg) / (o["count"] + K)).reindex(teams).fillna(lg)
            d = ((d["sum"] + K * lg) / (d["count"] + K)).reindex(teams).fillna(lg)
            ao, ad = o.copy(), d.copy()
            for _ in range(10):
                no = o - (opps_of.reindex(teams).map(lambda s: ad.reindex(s).mean() if isinstance(s, list) else lg) - lg).fillna(0)
                nd = d - (opps_def.reindex(teams).map(lambda s: ao.reindex(s).mean() if isinstance(s, list) else lg) - lg).fillna(0)
                ao, ad = no, nd
            R["lg"][m], R["off"][m], R["def"][m] = lg, ao, ad
        # form: last 3 games, raw
        last = g.sort_values("gameday").groupby("posteam").tail(3)
        R["form_pf"] = last.groupby("posteam")["pts"].mean()
        la = g.sort_values("gameday").groupby("defteam").tail(3)
        R["form_pa"] = la.groupby("defteam")["pts"].mean()
        # defence vs position (per game allowed to a position group / league per game, shrunk)
        pw = self.pw[self.pw["game_id"].isin(set(g["game_id"]))]
        R["pw"] = pw
        stats = ["receiving_yards", "receptions", "rushing_yards", "passing_yards", "completions", "attempts",
                 "passing_tds", "passing_interceptions", "rushing_tds", "skill_tds", "carries", "targets"]
        pg = pw.groupby(["opponent_team", "grp", "game_id"])[stats].sum().reset_index()
        lgp = pg.groupby("grp")[stats].mean()
        agg = pg.groupby(["opponent_team", "grp"])[stats].agg(["sum", "count"])
        fac, allowed = {}, {}
        for (dteam, grp), row in agg.iterrows():
            for st in stats:
                L = lgp.loc[grp, st]
                if not L or L <= 0:
                    continue
                n = row[(st, "count")]
                per = (row[(st, "sum")] + K_OPP * L) / (n + K_OPP)
                fac[(dteam, grp, st)] = float(per / L)
                allowed[(dteam, grp, st)] = float(row[(st, "sum")] / n) if n else np.nan
        R["fac"], R["allowed"], R["lg_pos"] = fac, allowed, lgp
        self._cache[key] = R
        return R


def rank(series: pd.Series, high_good: bool) -> pd.Series:
    return series.rank(ascending=not high_good, method="min").astype(int)


def expect(R, m, off, dfn, home_off=None):
    lg = R["lg"][m]
    v = lg + (R["off"][m].get(off, lg) - lg) + (R["def"][m].get(dfn, lg) - lg)
    if m.startswith("pts") and home_off is not None:
        share = {"pts": 1.0, "pts_h1": 0.5, "pts_q1": 0.25}[m]
        v += HOME * share * (1 if home_off else -1)
    return v if m.endswith("epa") else max(v, 0.0)


# ===================================================================== card
def game_card(D: Data, game: pd.Series, priced: pd.DataFrame | None = None) -> tuple[str, list]:
    R = D.ratings(game["gameday"])
    h, a = game["home_team"], game["away_team"]
    off_rank = {m: rank(R["off"][m], hi) for m, (hi, _) in METRICS.items()}
    def_rank = {m: rank(R["def"][m], not hi) for m, (hi, _) in METRICS.items()}   # 1 = best defence
    lines, rows = [], []
    gp = R["games"]
    lines.append(f"## {a} @ {h} — {game['gameday']:%a %d %b}   (games played: {a} {gp.get(a, 0)}, {h} {gp.get(h, 0)}; "
                 f"data before kickoff day; ranks 1-32, 1 = best; schedule-adjusted)\n")
    for off, dfn in ((a, h), (h, a)):
        lines.append(f"**{off} offence vs {dfn} defence**\n")
        lines.append(f"| Metric | {off} off | rank | {dfn} def | rank | League | Matchup expects | Edge |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m, (hi, label) in METRICS.items():
            ov, dv, lg = R["off"][m].get(off, np.nan), R["def"][m].get(dfn, np.nan), R["lg"][m]
            ex = expect(R, m, off, dfn, home_off=(off == h))
            orank, drank = int(off_rank[m].get(off, 0)), int(def_rank[m].get(dfn, 0))
            gap = drank - orank          # >0: offence ranks better than the defence it faces
            edge = "offence ++" if gap >= 12 else "offence +" if gap >= 5 else "defence ++" if gap <= -12 \
                else "defence +" if gap <= -5 else "even"
            f = (lambda x: f"{x:.1%}") if m in ("success", "explosive", "third", "rz_td", "pressure") else \
                (lambda x: f"{x:+.3f}") if m.endswith("epa") else (lambda x: f"{x:.1f}")
            lines.append(f"| {label} | {f(ov)} | {orank} | {f(dv)} | {drank} | {f(lg)} | {f(ex)} | {edge} |")
            rows.append(dict(game=f"{a}@{h}", offence=off, defence=dfn, metric=m, off_val=ov, off_rank=orank,
                             def_val=dv, def_rank=drank, league=lg, matchup=ex, edge=edge))
        lines.append("")
    # by position
    lines.append("**What each defence allows by position (per game; factor vs league, rank 1 = allows least)**\n")
    lines.append(f"| Position line | {h} D allows | factor | rank | {a} D allows | factor | rank |")
    lines.append("|---|---|---|---|---|---|---|")
    for grp, st, label in POS_ROWS:
        f = pd.Series({d: R["fac"].get((d, grp, st), np.nan) for d in R["def"]["pts"].index}).dropna()
        rk = f.rank(method="min")
        cell = lambda d: (f"{R['allowed'].get((d, grp, st), np.nan):.1f}", f"{f.get(d, np.nan):.2f}", int(rk.get(d, 0)))  # noqa: E731
        hh, aa = cell(h), cell(a)
        lines.append(f"| {label} | {hh[0]} | {hh[1]} | {hh[2]} | {aa[0]} | {aa[1]} | {aa[2]} |")
    lines.append("")
    # game projection vs book
    pa, ph = expect(R, "pts", a, h, False), expect(R, "pts", h, a, True)
    pa1, ph1 = expect(R, "pts_h1", a, h, False), expect(R, "pts_h1", h, a, True)
    pa_q, ph_q = expect(R, "pts_q1", a, h, False), expect(R, "pts_q1", h, a, True)
    sk = expect(R, "sacks", a, h) + expect(R, "sacks", h, a)
    td = expect(R, "tds", a, h) + expect(R, "tds", h, a)
    bk = book_lines(priced, h, a) if priced is not None else {}
    lines.append("**Matchup projection vs the book**\n")
    lines.append("| | Matchup | Book (vegas.hu) | nflverse market |")
    lines.append("|---|---|---|---|")
    nv_s = -game["spread_line"] if pd.notna(game.get("spread_line")) else np.nan     # nflverse: + = home favoured
    lines.append(f"| {h} margin | {ph - pa:+.1f} | {bk.get('margin', '—')} | {(-nv_s if pd.notna(nv_s) else np.nan):+.1f} |")
    lines.append(f"| Total | {pa + ph:.1f} | {bk.get('total', '—')} | {game.get('total_line', np.nan)} |")
    lines.append(f"| {a} points | {pa:.1f} | {bk.get(a, '—')} | |")
    lines.append(f"| {h} points | {ph:.1f} | {bk.get(h, '—')} | |")
    lines.append(f"| 1st half {a} / {h} | {pa1:.1f} / {ph1:.1f} | | |")
    lines.append(f"| Q1 {a} / {h} | {pa_q:.1f} / {ph_q:.1f} | | |")
    lines.append(f"| Sacks (both teams) | {sk:.1f} | {bk.get('sacks', '—')} | |")
    lines.append(f"| TDs (both teams) | {td:.1f} | {bk.get('tds', '—')} | |")
    lines.append(f"| Form, last 3: pts for / against | {a} {R['form_pf'].get(a, np.nan):.1f} / {R['form_pa'].get(a, np.nan):.1f}, "
                 f"{h} {R['form_pf'].get(h, np.nan):.1f} / {R['form_pa'].get(h, np.nan):.1f} | | |")
    lines.append("")
    # top players
    lines.append("**Main players: season average x what this defence allows his position**\n")
    lines.append("| Player | Pos | Stat | Avg | Defence factor | Matchup expects | Games |")
    lines.append("|---|---|---|---|---|---|---|")
    for team, opp in ((a, h), (h, a)):
        for pid, nm, pos, st, avg, n in key_players(R["pw"], team):
            fct = float(np.clip(R["fac"].get((opp, pos, st), 1.0), 0.7, 1.4))
            lines.append(f"| {nm} ({team}) | {pos} | {st.replace('_', ' ')} | {avg:.1f} | {fct:.2f} | {avg * fct:.1f} | {n} |")
    lines.append("")
    return "\n".join(lines), rows


def key_players(pw, team, n_each={"QB": 1, "RB": 2, "WR": 3, "TE": 1}):
    x = pw[pw["team"] == team]
    out = []
    main = {"QB": "passing_yards", "RB": "rushing_yards", "WR": "receiving_yards", "TE": "receiving_yards"}
    last_wk = x["week"].max()
    for pos, k in n_each.items():
        y = x[x["grp"] == pos].groupby(["player_id", "player_display_name"]).agg(
            v=(main[pos], "mean"), n=("game_id", "nunique"), lw=("week", "max"), tot=(main[pos], "sum")).reset_index()
        y = y[(y["n"] >= 2) & (y["lw"] >= last_wk - 2)].sort_values("tot", ascending=False).head(k)
        for r in y.itertuples():
            out.append((r.player_id, r.player_display_name, pos, main[pos], r.v, r.n))
            if pos in ("RB",) and k:
                pass
    return out


def book_lines(d: pd.DataFrame, h, a) -> dict:
    """the book's main lines from a priced file: the line whose margin-free p is nearest 50%."""
    if d is None or d.empty:
        return {}
    d = d.copy()
    d["pb"] = d["p_book"] / (1 - d["p_push"].fillna(0))
    out = {}
    m = d["market"].astype(str)
    def near(mask):
        x = d[mask & d["pb"].notna()]
        return None if x.empty else x.iloc[(x["pb"] - 0.5).abs().argsort()].iloc[0]
    r = near(m.str.match(r"^Pontok száma \(hossz") & d["selection"].str.contains("felett"))
    if r is not None:
        out["total"] = re.search(r"[\d,]+", r.selection).group(0)
    r = near(m.str.match(r"^Hendikep \(hossz") & d["selection"].astype(str).str.startswith(h if h != "WAS" else "WAS"))
    if r is not None:
        out["margin"] = f"{-float(r.selection.split()[-1].replace(',', '.')):+.1f}"
    for t in (h, a):
        r = near(m.str.match(rf"^{t}\S* \S+\s+pontok száma \(hossz") & d["selection"].str.contains("felett"))
        if r is not None:
            out[t] = re.search(r"[\d,]+", r.selection).group(0)
    r = near(m.str.match(r"^Szerel.sek \(sack\) sz.ma") & d["selection"].str.contains("felett"))
    if r is not None:
        out["sacks"] = re.search(r"[\d,]+", r.selection).group(0)
    r = near(m.str.match(r"^Touchdown-ok sz.ma") & d["selection"].str.contains("felett"))
    if r is not None:
        out["tds"] = re.search(r"[\d,]+", r.selection).group(0)
    return out


def card(a):
    D = Data()
    if a.week is None:      # default: the next week with a game not yet played
        open_ = D.sch[D.sch["result"].isna()]
        if open_.empty:
            sys.exit("no unplayed regular-season games left")
        a.week = int(open_["week"].min())
        wk = open_[open_["week"] == a.week]       # only the games still to play (Monday: just MNF)
    else:
        wk = D.sch[D.sch["week"] == a.week]
    if a.games:
        wk = wk[wk["home_team"].isin(a.games) | wk["away_team"].isin(a.games)]
    if wk.empty:
        sys.exit("no games")
    priced = {}
    for f in a.priced or []:
        d = pd.read_csv(f, low_memory=False)
        if len(d):
            priced[d["game_id"].iloc[0]] = d
    md, rows = [f"# Matchup cards — {D.season} week {a.week}\n",
                "Ranks 1-32 (1 = best offence / best defence), from games before kickoff day, schedule-adjusted. "
                "Edge = how far apart the offence's rank and the defence's rank are (5+ ranks: +, 12+: ++). "
                "These ratings have NOT beaten the book on their own (nfl_ratings.py, Oct 31); read them as a check "
                "on a leg, not as a price.\n"], []
    for g in wk.sort_values("gameday").itertuples():
        g = pd.Series(g._asdict())
        txt, r = game_card(D, g, priced.get(g["game_id"]))
        md.append(txt)
        rows += r
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"matchups_w{a.week}{'_' + '_'.join(a.games) if a.games else ''}.md")
    open(path, "w").write("\n".join(md))
    pd.DataFrame(rows).to_csv(path.replace(".md", ".csv"), index=False)
    print("\n".join(md))
    print(f"\n-> {path}")


# ===================================================================== leg probability from the matchup
def norm_sf(x, mu, sd):
    return 0.5 * math.erfc((x - mu) / (sd * math.sqrt(2)))


def pois_ge(n, lam):
    lam = max(lam, 1e-6)
    return 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(int(n)))


class PlayerIndex:
    def __init__(self, pw):
        last = pw.sort_values("week").groupby("player_id").agg(name=("player_display_name", "last"),
                                                                 team=("team", "last"), grp=("grp", "last"))
        last["key"] = last["name"].map(_norm_name)
        self.t = last

    def find(self, name, team):
        if not name:
            return None
        k = _norm_name(name)
        c = self.t[self.t["team"] == team] if team else self.t
        hit = c.index[c["key"] == k]
        if len(hit):
            return hit[0]
        tok = k.split()
        if len(tok) >= 2:
            fl = c.index[c["key"].map(lambda s: s.split()[:1] == tok[:1] and s.split()[-1:] == tok[-1:])]
            if len(fl) == 1:
                return fl[0]
        import difflib
        m = difflib.get_close_matches(k, list(c["key"]), n=1, cutoff=0.85)
        return c.index[c["key"] == m[0]][0] if m else None


def leg_p(R, spec, home, away, pid=None, pidx=None):
    """(matchup expectation, P(leg wins)) or (None, None) when the leg is outside what the matchup covers."""
    t = spec.get("type")
    side = spec.get("side")
    line = spec.get("line")
    per = spec.get("per")
    if t in ("total", "period_total", "handicap", "period_handicap", "moneyline"):
        if per not in (None, "h1", "q1"):
            return None, None
        m = {None: "pts", "h1": "pts_h1", "q1": "pts_q1"}[per]
        sd_team, sd_tot, sd_mar = SD[("pts", per)]
        ph, pa = expect(R, m, home, away, True), expect(R, m, away, home, False)
        team = spec.get("team")
        if t in ("total", "period_total"):
            if line is None or side is None:
                return None, None
            if team in (home, away):
                mu, sd = (ph if team == home else pa), sd_team
            else:
                mu, sd = ph + pa, sd_tot
            q = norm_sf(line, mu, sd)
            return mu, (q if side == "over" else 1 - q)
        if team not in (home, away):
            return None, None
        mar = (ph - pa) if team == home else (pa - ph)
        hcap = 0.0 if t == "moneyline" else spec.get("h")
        if hcap is None:
            return None, None
        return mar, norm_sf(-hcap, mar, sd_mar)
    if t in ("tds", "fgs", "sacks"):
        if per or line is None or side is None:
            return None, None
        m = {"tds": "tds", "fgs": "fgm", "sacks": "sacks"}[t]
        team = spec.get("team")
        if t == "sacks":     # a team's sacks market = sacks MADE by that defence = sacks its opponent's offence takes
            mu = expect(R, m, away, home) if team == home else expect(R, m, home, away) if team == away else \
                expect(R, m, home, away) + expect(R, m, away, home)
        else:
            mu = expect(R, m, team, away if team == home else home) if team in (home, away) else \
                expect(R, m, home, away) + expect(R, m, away, home)
        q = pois_ge(math.floor(line) + 1, mu)
        return mu, (q if side == "over" else 1 - q)
    stat = PSTAT.get(t, t)
    if t in ("passing_yards", "rushing_yards", "receiving_yards", "receptions", "completions", "attempts",
             "passing_tds", "passing_interceptions", "rush_td", "anytime"):
        if pid is None and pidx is not None:
            pid = pidx.find(spec.get("player"), spec.get("pteam"))
        if pid is None:
            return None, None
        pw = R["pw"]
        x = pw[pw["player_id"] == pid]
        if len(x) < 3:
            return None, None
        team = x.sort_values("week")["team"].iloc[-1]
        if team not in (home, away):
            return None, None
        opp = away if team == home else home
        grp = x["grp"].iloc[-1]
        st = "skill_tds" if t == "anytime" else stat
        avg = float(x[st].fillna(0).mean())
        fct = float(np.clip(R["fac"].get((opp, grp, st), 1.0), 0.7, 1.4))
        mu = avg * fct
        if t == "anytime":
            n = spec.get("n", 1) or 1
            return mu, pois_ge(n, mu)
        if line is None or side is None:
            return None, None
        if t in POISSON:
            q = pois_ge(math.floor(line) + 1, mu)
        else:
            sd = SDABS.get(t) or max(CV[t] * mu, 8.0)
            q = norm_sf(line, mu, sd)
        return mu, (q if side == "over" else 1 - q)
    return None, None


def tag_frame(D: Data, d: pd.DataFrame, pid_col=None) -> pd.DataFrame:
    gid = d["game_id"].iloc[0]
    g = D.sch[D.sch["game_id"] == gid]
    if g.empty:
        return d.assign(mx_exp=np.nan, mx_p=np.nan, mx_tag="")
    g = g.iloc[0]
    home, away = g["home_team"], g["away_team"]
    R = D.ratings(g["gameday"])
    pidx = PlayerIndex(R["pw"])
    teams = {home: home, away: away}
    ex, pp = [], []
    for i, r in enumerate(d.itertuples()):
        try:
            spec = classify(str(r.market), str(r.selection), teams)
        except Exception:
            spec = {"type": None}
        if spec.get("pteam"):
            spec["pteam"] = code(spec["pteam"])
        if spec.get("team"):
            spec["team"] = code(spec["team"])
        pid = None
        if pid_col and isinstance(getattr(r, pid_col), str) and getattr(r, pid_col).startswith("00-"):
            pid = getattr(r, pid_col)
        e, p = leg_p(R, spec, home, away, pid, pidx)
        ex.append(e)
        pp.append(p)
    d = d.copy()
    d["mx_exp"] = np.round(pd.to_numeric(pd.Series(ex, index=d.index), errors="coerce"), 2)
    d["mx_p"] = np.round(pd.to_numeric(pd.Series(pp, index=d.index), errors="coerce"), 3)
    pb = d["p_book"] / (1 - d["p_push"].fillna(0)) if "p_push" in d else d["p_book"]
    diff = d["mx_p"] - pb
    d["mx_tag"] = np.where(d["mx_p"].isna(), "", np.where(diff >= 0.05, "+", np.where(diff <= -0.05, "-", "0")))
    return d


def tag(a):
    D = Data()
    for f in a.priced:
        d = pd.read_csv(f, low_memory=False)
        if d.empty:
            continue
        d = tag_frame(D, d.drop(columns=[c for c in ("mx_exp", "mx_p", "mx_tag") if c in d]))
        d.to_csv(f, index=False)
        n = (d["mx_tag"] != "").sum()
        print(f"  {os.path.basename(f):<40} {n:>5} legs tagged  (+ {(d.mx_tag == '+').sum()}, - {(d.mx_tag == '-').sum()})")


# ===================================================================== backtest
def backtest(a):
    D = Data()
    L = pd.read_csv(os.path.join(OUT, "nfl_odds_ledger.csv"), low_memory=False)
    L = L[L["result"].isin(["won", "lost"]) & L["p_book"].notna() & L["odds"].notna()]
    L = L.drop_duplicates(["game_id", "market", "selection"])
    L["p_push"] = L.get("p_push_now", L["p_push"]).fillna(0)
    parts = []
    for gid, d in L.groupby("game_id"):
        parts.append(tag_frame(D, d, pid_col="player_id"))
    T = pd.concat(parts, ignore_index=True)
    T = T[T["mx_p"].notna()].copy()
    T["pb"] = T["p_book"] / (1 - T["p_push"].fillna(0))
    T["y"] = (T["result"] == "won").astype(float)
    T["profit"] = np.where(T["y"] == 1, T["odds"] - 1, -1)
    T["fam"] = np.select([T["type_now"].isin(["total", "handicap", "moneyline"]),
                          T["type_now"].isin(["period_total", "period_handicap"]),
                          T["type_now"].isin(["tds", "fgs", "sacks"])],
                         ["game points", "half / Q1 points", "TDs / FGs / sacks"], "player props")
    T.to_csv(os.path.join(OUT, "nfl_matchup_backtest_legs.csv"), index=False)
    print(f"{len(T):,} graded legs covered, {T.game_id.nunique()} games, weeks {sorted(T.game_id.str[5:7].unique())}\n")
    print("Accuracy (Brier, lower = better):")
    for fam, x in T.groupby("fam"):
        bl = 0.5 * x["pb"] + 0.5 * x["mx_p"]
        print(f"  {fam:<20} n {len(x):>6}  book {((x.pb - x.y) ** 2).mean():.4f}  matchup {((x.mx_p - x.y) ** 2).mean():.4f}"
              f"  50/50 mix {((bl - x.y) ** 2).mean():.4f}")

    def line(name, x):
        if len(x) < 20:
            print(f"  {name:<44} n {len(x)} — too few")
            return
        pg = x.groupby("game_id")["profit"].mean()
        diff = x.groupby("game_id").apply(lambda z: (z.y - z.pb).mean())
        t = diff.mean() / (diff.std() / np.sqrt(len(diff))) if len(diff) > 2 else np.nan
        print(f"  {name:<44} n {len(x):>5} {x.game_id.nunique():>3}g  won {x.y.mean():.1%}  book said {x.pb.mean():.1%}"
              f"  diff {x.y.mean() - x.pb.mean():+.1%} (t {t:+.1f})  return {x.profit.mean():+.1%}")

    for title, X in (("ALL legs the matchup covers", T), ("Legs the book rates 60-88% (the slip rule)", T[T.pb.between(0.6, 0.88)])):
        print(f"\n{title}:")
        for tg, lab in (("+", "matchup says MORE likely than book (+)"), ("0", "matchup agrees with book (0)"),
                        ("-", "matchup says LESS likely than book (-)")):
            line(lab, X[X.mx_tag == tg])
        for fam, x in X.groupby("fam"):
            print(f"   {fam}:")
            for tg in ("+", "0", "-"):
                line(f"     tag {tg}", x[x.mx_tag == tg])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("card")
    c.add_argument("--week", type=int, default=None, help="default: the next week with unplayed games")
    c.add_argument("--games", nargs="*", default=[])
    c.add_argument("--priced", nargs="*", default=[])
    t = sub.add_parser("tag")
    t.add_argument("priced", nargs="+")
    sub.add_parser("backtest")
    a = ap.parse_args()
    {"card": card, "tag": tag, "backtest": backtest}[a.cmd](a)
