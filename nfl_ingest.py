#!/usr/bin/env python3
"""
nfl_ingest.py — download + derive NFL tables (all 32 teams, every game, every player,
betting lines/results), verify keys, write a constrained DuckDB, publish CSVs to ProLyt.

Source: nflverse GitHub releases (public, no key, no IP blocks) — the same data that
nflreadr / nfl_data_py load. Updated nightly during the season.

Env:
  PROLYT_API_KEY   upload key (falls back to ANALYTIQ_API_KEY). Unset -> build locally, skip publish.
  PROLYT_URL       platform URL (falls back to ANALYTIQ_URL, default https://analytiq.dcentai.tech)
  NFL_SEASONS      "2026" or "2025 2026" (default: current season; NFL season = its start year)
  NFL_OUT          output dir (default ./data)
  NFL_PBP          1 (default) = also publish a slim play-by-play table; 0 = skip it
Needs: pandas, numpy, requests, pyarrow   (duckdb optional — only for the local .duckdb)

Tables (prefix nfl_<seasons>_):
  teams, schedule, team_game_stats, team_season_stats, team_advanced, quarter_scores,
  quarter_wide, player_game_stats, player_season_stats, snap_counts, injuries, rosters,
  depth_chart, adv_passing, adv_rushing, adv_receiving, adv_defense, ngs_passing,
  ngs_rushing, ngs_receiving, betting_lines, betting_team, betting_summary, [pbp]
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {   # per-season files: {yr}; whole-history files have no {yr} and are filtered by season
    "schedule":      "schedules/games.parquet",
    "teams":         "teams/teams_colors_logos.parquet",
    "player_week":   "stats_player/stats_player_week_{yr}.parquet",
    "player_reg":    "stats_player/stats_player_reg_{yr}.parquet",
    "team_week":     "stats_team/stats_team_week_{yr}.parquet",
    "team_reg":      "stats_team/stats_team_reg_{yr}.parquet",
    "pbp":           "pbp/play_by_play_{yr}.parquet",
    "rosters":       "weekly_rosters/roster_weekly_{yr}.parquet",
    "snaps":         "snap_counts/snap_counts_{yr}.parquet",
    "injuries":      "injuries/injuries_{yr}.parquet",
    "depth":         "depth_charts/depth_charts_{yr}.parquet",
    "adv_pass":      "pfr_advstats/advstats_week_pass_{yr}.parquet",
    "adv_rush":      "pfr_advstats/advstats_week_rush_{yr}.parquet",
    "adv_rec":       "pfr_advstats/advstats_week_rec_{yr}.parquet",
    "adv_def":       "pfr_advstats/advstats_week_def_{yr}.parquet",
    "ngs_passing":   "nextgen_stats/ngs_passing.parquet",
    "ngs_rushing":   "nextgen_stats/ngs_rushing.parquet",
    "ngs_receiving": "nextgen_stats/ngs_receiving.parquet",
}


# ================================================================ download
def fetch(kind: str, year: int | None = None, retries: int = 3) -> pd.DataFrame:
    url = f"{BASE}/{FILES[kind].format(yr=year)}"
    tag = f"{kind} {year}" if "{yr}" in FILES[kind] else kind
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 404:
                print(f"  [--] {tag}: not published (404)")
                return pd.DataFrame()
            r.raise_for_status()
            df = pd.read_parquet(io.BytesIO(r.content))
            print(f"  [ok] {tag}: {len(df):,} rows")
            return df
        except Exception as e:
            if attempt == retries:
                raise RuntimeError(f"{tag} failed: {e}") from e
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def load(kind: str, seasons: list[int]) -> pd.DataFrame:
    if "{yr}" not in FILES[kind]:
        df = fetch(kind)
        return df[df["season"].isin(seasons)].reset_index(drop=True) if "season" in df.columns else df
    parts = [p for p in (fetch(kind, y) for y in seasons) if not p.empty]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ================================================================ builders
def build_teams(teams: pd.DataFrame, sch: pd.DataFrame) -> pd.DataFrame:
    used = set(sch["home_team"]) | set(sch["away_team"])
    keep = ["team_abbr", "team_name", "team_nick", "team_conf", "team_division", "team_color", "team_logo_espn"]
    t = teams[teams["team_abbr"].isin(used)][keep].drop_duplicates("team_abbr")
    return t.sort_values("team_abbr").reset_index(drop=True)


def build_schedule(sch: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    s = sch.drop(columns=[c for c in ("old_game_id", "gsis", "nfl_detail_id", "pfr", "pff", "espn", "ftn")
                          if c in sch.columns]).copy()
    names = teams.set_index("team_abbr")["team_name"]
    s.insert(s.columns.get_loc("away_team") + 1, "away_team_name", s["away_team"].map(names))
    s.insert(s.columns.get_loc("home_team") + 1, "home_team_name", s["home_team"].map(names))
    s["gameday"] = pd.to_datetime(s["gameday"]).dt.date
    s["status"] = np.where(s["home_score"].notna(), "final", "scheduled")
    s["winner"] = np.select([s["result"] > 0, s["result"] < 0, s["result"] == 0],
                            [s["home_team"], s["away_team"], "TIE"], default=None)
    return s.sort_values(["season", "week", "gameday", "gametime", "game_id"]).reset_index(drop=True)


def _team_rows(sch: pd.DataFrame) -> pd.DataFrame:
    """one row per team per game (completed and scheduled) — the backbone of team tables."""
    base = ["game_id", "season", "game_type", "week", "gameday"]
    h = sch[base + ["home_team", "away_team", "home_score", "away_score", "home_rest", "away_rest"]].rename(
        columns={"home_team": "team", "away_team": "opponent", "home_score": "points_for",
                 "away_score": "points_against", "home_rest": "rest_days", "away_rest": "opp_rest_days"})
    h["is_home"] = True
    a = sch[base + ["away_team", "home_team", "away_score", "home_score", "away_rest", "home_rest"]].rename(
        columns={"away_team": "team", "home_team": "opponent", "away_score": "points_for",
                 "home_score": "points_against", "away_rest": "rest_days", "home_rest": "opp_rest_days"})
    a["is_home"] = False
    r = pd.concat([h, a], ignore_index=True)
    neutral = sch.set_index("game_id")["location"].eq("Neutral")
    r.loc[r["game_id"].map(neutral).fillna(False).astype(bool), "is_home"] = False
    r["margin"] = r["points_for"] - r["points_against"]
    r["result"] = np.select([r["margin"] > 0, r["margin"] < 0, r["margin"] == 0], ["W", "L", "T"], default=None)
    return r


def build_team_game_stats(tw: pd.DataFrame, sch: pd.DataFrame) -> pd.DataFrame:
    rows = _team_rows(sch)
    tw = tw.drop(columns=[c for c in ("season", "week", "season_type", "opponent_team") if c in tw.columns])
    df = rows.merge(tw, on=["game_id", "team"], how="inner")      # completed games only
    df["total_yards"] = df["passing_yards"].fillna(0) + df["rushing_yards"].fillna(0)
    df["turnovers"] = df["passing_interceptions"].fillna(0) + df["fumbles_lost_total"].fillna(0)
    df["takeaways"] = df["def_interceptions"].fillna(0) + df["fumble_recovery_opp"].fillna(0)
    # pre-game form (no leakage: shift(1) = only games BEFORE this one)
    df = df.sort_values(["team", "season", "week"])
    g = df.groupby(["team", "season"])
    for c in ("points_for", "points_against", "total_yards", "turnovers"):
        df[f"{c}_season_avg_pre"] = g[c].transform(lambda s: s.shift(1).expanding().mean()).round(2)
        df[f"{c}_l3_avg_pre"] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).round(2)
    return df.sort_values(["season", "week", "game_id", "is_home"]).reset_index(drop=True)


def build_team_season(tr: pd.DataFrame, tgs: pd.DataFrame) -> pd.DataFrame:
    rec = (tgs[tgs["game_type"] == "REG"].groupby(["season", "team"])
           .agg(wins=("result", lambda s: (s == "W").sum()), losses=("result", lambda s: (s == "L").sum()),
                ties=("result", lambda s: (s == "T").sum()), points_for=("points_for", "sum"),
                points_against=("points_against", "sum")).reset_index())
    rec["point_diff"] = rec["points_for"] - rec["points_against"]
    return tr.merge(rec, on=["season", "team"], how="left").sort_values(["season", "team"]).reset_index(drop=True)


def _mmss(v) -> float:
    try:
        m, s = str(v).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return np.nan


def build_team_advanced(pbp: pd.DataFrame, sch: pd.DataFrame) -> pd.DataFrame:
    """per team-game efficiency from play-by-play: EPA, success, explosiveness, situational."""
    p = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)].copy()
    p = p[p["posteam"].notna()]
    p["explosive"] = ((p["pass"] == 1) & (p["yards_gained"] >= 20)) | ((p["rush"] == 1) & (p["yards_gained"] >= 10))
    p["early_down"] = p["down"].isin([1, 2])
    g = p.groupby(["game_id", "posteam"])
    off = pd.DataFrame({
        "plays": g.size(),
        "epa_per_play": g["epa"].mean(),
        "success_rate": g["success"].mean(),
        "pass_plays": g["pass"].sum(),
        "rush_plays": g["rush"].sum(),
        "pass_epa_per_play": p[p["pass"] == 1].groupby(["game_id", "posteam"])["epa"].mean(),
        "rush_epa_per_play": p[p["rush"] == 1].groupby(["game_id", "posteam"])["epa"].mean(),
        "early_down_pass_rate": p[p["early_down"]].groupby(["game_id", "posteam"])["pass"].mean(),
        "explosive_plays": g["explosive"].sum(),
        "explosive_rate": g["explosive"].mean(),
        "sacks_taken": g["sack"].sum(),
        "cpoe": g["cpoe"].mean(),
        "pass_oe": g["pass_oe"].mean(),
    })
    allp = pbp[pbp["posteam"].notna()]
    ga = allp.groupby(["game_id", "posteam"])
    third_c, third_f = ga["third_down_converted"].sum(), ga["third_down_failed"].sum()
    fourth_c, fourth_f = ga["fourth_down_converted"].sum(), ga["fourth_down_failed"].sum()
    off["third_down_att"] = third_c + third_f
    off["third_down_conv_rate"] = third_c / (third_c + third_f).replace(0, np.nan)
    off["fourth_down_att"] = fourth_c + fourth_f
    off["fourth_down_conv_rate"] = fourth_c / (fourth_c + fourth_f).replace(0, np.nan)
    off["giveaways"] = ga["interception"].sum() + ga["fumble_lost"].sum()
    off["penalties"] = allp[allp["penalty_team"] == allp["posteam"]].groupby(["game_id", "posteam"])["penalty"].sum()
    # drives: red zone + scoring + time of possession (one row per drive)
    d = allp.dropna(subset=["fixed_drive"]).groupby(["game_id", "posteam", "fixed_drive"]).agg(
        min_yl=("yardline_100", "min"), result=("fixed_drive_result", "last"),
        top=("drive_time_of_possession", "last"), first_downs=("drive_first_downs", "last")).reset_index()
    d["rz"] = d["min_yl"] <= 20
    d["td"] = d["result"].eq("Touchdown")
    d["score"] = d["result"].isin(["Touchdown", "Field goal"])
    d["top_s"] = d["top"].map(_mmss)
    gd = d.groupby(["game_id", "posteam"])
    off["drives"] = gd.size()
    off["points_per_drive_proxy"] = (gd["td"].sum() * 7 + d[d["result"].eq("Field goal")].groupby(
        ["game_id", "posteam"]).size().reindex(gd.size().index, fill_value=0) * 3) / gd.size()
    off["scoring_drive_rate"] = gd["score"].mean()
    off["red_zone_trips"] = gd["rz"].sum()
    off["red_zone_td_rate"] = d[d["rz"]].groupby(["game_id", "posteam"])["td"].mean()
    off["time_of_possession_min"] = gd["top_s"].sum() / 60
    off = off.reset_index().rename(columns={"posteam": "team"})
    # defense = the opponent's offense in the same game
    opp = off.rename(columns={c: f"{c}_allowed" for c in off.columns if c not in ("game_id", "team")})
    opp = opp.rename(columns={"team": "opponent"})
    tr = _team_rows(sch)[["game_id", "season", "week", "team", "opponent", "is_home", "points_for",
                         "points_against", "result"]]
    df = tr.merge(off, on=["game_id", "team"], how="inner").merge(opp, on=["game_id", "opponent"], how="left")
    df["epa_per_play_net"] = df["epa_per_play"] - df["epa_per_play_allowed"]
    num = df.select_dtypes("number").columns.difference(["season", "week", "points_for", "points_against"])
    df[num] = df[num].round(4)
    return df.sort_values(["season", "week", "game_id", "is_home"]).reset_index(drop=True)


def build_quarters(pbp: pd.DataFrame, sch: pd.DataFrame):
    """points per quarter per team. Cumulative score at the end of each quarter from pbp; the LAST
    period is forced to final - previous so every game sums exactly to the official score."""
    p = pbp.sort_values(["game_id", "qtr", "order_sequence" if "order_sequence" in pbp else "play_id"])
    end = p.groupby(["game_id", "qtr"])[["total_home_score", "total_away_score"]].last().reset_index()
    fin = sch[sch["home_score"].notna()].set_index("game_id")[["home_score", "away_score", "home_team", "away_team"]]
    rows, fixed = [], 0
    for gid, e in end.groupby("game_id"):
        if gid not in fin.index:
            continue
        f = fin.loc[gid]
        e = e.sort_values("qtr")
        hc, ac = e["total_home_score"].tolist(), e["total_away_score"].tolist()
        if hc[-1] != f["home_score"] or ac[-1] != f["away_score"]:
            fixed += 1
        hc[-1], ac[-1] = f["home_score"], f["away_score"]
        ph, pa = np.diff([0] + hc), np.diff([0] + ac)
        for q, h, a in zip(e["qtr"], ph, pa):
            rows.append((gid, f["home_team"], f["away_team"], int(q), h, a))
            rows.append((gid, f["away_team"], f["home_team"], int(q), a, h))
    ql = pd.DataFrame(rows, columns=["game_id", "team", "opponent", "period", "points", "opp_points"])
    ql["period_label"] = ql["period"].map({1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}).fillna("OT")
    ql = ql.merge(sch[["game_id", "season", "week"]], on="game_id")
    ql = ql.groupby(["game_id", "season", "week", "team", "opponent", "period_label"], as_index=False)[
        ["points", "opp_points"]].sum()
    ql["period"] = ql["period_label"].map({"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "OT": 5})
    ql = ql.sort_values(["season", "week", "game_id", "team", "period"]).reset_index(drop=True)

    w = ql.pivot_table(index=["game_id", "season", "week", "team", "opponent"], columns="period_label",
                       values="points", aggfunc="sum").reset_index()
    wo = ql.pivot_table(index=["game_id", "team"], columns="period_label", values="opp_points",
                        aggfunc="sum").add_prefix("opp_").reset_index()
    w = w.merge(wo, on=["game_id", "team"])
    for c in ("Q1", "Q2", "Q3", "Q4", "OT", "opp_Q1", "opp_Q2", "opp_Q3", "opp_Q4", "opp_OT"):
        if c not in w:
            w[c] = 0
        w[c] = w[c].fillna(0).astype(int)
    w = w.rename(columns={c: c.lower() for c in w.columns if c.startswith(("Q", "opp_Q", "OT", "opp_OT"))})
    w["h1"], w["h2"] = w["q1"] + w["q2"], w["q3"] + w["q4"]
    w["opp_h1"], w["opp_h2"] = w["opp_q1"] + w["opp_q2"], w["opp_q3"] + w["opp_q4"]
    w["final"] = w["h1"] + w["h2"] + w["ot"]
    w["opp_final"] = w["opp_h1"] + w["opp_h2"] + w["opp_ot"]
    w["game_total_h1"], w["game_total_h2"] = w["h1"] + w["opp_h1"], w["h2"] + w["opp_h2"]
    w["higher_scoring_half"] = np.select([w["game_total_h1"] > w["game_total_h2"],
                                          w["game_total_h1"] < w["game_total_h2"]], ["H1", "H2"], "TIE")
    w["led_at_half"] = np.sign(w["h1"] - w["opp_h1"]).map({1: "led", 0: "tied", -1: "trailed"})
    w["won_game"] = w["final"] > w["opp_final"]
    lead = ["game_id", "season", "week", "team", "opponent", "q1", "q2", "q3", "q4", "ot", "h1", "h2", "final",
            "opp_q1", "opp_q2", "opp_q3", "opp_q4", "opp_ot", "opp_h1", "opp_h2", "opp_final"]
    w = w[lead + [c for c in w.columns if c not in lead]]
    print(f"  quarters: {w['game_id'].nunique()} games; {fixed} had pbp end-score != official (last period reconciled)")
    return ql, w.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


PROP_STATS = ["completions", "attempts", "passing_yards", "passing_tds", "passing_interceptions", "carries",
              "rushing_yards", "rushing_tds", "receptions", "targets", "receiving_yards", "receiving_tds",
              "def_sacks", "def_tackles_solo", "fantasy_points_ppr"]


def build_player_game_stats(pw: pd.DataFrame, sch: pd.DataFrame) -> pd.DataFrame:
    n0 = len(pw)
    pw = pw[pw["player_id"].notna()].copy()
    if len(pw) < n0:
        print(f"  player_game_stats: dropped {n0 - len(pw)} rows with no player_id (source gap)")
    pw = pw.rename(columns={"opponent_team": "opponent"})
    ctx = _team_rows(sch)[["game_id", "team", "gameday", "is_home", "points_for", "points_against", "result"]]
    ctx = ctx.rename(columns={"points_for": "team_score", "points_against": "opp_score", "result": "team_result"})
    df = pw.merge(ctx, on=["game_id", "team"], how="left")
    front = ["player_id", "player_display_name", "position", "team", "opponent", "season", "week",
             "season_type", "game_id", "gameday", "is_home", "team_score", "opp_score", "team_result"]
    df = df[front + [c for c in df.columns if c not in front]]
    df = df.sort_values(["player_id", "season", "week"])
    g = df.groupby(["player_id", "season"])
    df["games_played_before"] = g.cumcount()
    for c in PROP_STATS:   # pre-game form for props; shift(1) keeps it leak-free
        if c in df:
            df[f"{c}_season_avg_pre"] = g[c].transform(lambda s: s.shift(1).expanding().mean()).round(2)
            df[f"{c}_l3_avg_pre"] = g[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()).round(2)
    return df.sort_values(["season", "week", "team", "player_display_name"]).reset_index(drop=True)


def build_rosters(r: pd.DataFrame) -> pd.DataFrame:
    keep = ["season", "week", "game_type", "team", "gsis_id", "full_name", "position", "depth_chart_position",
            "jersey_number", "status", "status_description_abbr", "height", "weight", "birth_date", "college",
            "years_exp", "rookie_year", "draft_club", "draft_number", "espn_id", "pfr_id"]
    n0 = len(r)
    r = r.assign(gsis_id=r["gsis_id"].replace(r"^\s*$", np.nan, regex=True))   # source has '' ids
    r = r.dropna(subset=["season", "week", "game_type", "gsis_id"])[[c for c in keep if c in r.columns]]
    r = r.drop_duplicates(["season", "week", "game_type", "gsis_id"], keep="last")
    if len(r) < n0:
        print(f"  rosters: dropped {n0 - len(r)} rows (NULL key field or duplicate)")
    return r.sort_values(["season", "week", "team", "position", "full_name"]).reset_index(drop=True)


def build_depth_chart(d: pd.DataFrame) -> pd.DataFrame:
    """latest published depth chart per team (the full history is 600k+ snapshot rows)."""
    latest = d.groupby("team")["dt"].transform("max")
    d = d[d["dt"] == latest].drop_duplicates()
    keep = ["team", "pos_grp", "pos_name", "pos_abb", "pos_slot", "pos_rank", "player_name", "gsis_id", "espn_id", "dt"]
    return d[keep].rename(columns={"dt": "as_of"}).sort_values(["team", "pos_grp", "pos_slot", "pos_rank"]).reset_index(drop=True)


def build_ngs(n: pd.DataFrame) -> pd.DataFrame:
    n = n.copy()
    n["scope"] = np.where(n["week"] == 0, "season_total", "week")
    return n.sort_values(["season", "week", "team_abbr", "player_display_name"]).reset_index(drop=True)


# ---------------------------------------------------------------- betting
def _implied(ml):
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def _payout(ml):
    """profit per 1u stake at American odds"""
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, 100 / ml.abs(), ml / 100)


def build_betting_lines(sch: pd.DataFrame) -> pd.DataFrame:
    """per game. nflverse convention: spread_line = expected HOME margin (positive = home favored);
    in sportsbook notation the home handicap is -spread_line. Lines are closing consensus; upcoming
    games carry the current line and NULL results."""
    b = sch[["game_id", "season", "game_type", "week", "gameday", "away_team", "home_team", "away_score",
             "home_score", "result", "total", "spread_line", "total_line", "away_moneyline", "home_moneyline",
             "away_spread_odds", "home_spread_odds", "over_odds", "under_odds", "div_game", "roof",
             "surface", "temp", "wind", "status"]].copy()
    b = b[b["spread_line"].notna() | b["total_line"].notna() | b["home_moneyline"].notna()]
    b["home_handicap"] = -b["spread_line"]
    b["favorite"] = np.select([b["spread_line"] > 0, b["spread_line"] < 0], [b["home_team"], b["away_team"]], "PICK")
    ih, ia = _implied(b["home_moneyline"]), _implied(b["away_moneyline"])
    b["home_win_prob_novig"] = (ih / (ih + ia)).round(4)
    b["away_win_prob_novig"] = (ia / (ih + ia)).round(4)
    b["implied_home_points"] = ((b["total_line"] + b["spread_line"]) / 2).round(2)
    b["implied_away_points"] = ((b["total_line"] - b["spread_line"]) / 2).round(2)
    done = b["status"].eq("final")
    diff = b["result"] - b["spread_line"]                    # >0 home covered
    b["ats_winner"] = np.where(~done, None, np.select([diff > 0, diff < 0], [b["home_team"], b["away_team"]], "PUSH"))
    b["home_covered"] = np.where(~done, None, np.select([diff > 0, diff < 0], ["Y", "N"], "PUSH"))
    b["favorite_won"] = np.where(~done | b["favorite"].eq("PICK"), None,
                                 np.where(((b["result"] > 0) & (b["favorite"] == b["home_team"])) |
                                          ((b["result"] < 0) & (b["favorite"] == b["away_team"])), "Y", "N"))
    tdiff = b["total"] - b["total_line"]
    b["ou_result"] = np.where(~done, None, np.select([tdiff > 0, tdiff < 0], ["OVER", "UNDER"], "PUSH"))
    b["margin_vs_spread"] = np.where(done, diff, np.nan)
    b["total_vs_line"] = np.where(done, tdiff, np.nan)
    return b.sort_values(["season", "week", "gameday", "game_id"]).reset_index(drop=True)


def build_betting_team(bl: pd.DataFrame) -> pd.DataFrame:
    """one row per team per game — 'how did <team> do against the line'."""
    common = ["game_id", "season", "game_type", "week", "gameday", "total_line", "total", "ou_result", "status"]
    h = bl[common].copy()
    h["team"], h["opponent"], h["is_home"] = bl["home_team"], bl["away_team"], True
    h["points_for"], h["points_against"] = bl["home_score"], bl["away_score"]
    h["team_handicap"], h["moneyline"], h["spread_odds"] = -bl["spread_line"], bl["home_moneyline"], bl["home_spread_odds"]
    h["win_prob_novig"], h["implied_points"] = bl["home_win_prob_novig"], bl["implied_home_points"]
    a = bl[common].copy()
    a["team"], a["opponent"], a["is_home"] = bl["away_team"], bl["home_team"], False
    a["points_for"], a["points_against"] = bl["away_score"], bl["home_score"]
    a["team_handicap"], a["moneyline"], a["spread_odds"] = bl["spread_line"], bl["away_moneyline"], bl["away_spread_odds"]
    a["win_prob_novig"], a["implied_points"] = bl["away_win_prob_novig"], bl["implied_away_points"]
    t = pd.concat([h, a], ignore_index=True)
    t["role"] = np.select([t["team_handicap"] < 0, t["team_handicap"] > 0], ["favorite", "underdog"], "pick")
    done = t["status"].eq("final")
    t["margin"] = t["points_for"] - t["points_against"]
    cov = t["margin"] + t["team_handicap"]
    t["su_result"] = np.where(~done, None, np.select([t["margin"] > 0, t["margin"] < 0], ["W", "L"], "T"))
    t["ats_result"] = np.where(~done, None, np.select([cov > 0, cov < 0], ["W", "L"], "P"))
    t["cover_margin"] = np.where(done, cov, np.nan)
    pay = _payout(t["moneyline"])
    t["ml_units"] = np.where(~done | t["moneyline"].isna(), np.nan,
                             np.select([t["margin"] > 0, t["margin"] < 0], [pay, -1.0], 0.0)).round(3)
    t["team_total_vs_implied"] = np.where(done, t["points_for"] - t["implied_points"], np.nan)
    t = t.sort_values(["team", "season", "week"])
    g = t.groupby(["team", "season"])
    t["ats_cover_rate_pre"] = g["ats_result"].transform(
        lambda s: s.map({"W": 1, "L": 0}).shift(1).expanding().mean()).round(3)
    return t.sort_values(["season", "week", "game_id", "is_home"]).reset_index(drop=True)


def build_betting_summary(bt: pd.DataFrame) -> pd.DataFrame:
    d = bt[bt["status"].eq("final")]
    def agg(x):
        fav, dog = x[x["role"] == "favorite"], x[x["role"] == "underdog"]
        home, away = x[x["is_home"]], x[~x["is_home"]]
        cnt = lambda s, v: int((s == v).sum())  # noqa: E731
        decided = cnt(x["ats_result"], "W") + cnt(x["ats_result"], "L")
        ou = cnt(x["ou_result"], "OVER") + cnt(x["ou_result"], "UNDER")
        return pd.Series({
            "games": len(x), "su_w": cnt(x["su_result"], "W"), "su_l": cnt(x["su_result"], "L"),
            "ats_w": cnt(x["ats_result"], "W"), "ats_l": cnt(x["ats_result"], "L"), "ats_push": cnt(x["ats_result"], "P"),
            "ats_cover_pct": round(cnt(x["ats_result"], "W") / decided, 3) if decided else np.nan,
            "fav_games": len(fav), "fav_ats_w": cnt(fav["ats_result"], "W"), "fav_su_w": cnt(fav["su_result"], "W"),
            "dog_games": len(dog), "dog_ats_w": cnt(dog["ats_result"], "W"), "dog_su_w": cnt(dog["su_result"], "W"),
            "home_ats_w": cnt(home["ats_result"], "W"), "home_ats_l": cnt(home["ats_result"], "L"),
            "away_ats_w": cnt(away["ats_result"], "W"), "away_ats_l": cnt(away["ats_result"], "L"),
            "overs": cnt(x["ou_result"], "OVER"), "unders": cnt(x["ou_result"], "UNDER"),
            "over_pct": round(cnt(x["ou_result"], "OVER") / ou, 3) if ou else np.nan,
            "avg_cover_margin": round(x["cover_margin"].mean(), 2),
            "avg_points_vs_implied": round(x["team_total_vs_implied"].mean(), 2),
            "ml_units": round(x["ml_units"].sum(), 2),
        })
    out = d.groupby(["season", "team"]).apply(agg, include_groups=False).reset_index()
    ints = [c for c in out.columns if c not in ("season", "team") and not c.startswith(("ats_cover", "over_pct", "avg_", "ml_"))]
    out[ints] = out[ints].astype(int)
    return out


PBP_COLS = ["game_id", "play_id", "season", "week", "game_date", "home_team", "away_team", "posteam", "defteam",
            "qtr", "quarter_seconds_remaining", "game_seconds_remaining", "down", "ydstogo", "yardline_100",
            "fixed_drive", "fixed_drive_result", "play_type", "desc", "yards_gained", "pass", "rush", "shotgun",
            "no_huddle", "qb_scramble", "epa", "wpa", "wp", "success", "cpoe", "xpass", "pass_oe",
            "passer_player_name", "rusher_player_name", "receiver_player_name", "air_yards", "yards_after_catch",
            "complete_pass", "interception", "fumble_lost", "sack", "touchdown", "pass_touchdown",
            "rush_touchdown", "td_team", "field_goal_result", "kick_distance", "penalty", "penalty_team",
            "penalty_type", "penalty_yards", "posteam_score", "defteam_score", "total_home_score", "total_away_score"]


def build_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    p = pbp[[c for c in PBP_COLS if c in pbp.columns]]
    p = p[p["play_type"].notna()]                     # drop game-start / end-of-quarter marker rows
    return p.sort_values(["season", "week", "game_id", "play_id"]).reset_index(drop=True)


# ================================================================ checks
def sanity(sch, tgs, qw, bl) -> None:
    print("\n=== Sanity ===")
    done = sch[sch["status"] == "final"]
    print(f"  games final {len(done)} / scheduled {len(sch) - len(done)}; team-game rows {len(tgs):,} "
          f"(expect {2 * len(done)} once stats are posted)")
    j = tgs.merge(done[["game_id"]], on="game_id")
    miss = set(done["game_id"]) - set(tgs["game_id"])
    if miss:
        print(f"  ! {len(miss)} final games have no team stats yet (nflverse posts stats a few hours later): "
              f"{sorted(miss)[:5]}")
    bad = qw[qw["final"] != qw.merge(tgs[["game_id", "team", "points_for"]], on=["game_id", "team"], how="left")["points_for"]]
    print(f"  quarter sums == final score: {len(qw) - len(bad)}/{len(qw)}")
    f = bl[bl["status"] == "final"]
    print(f"  betting: {len(f)} graded games · favorites SU {(f['favorite_won'] == 'Y').mean():.1%} · "
          f"home covered {(f['home_covered'] == 'Y').mean():.1%} · overs {(f['ou_result'] == 'OVER').mean():.1%}"
          if len(f) else "  betting: no graded games")


# ================================================================ main
def main(seasons: list[int], out: str, with_pbp: bool) -> str:
    os.makedirs(out, exist_ok=True)
    prefix = f"nfl_{'_'.join(map(str, seasons))}"
    print(f"=== Downloading nflverse — seasons {seasons} ===")
    raw = {k: load(k, seasons) for k in FILES}
    if raw["schedule"].empty:
        sys.exit("schedule is empty — wrong season?")

    print("\n=== Building tables ===")
    teams = build_teams(raw["teams"], raw["schedule"])
    sch = build_schedule(raw["schedule"], teams)
    T = {"teams": teams, "schedule": sch}
    if not raw["team_week"].empty:
        T["team_game_stats"] = build_team_game_stats(raw["team_week"], sch)
        if not raw["team_reg"].empty:
            T["team_season_stats"] = build_team_season(raw["team_reg"], T["team_game_stats"])
    if not raw["pbp"].empty:
        T["team_advanced"] = build_team_advanced(raw["pbp"], sch)
        T["quarter_scores"], T["quarter_wide"] = build_quarters(raw["pbp"], sch)
        if with_pbp:
            T["pbp"] = build_pbp(raw["pbp"])
    if not raw["player_week"].empty:
        T["player_game_stats"] = build_player_game_stats(raw["player_week"], sch)
    if not raw["player_reg"].empty:
        ps = raw["player_reg"]
        if ps["player_id"].isna().any():
            print(f"  player_season_stats: dropped {int(ps['player_id'].isna().sum())} rows with no player_id (source gap)")
        T["player_season_stats"] = ps[ps["player_id"].notna()].sort_values(["season", "recent_team", "player_display_name"])
    if not raw["snaps"].empty:
        T["snap_counts"] = raw["snaps"].sort_values(["season", "week", "team", "player"])
    if not raw["injuries"].empty:
        T["injuries"] = raw["injuries"].sort_values(["season", "week", "team", "full_name"])
    if not raw["rosters"].empty:
        T["rosters"] = build_rosters(raw["rosters"])
    if not raw["depth"].empty:
        T["depth_chart"] = build_depth_chart(raw["depth"])
    for k, name in (("adv_pass", "adv_passing"), ("adv_rush", "adv_rushing"),
                    ("adv_rec", "adv_receiving"), ("adv_def", "adv_defense")):
        if not raw[k].empty:
            T[name] = raw[k].sort_values(["season", "week", "team", "pfr_player_name"])
    for k in ("ngs_passing", "ngs_rushing", "ngs_receiving"):
        if not raw[k].empty:
            T[k] = build_ngs(raw[k])
    T["betting_lines"] = build_betting_lines(sch)
    T["betting_team"] = build_betting_team(T["betting_lines"])
    T["betting_summary"] = build_betting_summary(T["betting_team"])

    if "team_game_stats" in T and "quarter_wide" in T:
        sanity(sch, T["team_game_stats"], T["quarter_wide"], T["betting_lines"])

    print(f"\n=== Writing CSVs to {out}/ ===")
    for name, df in T.items():
        path = f"{out}/{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:<22} {len(df):>8,} rows  {df.shape[1]:>4} cols  {os.path.getsize(path) / 1e6:6.1f} MB")
    return prefix


# ================================================================== keys + DuckDB + publish
# Keys are VERIFIED against the data before they are declared; a failed check is reported and
# the key skipped — that is the useful part (the join you assumed does not hold in THIS data).
PROLYT_URL = os.environ.get("PROLYT_URL") or os.environ.get("ANALYTIQ_URL") or "https://analytiq.dcentai.tech"
PROLYT_KEY = os.environ.get("PROLYT_API_KEY") or os.environ.get("ANALYTIQ_API_KEY", "")

PK_SPEC = {
    "teams":               ["team_abbr"],
    "schedule":            ["game_id"],
    "team_game_stats":     ["game_id", "team"],
    "team_season_stats":   ["season", "season_type", "team"],
    "team_advanced":       ["game_id", "team"],
    "quarter_scores":      ["game_id", "team", "period"],
    "quarter_wide":        ["game_id", "team"],
    "player_game_stats":   ["game_id", "player_id"],
    "player_season_stats": ["season", "season_type", "player_id"],
    "snap_counts":         ["game_id", "pfr_player_id"],
    "injuries":            ["season", "game_type", "week", "team", "gsis_id"],
    "rosters":             ["season", "game_type", "week", "gsis_id"],
    "adv_passing":         ["game_id", "pfr_player_id"],
    "adv_rushing":         ["game_id", "pfr_player_id"],
    "adv_receiving":       ["game_id", "pfr_player_id"],
    "adv_defense":         ["game_id", "pfr_player_id"],
    "ngs_passing":         ["season", "season_type", "week", "player_gsis_id"],
    "ngs_rushing":         ["season", "season_type", "week", "player_gsis_id"],
    "ngs_receiving":       ["season", "season_type", "week", "player_gsis_id"],
    "betting_lines":       ["game_id"],
    "betting_team":        ["game_id", "team"],
    "betting_summary":     ["season", "team"],
    "pbp":                 ["game_id", "play_id"],
}
FK_SPEC = [  # (child, child cols, parent, parent cols)
    ("schedule", ["home_team"], "teams", ["team_abbr"]),
    ("schedule", ["away_team"], "teams", ["team_abbr"]),
    ("team_game_stats", ["game_id"], "schedule", ["game_id"]),
    ("team_game_stats", ["team"], "teams", ["team_abbr"]),
    ("team_season_stats", ["team"], "teams", ["team_abbr"]),
    ("team_advanced", ["game_id", "team"], "team_game_stats", ["game_id", "team"]),
    ("quarter_scores", ["game_id", "team"], "quarter_wide", ["game_id", "team"]),
    ("quarter_wide", ["game_id"], "schedule", ["game_id"]),
    ("player_game_stats", ["game_id"], "schedule", ["game_id"]),
    ("player_game_stats", ["team"], "teams", ["team_abbr"]),
    ("snap_counts", ["game_id"], "schedule", ["game_id"]),
    ("injuries", ["team"], "teams", ["team_abbr"]),
    ("rosters", ["team"], "teams", ["team_abbr"]),
    ("depth_chart", ["team"], "teams", ["team_abbr"]),
    ("adv_passing", ["game_id"], "schedule", ["game_id"]),
    ("adv_rushing", ["game_id"], "schedule", ["game_id"]),
    ("adv_receiving", ["game_id"], "schedule", ["game_id"]),
    ("adv_defense", ["game_id"], "schedule", ["game_id"]),
    ("betting_lines", ["game_id"], "schedule", ["game_id"]),
    ("betting_team", ["game_id"], "betting_lines", ["game_id"]),
    ("betting_team", ["team"], "teams", ["team_abbr"]),
    ("betting_summary", ["team"], "teams", ["team_abbr"]),
    ("pbp", ["game_id"], "schedule", ["game_id"]),
]
INDEX_SPEC = {
    "schedule": ["season", "week", "home_team", "away_team", "gameday"],
    "team_game_stats": ["team", "season", "week"],
    "team_advanced": ["team", "season", "week"],
    "player_game_stats": ["player_display_name", "team", "season", "week", "position"],
    "player_season_stats": ["player_display_name", "recent_team"],
    "quarter_wide": ["team"], "quarter_scores": ["team", "period"],
    "betting_lines": ["season", "week", "home_team", "away_team"],
    "betting_team": ["team", "season", "week"], "betting_summary": ["team"],
    "snap_counts": ["player", "team"], "injuries": ["full_name", "team"], "rosters": ["full_name", "team"],
    "pbp": ["posteam", "defteam", "play_type"],
}


def _verify_pk(df, cols):
    absent = [c for c in cols if c not in df.columns]
    if absent:
        return False, f"columns absent: {absent}"
    sub = df[cols]
    if sub.isna().any().any():
        return False, f"{int(sub.isna().any(axis=1).sum()):,} rows have a NULL in the key"
    dup = int(len(sub) - len(sub.drop_duplicates()))
    return (False, f"{dup:,} duplicate key rows") if dup else (True, f"{len(sub):,} rows, all distinct")


def _verify_fk(child, ccols, parent, pcols):
    absent = [c for c in ccols if c not in child.columns] + [c for c in pcols if c not in parent.columns]
    if absent:
        return False, f"columns absent: {absent}"
    ck = child[ccols].dropna()
    if ck.empty:
        return False, "no non-null child keys"
    pk = parent[pcols].drop_duplicates()
    pk.columns = ccols
    m = ck.merge(pk, on=ccols, how="left", indicator=True)
    orphans = m[m["_merge"] == "left_only"]
    if len(orphans):
        ex = orphans[ccols].drop_duplicates().head(3).to_dict("records")
        return False, f"{len(orphans):,} child rows point at no parent, e.g. {ex}"
    return True, f"{len(ck):,} child rows all resolve"


def publish(path):
    """POST one file to /upload. overwrite=1 is REQUIRED unattended (no one to answer the collision dialog)."""
    try:
        with open(path, "rb") as fh:
            r = requests.post(f"{PROLYT_URL}/upload", headers={"X-API-Key": PROLYT_KEY},
                              files={"file": (os.path.basename(path), fh, "application/octet-stream")},
                              data={"overwrite": "1"}, timeout=600)
    except Exception as e:      # one file's network/TLS error must not abort the others
        print(f"  FAIL {os.path.basename(path):<46} {type(e).__name__}: {str(e)[:80]}")
        return False
    ok = r.status_code == 200
    print(f"  {'OK  ' if ok else 'FAIL'} {os.path.basename(path):<46} {r.status_code}  {'' if ok else r.text[:90]}")
    return ok


def ingest(out_dir: str, prefix: str) -> None:
    try:
        import duckdb
    except ImportError:
        print("\n! duckdb not installed — CSVs written, DuckDB step skipped. Publishing still runs.")
        duckdb = None
    frames, paths = {}, {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".csv") and f.startswith(prefix + "_"):
            short = f[len(prefix) + 1:-4]
            frames[short] = pd.read_csv(f"{out_dir}/{f}", low_memory=False)
            paths[short] = f"{out_dir}/{f}"

    print("\n=== Verifying keys before declaring them ===")
    good_pk, bad_pk, good_fk, bad_fk = {}, {}, [], []
    for t, cols in PK_SPEC.items():
        if t in frames:
            ok, why = _verify_pk(frames[t], cols)
            (good_pk if ok else bad_pk)[t] = (cols, why)
            print(f"  {'PK OK' if ok else 'PK NO'} {t:<20} {'+'.join(cols):<40} {why}")
    for c, cc, p, pc in FK_SPEC:
        if c not in frames or p not in frames:
            continue
        if p not in good_pk:
            bad_fk.append((c, cc, p, pc, "parent PK not verified")); print(f"  FK NO {c} -> {p}: parent PK not verified"); continue
        ok, why = _verify_fk(frames[c], cc, frames[p], pc)
        (good_fk.append((c, cc, p, pc)) if ok else bad_fk.append((c, cc, p, pc, why)))
        print(f"  {'FK OK' if ok else 'FK NO'} {c}.{'+'.join(cc)} -> {p:<16} {why}")

    db_path, rel_path = f"{out_dir}/{prefix}.duckdb", f"{out_dir}/{prefix}_relationships.json"
    if duckdb is not None:
        if os.path.exists(db_path):
            os.remove(db_path)
        con = duckdb.connect(db_path)
        q = lambda cs: ", ".join('"' + c + '"' for c in cs)  # noqa: E731
        order, seen = [], set()
        def visit(t):               # parents before children so FKs can be declared
            if t in seen or t not in frames:
                return
            seen.add(t)
            for c, _cc, p, _pc in good_fk:
                if c == t and p != t:
                    visit(p)
            order.append(t)
        for t in frames:
            visit(t)
        n_con = n_idx = 0
        for t in order:
            df = frames[t]
            con.register("_src", df)
            pk = good_pk.get(t)
            fks = [(cc, p, pc) for (c, cc, p, pc) in good_fk if c == t]
            try:
                if pk or fks:
                    defs = [f'"{n}" {ty}' for n, ty, *_ in con.execute("DESCRIBE _src").fetchall()]
                    if pk:
                        defs.append(f"PRIMARY KEY ({q(pk[0])})")
                    # DuckDB FKs must reference a PK/UNIQUE — only declare those that do
                    defs += [f'FOREIGN KEY ({q(cc)}) REFERENCES "{prefix}_{p}" ({q(pc)})'
                             for cc, p, pc in fks if good_pk.get(p, ([None],))[0] == pc]
                    con.execute(f'CREATE TABLE "{prefix}_{t}" ({", ".join(defs)})')
                    con.execute(f'INSERT INTO "{prefix}_{t}" SELECT * FROM _src')
                    n_con += 1
                else:
                    con.execute(f'CREATE TABLE "{prefix}_{t}" AS SELECT * FROM _src')
            except Exception as e:
                print(f"  ! {t}: constraints rejected ({str(e)[:90]}) — written without them")
                con.execute(f'DROP TABLE IF EXISTS "{prefix}_{t}"')
                con.execute(f'CREATE TABLE "{prefix}_{t}" AS SELECT * FROM _src')
            con.unregister("_src")
            for col in INDEX_SPEC.get(t, []):
                if col in df.columns:
                    try:
                        idx = f"idx_{t}_{col}"[:60]
                        con.execute(f'CREATE INDEX "{idx}" ON "{prefix}_{t}" ("{col}")')
                        n_idx += 1
                    except Exception:
                        pass
        con.close()
        print(f"\n=== DuckDB written === {db_path}  {os.path.getsize(db_path) / 1e6:.1f} MB · "
              f"{len(frames)} tables · {n_con} constrained · {n_idx} indexes")

    with open(rel_path, "w", encoding="utf-8") as f:
        json.dump({"_about": "NFL relationships VERIFIED against the data by this run. one_table is the ONE side: "
                             "summing one of ITS columns across a join to many_table double-counts it. "
                             "betting_lines.spread_line = expected HOME margin (positive = home favored); "
                             "the sportsbook home handicap is -spread_line. team_handicap in betting_team is "
                             "already in sportsbook notation (negative = favorite).",
                   "generated": pd.Timestamp.now(tz="UTC").isoformat(),
                   "primary_keys": {f"{prefix}_{t}": cols for t, (cols, _) in good_pk.items()},
                   "relationships": [{"one_table": f"{prefix}_{p}", "one_key": pc, "many_table": f"{prefix}_{c}",
                                      "many_key": cc, "cardinality": "1:N", "evidence": "verified-at-build"}
                                     for (c, cc, p, pc) in good_fk],
                   "skipped": ([{"table": t, "key": cols, "reason": why} for t, (cols, why) in bad_pk.items()]
                               + [{"child": c, "parent": p, "reason": w} for (c, _cc, p, _pc, w) in bad_fk])},
                  f, indent=2, default=str)

    if not PROLYT_KEY:
        print("\n=== Publish skipped === PROLYT_API_KEY / ANALYTIQ_API_KEY not set — files written locally only.")
        return
    print(f"\n=== Publishing to {PROLYT_URL} ===")
    sent = sum(publish(p) for p in paths.values())
    publish(rel_path)
    print(f"  {sent} of {len(paths)} tables published."
          + ("" if sent == len(paths) else "  Failures are usually the key (401), size (413), or the host being down."))
    if sent < len(paths):
        sys.exit(1)     # make the Actions run RED, not a silent no-op


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 6 else today.year - 1   # Jan-May still belongs to last season


if __name__ == "__main__":
    out = os.environ.get("NFL_OUT", "./data")
    seasons = [int(s) for s in os.environ.get("NFL_SEASONS", "").split()] or [current_season()]
    with_pbp = os.environ.get("NFL_PBP", "1") != "0"
    prefix = main(seasons, out, with_pbp)
    ingest(out, prefix)
