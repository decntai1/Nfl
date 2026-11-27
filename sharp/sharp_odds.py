#!/usr/bin/env python3
"""
sharp_odds.py — sharp CONSENSUS prices through The Odds API, compared with vegas.hu, logged and graded.

There is no single sharp book for player props (Pinnacle and Circa post few). The fair price is a consensus:
each book's margin removed on the exact same line, then a weighted median across books that set their own numbers.

Lives in sharp/ of the Nfl repo; pulls land in sharp/pulls/ (the workflow .github/workflows/sharp-odds.yml runs here).

    python sharp_odds.py pull                                 main lines, every NFL game -> pulls/  (~3 credits)
    python sharp_odds.py pull --props                         + lean props set                  (~10 credits a game)
    python sharp_odds.py pull --props --set full              + every prop incl. TDs, all alts  (~23 credits a game)
    python sharp_odds.py pull --props --only CIN WAS          props only for games naming CIN or WAS
    python sharp_odds.py compare pulls/nfl_sharp_<stamp>.csv.gz --priced priced_week12/priced_*.csv
    python sharp_odds.py report                               graded record of every logged edge (needs the ledger)

Key: ODDS_API_KEY in the environment (GitHub: repo secret ODDS_API_KEY). Never write the key into a file.

Books (10 = billed as ONE region; an 11th would double the cost):
  pinnacle, betfair_ex_eu                 sharp main lines
  fanduel, williamhill_us (Caesars)       make their own prop numbers; the two best prop indicators in the US
  draftkings, betmgm                      same prop source -> counted together as ONE vote
  betonlineag                             early offshore props (many offshore books share its tool: one vote anyway)
  novig, prophetx                         exchanges, near-zero margin
  kalshi                                  prediction market; thin on depth players -> low weight
A book missing from your plan or a market simply returns nothing; nothing is guessed.

Margin removal:
  both sides on the line  -> odds-ratio de-margin of the pair (exact).
  one side only (alt ladders, anytime TD "Yes")  -> 1/price divided by that book's median overround on its own
                                                    two-sided lines of the same market in this pull ("one-sided", *).
Consensus = weighted median over independent sources; shown only with >= MIN_SRC sources.
edge = vegas_odds x consensus_p - 1. Whole-number lines (pushes) are left out.

Cost (the-odds-api.com): a request costs markets returned x regions. Main lines for every game = 3 in one request.
Props come one game per request: lean set ~10 credits a game (~130 for a Sunday), full set ~23 (~300).
Free plan (500 a month): lean Sundays, ~3 a month. $30 plan (20,000): anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

API = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
BOOKS = "pinnacle,betfair_ex_eu,fanduel,williamhill_us,draftkings,betmgm,betonlineag,novig,prophetx,kalshi"
GROUP = {"draftkings": "dk_mgm", "betmgm": "dk_mgm"}
WEIGHT = {"pinnacle": 1.0, "fanduel": 1.0, "williamhill_us": 1.0, "betfair_ex_eu": 0.8, "dk_mgm": 0.7,
          "novig": 0.7, "prophetx": 0.6, "betonlineag": 0.6, "kalshi": 0.4}
MIN_SRC = 3
MAIN = "h2h,spreads,totals"
PROP_BASE = ["player_pass_yds", "player_pass_tds", "player_pass_completions", "player_pass_attempts",
             "player_pass_interceptions", "player_rush_yds", "player_rush_tds", "player_receptions",
             "player_reception_yds", "player_anytime_td", "player_tds_over"]
PROP_ALT = ["player_pass_yds_alternate", "player_rush_yds_alternate", "player_reception_yds_alternate",
            "player_receptions_alternate", "player_pass_tds_alternate", "player_pass_completions_alternate",
            "player_pass_attempts_alternate", "player_rush_tds_alternate"]
GAME_EXTRA = ["alternate_spreads", "alternate_totals", "team_totals", "alternate_team_totals"]
# lean (default): only the markets the slip rules play (book 60-88% legs: yards / catches ladders, game + team
# lines). No anytime TD, 2+ TDs, pass/rush TDs, INTs, completions, attempts. 10 markets = ~10 credits a game.
LEAN = ["player_pass_yds", "player_rush_yds", "player_rush_yds_alternate", "player_reception_yds",
        "player_reception_yds_alternate", "player_receptions", "player_receptions_alternate",
        "alternate_spreads", "alternate_totals", "team_totals"]
NICK = {"Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF", "Panthers": "CAR", "Bears": "CHI",
        "Bengals": "CIN", "Browns": "CLE", "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
        "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC", "Raiders": "LV", "Chargers": "LAC",
        "Rams": "LA", "Dolphins": "MIA", "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
        "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "49ers": "SF", "Seahawks": "SEA", "Buccaneers": "TB",
        "Titans": "TEN", "Commanders": "WAS"}
VEGAS_CODE = {"ARZ": "ARI", "NOS": "NO", "LVR": "LV", "PS": "PIT", "LAR": "LA", "WSH": "WAS", "JAC": "JAX"}
# vegas.hu prop type (nfl_odds_pricer.classify) -> The Odds API base market
PROP_OF = {"passing_yards": "player_pass_yds", "rushing_yards": "player_rush_yds",
           "receiving_yards": "player_reception_yds", "receptions": "player_receptions",
           "passing_tds": "player_pass_tds", "completions": "player_pass_completions",
           "attempts": "player_pass_attempts", "passing_interceptions": "player_pass_interceptions",
           "rush_td": "player_rush_tds"}
OUT = os.environ.get("NFL_OUT", "./data")
LOG = os.path.join(OUT, "sharp_value_log.csv")


def code(full) -> str | None:
    for nick, c in NICK.items():
        if str(full).endswith(nick):
            return c
    return None


def pkey(name) -> str:
    x = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    x = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", x)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", x)).strip()


# ================================================================== pull
class OutOfCredits(Exception):
    pass


LEFT = {"n": None}            # credits left, from the last response header
LAST = {"status": None, "text": "", "spent": 0}   # last response + credits spent this run
NOTES: list[str] = []         # every rejected / empty request, shown on the GitHub run page


def get(path: str, **params):
    """JSON, or None for a market/event the API won't serve. Raises OutOfCredits when the quota is gone."""
    import time
    import requests
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("set ODDS_API_KEY")
    label = path.split("/")[-2] if "/events/" in path else path.split("/")[-1]
    for attempt in range(3):
        try:
            r = requests.get(f"{API}/{path}", params={"apiKey": key, **params}, timeout=40)
        except requests.RequestException as e:
            print(f"  {label}: network error {e}; retry")
            time.sleep(3 * (attempt + 1))
            continue
        left, used = r.headers.get("x-requests-remaining"), r.headers.get("x-requests-last")
        if left is not None:
            try:
                LEFT["n"] = float(left)
            except ValueError:
                pass
        try:
            LAST["spent"] += int(float(used or 0))
        except ValueError:
            pass
        LAST["status"], LAST["text"] = r.status_code, r.text[:300]
        if r.status_code == 200:
            print(f"  {label:<34} cost {used:>3}  credits left {left}")
            return r.json()
        txt = r.text[:200]
        if r.status_code == 429 and "QUOTA" not in txt.upper() and "CREDIT" not in txt.upper():
            time.sleep(3 * (attempt + 1))         # too many requests per second: wait and retry
            continue
        if r.status_code in (401, 402, 429) and any(w in txt.upper() for w in ("QUOTA", "CREDIT", "USAGE")):
            raise OutOfCredits(txt)
        if r.status_code == 401:
            sys.exit(f"{label}: HTTP 401 {txt}  (wrong or missing key?)")
        msg = f"{label} {params.get('markets', '')[:60]}: HTTP {r.status_code} {txt}"
        print(f"::warning::{msg}")                          # 422 = a market not offered; others: skip
        NOTES.append(msg)
        return None
    print(f"  {label}: gave up after 3 tries — skipped")
    return None


def flatten(events: list, stamp: str) -> list[dict]:
    rows = []
    for ev in events:
        home, away = code(ev["home_team"]), code(ev["away_team"])
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    rows.append({"captured": stamp, "event_id": ev["id"], "commence": ev["commence_time"],
                                 "home": home, "away": away, "book": bk["key"], "market": mk["key"],
                                 "name": o.get("name"), "team": code(o.get("name")) or code(o.get("description", "")),
                                 "player": o.get("description"), "point": o.get("point"), "price": o.get("price"),
                                 "updated": mk.get("last_update")})
    return rows


def pull(a):
    """Saves whatever was pulled, always. Out of credits mid-way -> keeps the games already paid for, stops cleanly.
    Props go nearest kickoff first, so a short quota covers the games you are about to bet."""
    os.makedirs(a.out, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M")
    rows, stopped = [], ""
    try:
        ev = get(f"sports/{SPORT}/odds", regions="eu", markets=MAIN, bookmakers=BOOKS, oddsFormat="decimal") or []
    except OutOfCredits as e:
        print(f"::error::out of Odds API credits before the first request: {e}")
        sys.exit(1)
    rows += flatten(ev, stamp)
    if a.props:
        mk = ",".join(LEAN if a.set == "lean" else PROP_BASE + ([] if a.no_alt else PROP_ALT) + GAME_EXTRA)
        n_mk = len(mk.split(","))
        print(f"  market set: {a.set} ({n_mk} markets a game, costs up to {n_mk} credits a game)")
        todo = [e for e in sorted(ev, key=lambda e: e["commence_time"])
                if (not a.only or any(t in (code(e["home_team"]), code(e["away_team"])) for t in a.only))
                and pd.Timestamp(e["commence_time"]) > pd.Timestamp.now(tz="UTC")
                and (not a.hours or pd.Timestamp(e["commence_time"]) <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=a.hours))]
        for i, e in enumerate(todo):
            if LEFT["n"] is not None and LEFT["n"] < n_mk + a.reserve:
                stopped = (f"stopped before {e['away_team']} @ {e['home_team']}: {LEFT['n']:.0f} credits left, "
                           f"a game costs up to {n_mk}, reserve {a.reserve}; {len(todo) - i} game(s) not pulled")
                break
            game = f"{code(e['away_team'])}@{code(e['home_team'])}"
            url = f"sports/{SPORT}/events/{e['id']}/odds"
            try:
                x = get(url, regions="eu", markets=mk, bookmakers=BOOKS, oddsFormat="decimal")
                if x is None and LAST["status"] == 422:
                    # one bad market rejects the whole request: retry market by market (a rejected one costs nothing)
                    NOTES.append(f"{game}: whole request rejected, retrying one market at a time")
                    for m1 in mk.split(","):
                        y = get(url, regions="eu", markets=m1, bookmakers=BOOKS, oddsFormat="decimal")
                        if y:
                            rows += flatten([y], stamp)
                    x = None
            except OutOfCredits as err:
                stopped = f"out of credits at {game} ({err}); {len(todo) - i} game(s) not pulled"
                break
            if x:
                got = flatten([x], stamp)
                if not got:
                    NOTES.append(f"{game}: no prop prices posted yet by these books (cost 0)")
                rows += got
    df = pd.DataFrame(rows)
    summary(df, a, stopped)
    if df.empty:
        print("::warning::nothing pulled (no games listed?)")
        return
    path = os.path.join(a.out, f"nfl_sharp_{stamp}.csv.gz")
    df.to_csv(path, index=False)
    print(f"{len(df):,} prices, {df['event_id'].nunique()} games, books {sorted(df['book'].unique())} -> {path}")
    if stopped:
        print(f"::warning::{stopped}. Saved what was pulled.")


def summary(df: pd.DataFrame, a, stopped: str):
    """what came back, per game and market, + credits: printed, and on the GitHub run page when run there."""
    lines = [f"## Sharp odds pull — {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC",
             f"**Credits spent this run: {LAST['spent']}** · credits left: {LEFT['n']:.0f}" if LEFT["n"] is not None
             else f"**Credits spent this run: {LAST['spent']}**", ""]
    if len(df):
        df = df.assign(game=df["away"].astype(str) + "@" + df["home"].astype(str))
        props = df[~df["market"].isin(["h2h", "h2h_lay", "spreads", "totals"])]
        lines.append(f"Main lines: {df[df.market.isin(['h2h', 'spreads', 'totals'])]['event_id'].nunique()} games, "
                     f"books: {', '.join(sorted(df['book'].unique()))}")
        if len(props):
            lines += ["", "| Game | Market | Books | Prices |", "|---|---|---|---|"]
            for (g, m), x in props.groupby(["game", "market"]):
                lines.append(f"| {g} | {m} | {x['book'].nunique()} ({', '.join(sorted(x['book'].unique()))}) | {len(x)} |")
        elif a.props:
            lines.append("**No prop prices came back** — see notes.")
    if NOTES or stopped:
        lines += ["", "**Notes**"] + [f"- {n}" for n in NOTES] + ([f"- {stopped}"] if stopped else [])
    text = "\n".join(lines)
    print(text)
    p = os.environ.get("GITHUB_STEP_SUMMARY")
    if p:
        with open(p, "a") as f:
            f.write(text + "\n")


# ================================================================== fair p per book, then consensus
def _or_demargin(q: np.ndarray) -> np.ndarray:
    lo, hi = -10.0, 10.0                                # odds-ratio: p_i = q_i / (q_i + k(1-q_i)), sum p = 1
    for _ in range(60):
        k = math.exp((lo + hi) / 2)
        s = (q / (q + k * (1 - q))).sum()
        lo, hi = ((lo + hi) / 2, hi) if s > 1 else (lo, (lo + hi) / 2)
    return q / (q + k * (1 - q))


def fair(df: pd.DataFrame) -> pd.DataFrame:
    """one row per (event, book, market, subject, point, side) with that book's margin-free p."""
    d = df[df["price"].notna() & (df["price"] > 1)].copy()
    d["base"] = d["market"].str.replace("_alternate", "", regex=False).replace(
        {"alternate_spreads": "spreads", "alternate_totals": "totals", "alternate_team_totals": "team_totals"})
    d["subj"] = np.where(d["base"].str.startswith("player_"), d["player"].map(pkey),
                         np.where(d["base"] == "team_totals", d["team"], ""))
    d["side"] = np.where(d["base"].isin(["h2h", "spreads"]), d["team"], d["name"].astype(str).str.title())
    d["pt"] = np.where(d["base"] == "spreads", d["point"], d["point"].abs())
    d["pair"] = d["point"].abs()
    d = d.sort_values("captured").drop_duplicates(["event_id", "book", "base", "subj", "pt", "side"], keep="last")
    d["q"] = 1 / d["price"]
    out = []
    for _, g in d.groupby(["event_id", "book", "base", "subj", "pair"], dropna=False):
        if len(g) == 2 and g["side"].nunique() == 2:
            out.append(g.assign(p=_or_demargin(g["q"].to_numpy()), method="pair", over=g["q"].sum() - 1))
        else:
            out.append(g.assign(p=np.nan, method="one-sided", over=np.nan))
    f = pd.concat(out, ignore_index=True) if out else d.assign(p=np.nan, method="", over=np.nan)
    med = f[f["method"] == "pair"].groupby(["book", "base"])["over"].median()
    med_book = f[f["method"] == "pair"].groupby("book")["over"].median()
    one = f["method"] == "one-sided"
    m = [med.get((b, x), med_book.get(b, 0.05)) for b, x in zip(f.loc[one, "book"], f.loc[one, "base"])]
    f.loc[one, "p"] = (f.loc[one, "q"] / (1 + np.array(m, dtype=float))).clip(0.001, 0.999)
    return f


def wmedian(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, c[-1] / 2)])


def consensus(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    f["src"] = f["book"].map(lambda b: GROUP.get(b, b))
    k7 = ["event_id", "home", "away", "base", "subj", "pt", "side"]
    g = f.groupby(k7 + ["src"], dropna=False).agg(p=("p", "mean"), method=("method", "first")).reset_index()
    g["w"] = g["src"].map(WEIGHT).fillna(0.5) * np.where(g["method"] == "pair", 1.0, 0.6)
    rows = []
    for k, x in g.groupby(k7, dropna=False):
        rows.append(dict(zip(k7, k), cons_p=wmedian(x["p"].to_numpy(), x["w"].to_numpy()), n_src=x["src"].nunique(),
                         spread_p=float(x["p"].max() - x["p"].min()),
                         sources=" ".join(f"{s}:{p:.2f}{'' if m == 'pair' else '*'}" for s, p, m in
                                          zip(x["src"], x["p"], x["method"]))))
    return pd.DataFrame(rows)


# ================================================================== vegas.hu rows -> the same keys
def num(s):
    m = re.search(r"([+-]?\d+(?:,\d)?)", str(s).replace(" ", ""))
    return float(m.group(1).replace(",", ".")) if m else None


def vegas_keys(p: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    try:
        from nfl_odds_pricer import classify
    except ImportError:
        classify = None
        print("  (nfl_odds_pricer.py not found: player props skipped, game lines only)")
    rows = []
    for r in p.itertuples():
        m, s = str(r.market), str(r.selection)
        subj = VEGAS_CODE.get(str(r.subject), str(r.subject))
        if m.startswith("Győztes (hossz"):
            rows.append((r.Index, "h2h", "", np.nan, subj))
        elif m.startswith("Hendikep (hossz"):
            rows.append((r.Index, "spreads", "", num(s.split()[-1]), subj))
        elif m.startswith("Pontok száma (hossz"):
            rows.append((r.Index, "totals", "", num(s), "Over" if "felett" in s else "Under"))
        elif re.match(r"^[A-Z]{2,3} \S+\s+pontok száma \(hossz", m):
            t = m[:3].strip()
            rows.append((r.Index, "team_totals", VEGAS_CODE.get(t, t), num(s), "Over" if "felett" in s else "Under"))
        elif classify is not None and r.type in set(PROP_OF) | {"anytime"} and "védelmi" not in m:
            sp = classify(m, s, {home: home, away: away})
            if not sp.get("player"):
                continue
            if sp["type"] == "anytime":
                n = sp.get("n", 1) or 1
                if n == 1 and sp.get("side", "over") == "over":
                    rows.append((r.Index, "player_anytime_td", pkey(sp["player"]), np.nan, "Yes"))
                elif sp.get("line") is not None:
                    rows.append((r.Index, "player_tds_over", pkey(sp["player"]), sp["line"],
                                 "Over" if sp["side"] == "over" else "Under"))
            elif sp.get("line") is not None and sp.get("side"):
                rows.append((r.Index, PROP_OF[sp["type"]], pkey(sp["player"]), sp["line"],
                             "Over" if sp["side"] == "over" else "Under"))
    k = pd.DataFrame(rows, columns=["idx", "base", "subj", "pt", "side"]).set_index("idx")
    k = k[k["pt"].isna() | (k["pt"] % 1 != 0)]          # whole-number lines can push: left out, not guessed
    return p.join(k, how="inner")


def match_player(subj: str, names: set) -> str | None:
    if subj in names:
        return subj
    t = subj.split()
    if not t:
        return None
    c = [n for n in names if n.split()[-1:] == t[-1:] and n.split()[:1] == t[:1]]
    if len(c) == 1:
        return c[0]
    c = [n for n in names if n.split()[:1] == t[:1] and t[-1] in n.split()]    # "jacory merritt" ~ "jacory croskey merritt"
    if len(c) == 1:
        return c[0]
    import difflib
    m = difflib.get_close_matches(subj, list(names), n=1, cutoff=0.85)
    return m[0] if m else None


def compare(a):
    raw = pd.concat([pd.read_csv(f) for f in a.sharp], ignore_index=True)
    C = consensus(fair(raw))
    stamp = str(raw["captured"].max())
    out = []
    for f in a.priced:
        p = pd.read_csv(f, low_memory=False)
        if p.empty:
            continue
        gid = p["game_id"].iloc[0]
        away, home = gid.split("_")[2:4]
        V = vegas_keys(p, home, away)
        g = C[((C["home"] == home) & (C["away"] == away)) | ((C["home"] == away) & (C["away"] == home))]
        if g.empty:
            print(f"  {away}@{home}: not in the sharp pull")
            continue
        names = set(g.loc[g["base"].str.startswith("player_"), "subj"])
        idx = {(b, s, None if pd.isna(t) else round(float(t), 1), sd): r
               for b, s, t, sd, r in zip(g["base"], g["subj"], g["pt"], g["side"], g.itertuples())}
        for r in V.itertuples():
            subj = r.subj
            if r.base.startswith("player_"):
                subj = match_player(r.subj, names)
                if subj is None:
                    continue
            c = idx.get((r.base, subj, None if pd.isna(r.pt) else round(float(r.pt), 1), r.side))
            if c is None or c.n_src < a.min_src:
                continue
            push = getattr(r, "p_push", 0)
            pb = r.p_book / (1 - (push if pd.notna(push) else 0)) if pd.notna(r.p_book) else np.nan
            out.append({"captured": stamp, "game_id": gid, "market": r.market, "selection": r.selection,
                        "vegas_odds": r.odds, "vegas_p_book": round(pb, 3) if pd.notna(pb) else None,
                        "cons_p": round(c.cons_p, 3), "n_src": c.n_src, "spread_p": round(c.spread_p, 3),
                        "edge": round(r.odds * c.cons_p - 1, 3), "sources": c.sources})
    R = pd.DataFrame(out)
    if R.empty:
        sys.exit("nothing matched — different games, or no line at vegas.hu's points")
    R = R.sort_values("edge", ascending=False)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    R.to_csv(a.out, index=False)
    os.makedirs(OUT, exist_ok=True)
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if len(old):
        old = old[~(old["captured"].astype(str).eq(stamp) & old["game_id"].isin(R["game_id"]))]
    pd.concat([old, R], ignore_index=True).to_csv(LOG, index=False)
    pd.options.display.width = 250
    pd.options.display.max_colwidth = 60
    print(f"{len(R):,} vegas.hu prices matched to a {a.min_src}+ source consensus; "
          f"{int((R.edge > 0).sum())} pay more than it; logged -> {LOG}")
    print(f"shown: consensus p {a.band[0]:.0%}-{a.band[1]:.0%}, edge > {a.min_edge:+.0%}")
    print(R[(R.edge > a.min_edge) & R.cons_p.between(*a.band)][["game_id", "selection", "vegas_odds", "vegas_p_book", "cons_p", "n_src",
                                  "spread_p", "edge", "sources"]].head(50).to_string(index=False))
    print(f"\nmean |vegas.hu p - consensus p|: {(R.vegas_p_book - R.cons_p).abs().mean():.3f}   "
          f"(* = one-sided price, margin removed with that book's usual overround)")


# ================================================================== report: did the edges win?
def report(_a):
    if not os.path.exists(LOG):
        sys.exit("no log yet")
    log = pd.read_csv(LOG)
    L = pd.read_csv(os.path.join(OUT, "nfl_odds_ledger.csv"), low_memory=False)
    nm = lambda x: re.sub(r"\s+", " ", str(x).replace("(", "").replace(")", "")).strip()  # noqa: E731
    res = {(g, nm(m), nm(s)): r for g, m, s, r in zip(L["game_id"], L["market"], L["selection"], L["result"])}
    log["result"] = [res.get((g, nm(m), nm(s)), "open") for g, m, s in zip(log["game_id"], log["market"], log["selection"])]
    x = log[log["result"].isin(["won", "lost"])].sort_values("captured").drop_duplicates(
        ["game_id", "market", "selection"], keep="last")          # the last capture before the game
    if x.empty:
        print("nothing graded yet")
        return
    x["y"] = (x["result"] == "won").astype(float)
    x["profit"] = np.where(x["y"] == 1, x["vegas_odds"] - 1, -1)
    print(f"{len(x):,} graded selections, {x.game_id.nunique()} games")
    print(f"Brier: consensus {((x.cons_p - x.y) ** 2).mean():.4f}   vegas.hu {((x.vegas_p_book - x.y) ** 2).mean():.4f}")
    for lo, hi in ((-1, -0.05), (-0.05, 0), (0, 0.03), (0.03, 0.06), (0.06, 1)):
        z = x[(x.edge > lo) & (x.edge <= hi)]
        if len(z):
            print(f"  edge {lo:+.0%}..{hi:+.0%}  n {len(z):>5}  won {z.y.mean():.1%}  consensus said {z.cons_p.mean():.1%}"
                  f"  return at vegas.hu odds {z.profit.mean():+.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("pull")
    p1.add_argument("--out", default="pulls")
    p1.add_argument("--props", action="store_true")
    p1.add_argument("--set", choices=["lean", "full"], default="lean",
                    help="lean (default): 10 slip-rule markets, ~10 credits a game; full: 23 markets incl. TDs, ~23")
    p1.add_argument("--no-alt", action="store_true", help="full set without alternate lines")
    p1.add_argument("--only", nargs="*", default=[], help="team codes, props only for these games")
    p1.add_argument("--hours", type=float, default=0, help="props only for games starting within N hours (0 = all)")
    p1.add_argument("--reserve", type=int, default=5, help="stop props when fewer than (a game's cost + this) credits are left")
    p2 = sub.add_parser("compare")
    p2.add_argument("sharp", nargs="+", help="one or more nfl_sharp_*.csv(.gz); later captures win on duplicates")
    p2.add_argument("--priced", nargs="+", required=True)
    p2.add_argument("--out", default="value.csv")
    p2.add_argument("--min-edge", type=float, default=0.0)
    p2.add_argument("--band", nargs=2, type=float, default=[0.60, 0.88], metavar=("LO", "HI"),
                    help="show only legs whose consensus p is in this band (slip rule); all matches are still logged")
    p2.add_argument("--min-src", type=int, default=MIN_SRC)
    sub.add_parser("report")
    a = ap.parse_args()
    {"pull": pull, "compare": compare, "report": report}[a.cmd](a)
