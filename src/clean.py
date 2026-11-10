"""Clean CFBD plays into offense snaps with success flags and opponent-quality flags.
Data source: CollegeFootballData.com (CFBD) API."""
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RAW, PROC, ensure_dirs
ensure_dirs()
YEARS = [2019, 2021, 2022, 2023, 2024, 2025]

OFFENSE_TYPES = {"Rush", "Pass Reception", "Pass Incompletion", "Sack", "Passing Touchdown",
    "Rushing Touchdown", "Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Own)", "Fumble Recovery (Opponent)", "Fumble Return Touchdown", "Safety"}
AUTO_FAIL = {"Pass Interception Return", "Interception", "Interception Return Touchdown",
    "Fumble Recovery (Opponent)", "Fumble Return Touchdown", "Safety"}
TD_TYPES = {"Passing Touchdown", "Rushing Touchdown"}
KNEEL = re.compile(r"\b(kneel|knelt|takes a knee)\b", re.I)
SPIKE = re.compile(r"\b(spike|spiked the ball|spikes the ball)\b", re.I)
POWER_PRE24 = {"ACC", "Big Ten", "Big 12", "SEC", "Pac-12"}
POWER_24 = {"ACC", "Big Ten", "Big 12", "SEC"}
KEEP_COLS = ["id", "gameId", "driveId", "offense", "offenseConference", "defense", "defenseConference",
    "home", "away", "period", "offenseScore", "defenseScore", "yardsToGoal", "down", "distance",
    "yardsGained", "scoring", "playType", "playText", "ppa"]

def load_json(p):
    return json.loads(Path(p).read_text())

def team_maps(year):
    teams = load_json(RAW / f"teams_fbs_{year}.json")
    alias, conf = {}, {}
    for t in teams:
        s = t["school"]; conf[s] = t.get("conference")
        alias[s] = s
        for a in t.get("alternateNames") or []:
            alias.setdefault(a, s)
    return alias, conf

def sp_prior(year):
    sp = [r for r in load_json(RAW / f"sp_{year-1}.json") if r.get("team") != "nationalAverages" and r.get("rating") is not None]
    ratings = {r["team"]: r["rating"] for r in sp}
    return ratings, float(np.percentile(list(ratings.values()), 25)), len(ratings)

def success(df):
    yg, dist, dn = df.yardsGained, df.distance, df.down
    classic = np.select([dn == 1, dn == 2, dn.isin([3, 4])], [yg >= 0.5 * dist, yg >= 0.7 * dist, yg >= dist], default=False)
    fail = df.playType.isin(AUTO_FAIL) | ((df.playType == "Sack") & df.scoring)
    win = df.scoring & df.playType.isin(TD_TYPES)
    return np.where(fail, 0, np.where(win, 1, classic.astype(int)))

def load_year(year):
    frames, weeks = [], {}
    for p in sorted(RAW.glob(f"plays_{year}_w*.json"), key=lambda p: int(p.stem.split("_w")[1])):
        w = int(p.stem.split("_w")[1]); d = load_json(p); weeks[w] = len(d)
        if d:
            f = pd.DataFrame(d); f["year"] = year; f["week"] = w; frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    return df, weeks

def build():
    funnel, cutoffs, all_snaps, weeks_all = [], {}, [], {}
    for y in YEARS:
        raw, weeks = load_year(y); weeks_all[y] = weeks
        n_raw = len(raw)
        raw = raw.drop_duplicates("id")
        n_dedup = len(raw)
        df = raw[raw.playType.isin(OFFENSE_TYPES)].copy()
        n_type = len(df)
        txt = df.playText.fillna("")
        kneel, spike = txt.str.contains(KNEEL), txt.str.contains(SPIKE)
        df = df[~kneel & ~spike]
        n_ks = len(df)
        df = df[df.down.isin([1, 2, 3, 4])]
        n_down = len(df)
        df = df[[c for c in KEEP_COLS if c in df.columns] + ["year", "week"]].copy()
        df["scoring"] = df.scoring.fillna(False).astype(bool)
        df["success"] = success(df)
        alias, conf = team_maps(y)
        df["off_school"] = df.offense.map(alias); df["def_school"] = df.defense.map(alias)
        df["is_fbs_offense"] = df.off_school.notna(); df["is_fbs_defense"] = df.def_school.notna()
        df["def_conf_season"] = df.def_school.map(conf)
        power = POWER_24 if y >= 2024 else POWER_PRE24
        df["is_power_defense"] = df.is_fbs_defense & (df.def_conf_season.isin(power) | (df.def_school == "Notre Dame"))
        ratings, p25, n_sp = sp_prior(y)
        cutoffs[y] = {"prior_year": y - 1, "p25_sp": round(p25, 3), "n_sp_teams": n_sp}
        df["def_prior_sp"] = df.def_school.map(ratings)
        df["is_low_end_defense"] = df.is_fbs_defense & ~df.is_power_defense & (df.def_prior_sp.isna() | (df.def_prior_sp < p25))
        df["use_in_model"] = df.is_fbs_offense & df.is_fbs_defense & ~df.is_low_end_defense
        df["opp_bucket"] = np.select(
            [~df.is_fbs_defense, df.is_low_end_defense, df.is_power_defense],
            ["FCS/non-FBS", "Low-end non-power", "Power"], default="Other FBS")
        df["score_diff"] = df.offenseScore - df.defenseScore
        funnel.append({"year": y, "weeks": len([w for w, n in weeks.items() if n]), "raw_plays": n_raw,
            "dupes_dropped": n_raw - n_dedup, "offense_type_plays": n_type, "kneel_spike_dropped": n_type - n_ks,
            "kneels": int(kneel.sum()), "spikes": int(spike.sum()), "bad_down_dropped": n_ks - n_down,
            "offense_snaps": n_down, "fbs_offense_snaps": int(df.is_fbs_offense.sum()),
            "modeling_snaps": int(df.use_in_model.sum())})
        all_snaps.append(df)
    snaps = pd.concat(all_snaps, ignore_index=True)
    return snaps, pd.DataFrame(funnel), cutoffs, weeks_all

if __name__ == "__main__":
    snaps, funnel, cutoffs, weeks = build()
    snaps.to_parquet(PROC / "offense_snaps.parquet", index=False)
    funnel.to_csv(PROC / "funnel.csv", index=False)
    (PROC / "sp_cutoffs.json").write_text(json.dumps(cutoffs, indent=1))
    (PROC / "weeks.json").write_text(json.dumps(weeks, indent=1))
    print(funnel.to_string()); print(cutoffs)
