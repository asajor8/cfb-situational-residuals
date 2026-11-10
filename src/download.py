"""Download CFBD (collegefootballdata.com) plays, FBS team lists, and SP+ ratings.
Caches raw JSON under data/raw/; reruns skip files that exist and parse as valid non-empty JSON.
API key read from env CFBD_API_KEY (or a local .env); never printed or written."""
import json, os, sys, time
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass
from paths import RAW, ensure_dirs

BASE = "https://api.collegefootballdata.com"
ensure_dirs()
YEARS = [2019, 2021, 2022, 2023, 2024, 2025]
SP_YEARS = list(range(2018, 2025))
TEAM_YEARS = [2018] + YEARS
MAX_WEEK = 17
LOG = RAW / "download_log.json"

def _headers():
    k = os.environ.get("CFBD_API_KEY")
    if not k:
        sys.exit("CFBD_API_KEY not set")
    return {"Authorization": f"Bearer {k}", "Accept": "application/json"}

def get(path, params, tries=8):
    delay = 2
    for i in range(tries):
        try:
            r = requests.get(BASE + path, params=params, headers=_headers(), timeout=120)
        except requests.RequestException as e:
            print(f"  net error {type(e).__name__}, retry in {delay}s"); time.sleep(delay); delay = min(delay * 2, 120); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            wait = int(ra) if ra and ra.isdigit() else delay
            print(f"  HTTP {r.status_code} on {path} {params}, retry in {wait}s"); time.sleep(wait); delay = min(delay * 2, 120); continue
        raise RuntimeError(f"HTTP {r.status_code} on {path} {params}: {r.text[:200]}")
    raise RuntimeError(f"gave up on {path} {params}")

def valid(p):
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        json.loads(p.read_text()); return True
    except Exception:
        return False

def cached(p, path, params, allow_empty=False):
    if valid(p):
        return json.loads(p.read_text())
    data = get(path, params)
    if data is None:
        data = []
    if data or allow_empty:
        p.write_text(json.dumps(data))
    return data

def main():
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    for y in TEAM_YEARS:
        cached(RAW / f"teams_fbs_{y}.json", "/teams/fbs", {"year": y})
    for y in SP_YEARS:
        cached(RAW / f"sp_{y}.json", "/ratings/sp", {"year": y})
    for y in YEARS:
        counts = {}
        empties = 0
        for w in range(1, MAX_WEEK + 1):
            p = RAW / f"plays_{y}_w{w}.json"
            data = cached(p, "/plays", {"year": y, "week": w, "seasonType": "regular", "classification": "fbs"})
            counts[w] = len(data)
            print(f"{y} w{w}: {len(data)} plays", flush=True)
            if not data:
                empties += 1
                if empties >= 2:
                    break
            else:
                empties = 0
            time.sleep(0.3)
        log[str(y)] = counts
        LOG.write_text(json.dumps(log, indent=1))

if __name__ == "__main__":
    main()
