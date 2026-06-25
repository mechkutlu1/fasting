"""
Pull your logged month from the Fasting Console API (or the SQLite file directly)
and run *real* SHAP on it with scikit-learn + the shap library.

The feature engineering mirrors the in-app analysis exactly, so the browser's
closed-form linear-SHAP and shap.LinearExplainer here agree (up to scaling).

Usage:
    pip install -r requirements.txt
    python analyze.py --api http://localhost:8000 --target kcal
    # or read the DB file written by main.py:
    python analyze.py --file fasting.db --target kcal
"""
import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False


# ---------- load state ----------
def load_from_api(base: str) -> dict:
    import requests
    return requests.get(base.rstrip("/") + "/state", timeout=10).json()


def load_from_file(path: str) -> dict:
    con = sqlite3.connect(path)
    row = con.execute("SELECT json FROM state WHERE id = 1").fetchone()
    con.close()
    return json.loads(row[0]) if row else {"settings": {}, "days": {}}


# ---------- feature engineering (mirrors the app) ----------
def hhmm_to_dec(t):
    if not t or ":" not in t:
        return np.nan
    h, m = t.split(":")
    return int(h) + int(m) / 60.0


def day_macros(rec):
    k = p = c = f = 0.0
    for e in rec.get("entries", []):
        k += e["kcal"]; p += e["p"]; c += e["c"]; f += e["f"]
    return k, p, c, f


def day_exercise(rec):
    """Returns (minutes, kcal, did_exercise) tolerating legacy records."""
    exs = rec.get("exercises")
    if exs:
        mins = sum(e.get("minutes", 0) for e in exs)
        kcal = sum(e.get("kcal", 0) for e in exs)
        return mins, kcal, 1 if mins > 0 else 0
    if rec.get("exercised"):
        return rec.get("exerciseMin", 0) or 0, 0, 1
    return 0, 0, 0


def build_frame(state: dict) -> pd.DataFrame:
    s = state.get("settings", {})
    lo, hi = s.get("rangeLow", 1700), s.get("rangeHigh", 2100)
    rows = []
    for date, rec in sorted(state.get("days", {}).items()):
        if not rec.get("entries"):
            continue
        k, p, c, f = day_macros(rec)
        ex_min, ex_kcal, ex_flag = day_exercise(rec)
        first = hhmm_to_dec(rec.get("firstMeal"))
        last = hhmm_to_dec(rec.get("lastMeal"))
        window = (last - first) if (not np.isnan(first) and not np.isnan(last) and last >= first) else np.nan
        rows.append({
            "date": date,
            "fast_h": (24 - window) if not np.isnan(window) else np.nan,
            "window_h": window,
            "first_h": first,
            "exercised": ex_flag,
            "ex_min": ex_min,
            "ex_kcal": ex_kcal,
            "meals": len(rec.get("entries", [])),
            "within_range": 1 if (lo <= k <= hi) else 0,
            "protein": p, "carb": c, "fat": f,
            "kcal": k, "net": k - ex_kcal, "weight": rec.get("weight"),
        })
    return pd.DataFrame(rows).set_index("date")


# ---------- analysis ----------
TARGET_EXCLUDE = {
    "kcal": ["protein", "carb", "fat", "within_range", "weight", "net"],
    "net": ["protein", "carb", "fat", "within_range", "weight", "kcal", "ex_kcal"],
    "weight": ["kcal", "net", "weight"],
    "protein": ["protein", "kcal", "net", "weight"],
}
FEATURE_COLS = ["fast_h", "window_h", "first_h", "exercised", "ex_min", "ex_kcal",
                "meals", "within_range", "protein", "carb", "fat"]


def analyse(df: pd.DataFrame, target: str, alpha: float = 1.0):
    cols = [c for c in FEATURE_COLS if c not in TARGET_EXCLUDE.get(target, [])]
    data = df[cols + [target]].dropna()
    if len(data) < 8:
        print(f"Only {len(data)} complete days for target '{target}'. Need >= 8.")
        return

    X = data[cols].values.astype(float)
    y = data[target].values.astype(float)
    keep = [i for i, c in enumerate(cols) if X[:, i].std() > 1e-9]
    cols = [cols[i] for i in keep]
    X = X[:, keep]

    scaler = StandardScaler().fit(X)
    Xz = scaler.transform(X)
    model = Ridge(alpha=alpha).fit(Xz, y)

    print(f"\nTarget: {target}   n={len(data)} days   ridge alpha={alpha}")
    print("-" * 58)

    if HAVE_SHAP:
        explainer = shap.LinearExplainer(model, Xz)
        sv = explainer.shap_values(Xz)
        importance = np.abs(sv).mean(axis=0)
        order = np.argsort(importance)[::-1]
        print(f"{'feature':<16}{'mean|SHAP|':>12}{'effect/+1SD':>14}{'direction':>11}")
        for i in order:
            d = model.coef_[i]
            print(f"{cols[i]:<16}{importance[i]:>12.2f}{d:>+14.1f}"
                  f"{'  raises' if d >= 0 else '  lowers':>11}")
    else:
        # closed form: identical to shap.LinearExplainer for a linear model
        print("(shap not installed — using exact closed-form linear SHAP)")
        importance = np.abs(model.coef_) * np.abs(Xz).mean(axis=0)
        order = np.argsort(importance)[::-1]
        for i in order:
            d = model.coef_[i]
            print(f"{cols[i]:<16}{importance[i]:>12.2f}{d:>+14.1f}"
                  f"{'  raises' if d >= 0 else '  lowers':>11}")

    # bootstrap 95% CI on coefficients (you asked about this for manuscripts)
    rng = np.random.default_rng(0)
    boot = []
    n = len(y)
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        boot.append(Ridge(alpha=alpha).fit(Xz[idx], y[idx]).coef_)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    print("\nbootstrap 95% CI on standardized coefficients:")
    for i in order:
        flag = "*" if (lo[i] > 0 or hi[i] < 0) else " "
        print(f"  {cols[i]:<16}[{lo[i]:+7.1f}, {hi[i]:+7.1f}] {flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", help="base URL, e.g. http://localhost:8000")
    ap.add_argument("--file", help="path to fasting.db")
    ap.add_argument("--target", default="kcal", choices=["kcal", "net", "weight", "protein"])
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    if args.api:
        state = load_from_api(args.api)
    elif args.file:
        state = load_from_file(args.file)
    else:
        raise SystemExit("Pass --api URL or --file fasting.db")

    df = build_frame(state)
    print(f"Loaded {len(df)} logged days.")
    analyse(df, args.target, args.alpha)
