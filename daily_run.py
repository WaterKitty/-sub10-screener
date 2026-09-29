#!/usr/bin/env python3
"""
Daily runner for the sub-$10 screener.
Runs the model, assigns tiers, tracks history, writes a mobile dashboard to
docs/index.html (served by GitHub Pages) and optionally pushes a phone
notification via ntfy.

    python daily_run.py --out docs            # normal scheduled run
    python daily_run.py --out docs --force    # run now regardless of time
    python daily_run.py --out docs --demo     # synthetic data, no network
"""
import argparse
import html
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import sub10_growth_screener as S

TZ = ZoneInfo("America/Los_Angeles")

# Tier rules (composite is 0-100, ~50 = middle of today's universe)
TOP_MIN = 62        # Top tier: composite >= this AND zero red flags
TOP_MAX = 10        # at most this many in Top tier
WATCH_MIN = 52      # Watchlist: composite >= this


# --------------------------------------------------------------------------
def assign_tiers(df):
    df = df.copy()
    df["tier"] = np.where(df["composite"] >= WATCH_MIN, "Watchlist", "Below threshold")
    top = df[(df["composite"] >= TOP_MIN) & (df["n_flags"] == 0)].head(TOP_MAX).index
    df.loc[top, "tier"] = "Top tier"
    return df


def update_history(df, out, today):
    """Append today's tiered names; compute 'new' flag and top-tier streak."""
    path = out / "history.csv"
    hist = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["date", "ticker", "tier", "composite"])
    hist = hist[hist["date"] != today]
    today_rows = df[df["tier"] != "Below threshold"][["ticker", "tier", "composite"]].assign(date=today)
    hist = pd.concat([hist, today_rows[["date", "ticker", "tier", "composite"]]], ignore_index=True)
    dates = sorted(hist["date"].unique())
    hist = hist[hist["date"].isin(dates[-90:])]  # keep ~90 trading days
    hist.to_csv(path, index=False)

    prev_dates = [d for d in dates if d < today]
    prev_top = set(hist[(hist["date"] == prev_dates[-1]) & (hist["tier"] == "Top tier")]["ticker"]) if prev_dates else set()
    top_by_date = {d: set(hist[(hist["date"] == d) & (hist["tier"] == "Top tier")]["ticker"]) for d in dates}

    def streak(t):
        n = 0
        for d in reversed([d for d in dates if d <= today]):
            if t in top_by_date.get(d, set()):
                n += 1
            else:
                break
        return n

    df = df.copy()
    df["streak"] = df["ticker"].apply(streak)
    df["new_today"] = (df["tier"] == "Top tier") & (~df["ticker"].isin(prev_top)) & bool(prev_dates)
    dropped = sorted(prev_top - set(df[df["tier"] == "Top tier"]["ticker"]))
    return df, dropped


# --------------------------------------------------------------------------
def pct(x, signed=True):
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x:+.0%}" if signed else f"{x:.0%}"


def money(x):
    if x is None or not np.isfinite(x):
        return "–"
    for div, suf in ((1e9, "B"), (1e6, "M")):
        if abs(x) >= div:
            return f"${x/div:.1f}{suf}"
    return f"${x:,.0f}"


def e(s):
    return html.escape(str(s if s is not None else ""))


FACTORS = [("growth", "Growth"), ("momentum", "Momentum"), ("quality", "Quality"),
           ("upside", "Upside"), ("sentiment", "Sentiment")]


def card(r):
    bars = "".join(
        f'<div class="f"><span>{lbl}</span><div class="bar"><i style="width:{r[k + "_score"]:.0f}%"></i></div>'
        f'<b>{r[k + "_score"]:.0f}</b></div>' for k, lbl in FACTORS)
    badges = ""
    if r.get("new_today"):
        badges += '<span class="badge new">New</span>'
    if r.get("streak", 0) > 1:
        badges += f'<span class="badge">{int(r["streak"])} days</span>'
    flags = f'<p class="flags">⚠ {e(r["red_flags"])}</p>' if r["red_flags"] else ""
    return f"""
<details class="card">
  <summary>
    <div class="row1"><span class="tk">{e(r['ticker'])}</span>{badges}<span class="sc">{r['composite']:.0f}</span></div>
    <div class="row2"><span>${r['price']:.2f}</span><span class="nm">{e(r['name'])}</span></div>
  </summary>
  <div class="body">
    {bars}
    <dl>
      <dt>Revenue growth</dt><dd>{pct(r['revenue_growth'])}</dd>
      <dt>12-1 mo momentum</dt><dd>{pct(r['mom_12_1'])}</dd>
      <dt>Analyst upside</dt><dd>{pct(r['analyst_upside'])} ({'' if not np.isfinite(r['n_analysts']) else int(r['n_analysts'])} analysts)</dd>
      <dt>Gross margin</dt><dd>{pct(r['gross_margin'], False)}</dd>
      <dt>FCF yield</dt><dd>{pct(r['fcf_yield'])}</dd>
      <dt>Market cap</dt><dd>{money(r['market_cap'])}</dd>
      <dt>Sector</dt><dd>{e(r['sector'])}</dd>
    </dl>
    {flags}
    <a href="https://finance.yahoo.com/quote/{e(r['ticker'])}" target="_blank" rel="noopener">Quote &amp; filings ↗</a>
  </div>
</details>"""


def build_html(df, n_excluded, dropped, stamp, demo):
    top = df[df["tier"] == "Top tier"]
    watch = df[df["tier"] == "Watchlist"]
    top_html = "".join(card(r) for _, r in top.iterrows()) or '<p class="empty">No stock cleared the top-tier bar today.</p>'
    watch_html = "".join(card(r) for _, r in watch.head(20).iterrows()) or '<p class="empty">None today.</p>'
    dropped_html = (f'<p class="muted">Left the top tier since last run: {e(", ".join(dropped))}</p>' if dropped else "")
    demo_html = '<p class="demo">DEMO DATA — synthetic tickers, not real stocks.</p>' if demo else ""
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Sub-$10 Screen">
<title>Sub-$10 Screen · {e(stamp)}</title>
<style>
:root{{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--muted:#6b7280;--line:#e5e7eb;--acc:#0f766e;--acc2:#ccfbf1;--warn:#b45309;
  padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0e1013;--card:#181b20;--ink:#eceef2;--muted:#9aa1ad;--line:#2a2f37;--acc:#2dd4bf;--acc2:#134e4a;--warn:#fbbf24}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}}
main{{max-width:640px;margin:0 auto;padding:16px}}
h1{{font-size:22px;margin:4px 0 2px}} h2{{font-size:13px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:24px 0 8px}}
.muted{{color:var(--muted);font-size:13px;margin:4px 0}}
.demo{{background:#fde68a;color:#78350f;padding:8px 10px;border-radius:8px;font-weight:600}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:8px 0;overflow:hidden}}
summary{{list-style:none;padding:12px 14px;cursor:pointer}} summary::-webkit-details-marker{{display:none}}
.row1{{display:flex;align-items:center;gap:6px}} .tk{{font-weight:700;font-size:17px}}
.sc{{margin-left:auto;font-weight:700;font-size:18px;color:var(--acc)}}
.row2{{display:flex;gap:10px;color:var(--muted);font-size:13px}} .nm{{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.badge{{font-size:11px;padding:2px 7px;border-radius:99px;background:var(--line);color:var(--muted)}}
.badge.new{{background:var(--acc2);color:var(--acc);font-weight:600}}
.body{{padding:0 14px 14px}}
.f{{display:grid;grid-template-columns:78px 1fr 28px;align-items:center;gap:8px;font-size:12px;color:var(--muted);margin:3px 0}}
.bar{{height:6px;background:var(--line);border-radius:3px;overflow:hidden}} .bar i{{display:block;height:100%;background:var(--acc)}}
.f b{{text-align:right;color:var(--ink);font-weight:600}}
dl{{display:grid;grid-template-columns:1fr auto;gap:4px 12px;font-size:13px;margin:12px 0 8px}} dt{{color:var(--muted)}} dd{{margin:0;text-align:right}}
.flags{{color:var(--warn);font-size:13px;margin:6px 0}}
a{{color:var(--acc);font-size:13px;text-decoration:none}}
.empty{{color:var(--muted);font-style:italic}}
footer{{color:var(--muted);font-size:12px;margin:28px 0 12px;border-top:1px solid var(--line);padding-top:12px}}
</style></head><body><main>
<h1>Sub-$10 Growth Screen</h1>
<p class="muted">Run {e(stamp)} PT · {len(df)} stocks scored · {n_excluded} filtered out</p>
{demo_html}
<h2>Top tier · {len(top)}</h2>
<p class="muted">Score ≥ {TOP_MIN}, no red flags. Tap a stock for details.</p>
{top_html}
{dropped_html}
<h2>Watchlist · {len(watch)}</h2>
<p class="muted">Score ≥ {WATCH_MIN}, or a high score with red flags. Showing up to 20.</p>
{watch_html}
<footer>Scores rank each stock against today's other $1–$10 US stocks on growth, momentum, quality,
upside and sentiment, minus 5 points per red flag. They are a research shortlist, not buy recommendations
or price forecasts. Data from Yahoo Finance and may be delayed or incomplete — check the company's latest
10-Q before acting.</footer>
</main></body></html>"""


# --------------------------------------------------------------------------
def notify(top, stamp, url):
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        return
    names = ", ".join(f"{r.ticker} {r.composite:.0f}" for r in top.head(5).itertuples())
    body = f"{len(top)} top-tier: {names}" if len(top) else "No stocks cleared the top tier today."
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode(), method="POST",
                                 headers={"Title": f"Sub-$10 screen {stamp[:10]}", "Tags": "chart_with_upwards_trend"})
    if url:
        req.add_header("Click", url)
    try:
        urllib.request.urlopen(req, timeout=15)
        print("Notification sent")
    except Exception as ex:
        print(f"Notification failed: {ex}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs")
    ap.add_argument("--max-universe", type=int, default=600)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="ignore the time/already-ran guard")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()

    now = datetime.now(TZ)
    today, stamp = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H:%M")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # GitHub cron is UTC, so the workflow fires at two UTC times to cover
    # daylight saving. Only the one landing at 6-7 AM PT does the work.
    if not (a.force or a.demo):
        latest = out / "latest.json"
        if now.hour not in (6, 7):
            sys.exit(f"{stamp} PT is outside the 6-7 AM window; skipping.")
        if latest.exists() and json.loads(latest.read_text()).get("date") == today:
            sys.exit("Already ran today; skipping.")

    if a.demo:
        raw = S.demo_data()
    else:
        try:
            symbols = S.screen_universe(a.max_universe)
        except Exception as ex:
            sys.exit(f"Market screen failed: {ex}")
        print(f"Fetching {len(symbols)} symbols...", file=sys.stderr)
        raw = S.fetch_all(symbols, a.workers)
        if raw.empty:
            sys.exit("No data returned from Yahoo (possibly rate-limited).")

    passed, excluded = S.apply_filters(raw)
    if passed.empty:
        sys.exit("Nothing passed the hard filters.")
    ranked = assign_tiers(S.score(passed))
    ranked, dropped = update_history(ranked, out, today)

    url = os.environ.get("PAGES_URL", "").strip()
    (out / "index.html").write_text(build_html(ranked, len(excluded), dropped, stamp, a.demo), encoding="utf-8")
    cols = [c for c in S.DISPLAY_COLS if c in ranked.columns] + ["tier", "streak", "new_today"]
    ranked[cols].to_csv(out / "latest.csv", index=False)
    top = ranked[ranked["tier"] == "Top tier"]
    (out / "latest.json").write_text(json.dumps({
        "date": today, "run_at": stamp, "scored": len(ranked), "excluded": len(excluded),
        "top_tier": top["ticker"].tolist(),
        "watchlist": ranked[ranked["tier"] == "Watchlist"]["ticker"].tolist(),
    }, indent=2))

    print(f"{stamp} PT: {len(top)} top tier, {(ranked['tier'] == 'Watchlist').sum()} watchlist")
    print(top[["ticker", "price", "composite", "streak"]].to_string(index=False))
    if not a.demo:
        notify(top, stamp, url)


if __name__ == "__main__":
    main()
