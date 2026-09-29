#!/usr/bin/env python3
"""
Sub-$10 Growth Potential Screener
=================================
Ranks US-listed stocks priced $1-$10 on a 5-factor model built to surface
names with 12-24 month upside potential while screening out the usual
low-priced-stock traps (dilution, cash burn, reverse splits, illiquidity).

    pip install yfinance pandas numpy openpyxl
    python sub10_growth_screener.py                 # screen the whole US market
    python sub10_growth_screener.py --tickers my_list.txt
    python sub10_growth_screener.py --demo          # synthetic data, no network

Output: sub10_screen_<date>.xlsx (Ranked, Excluded, Methodology) + .csv

This is a research/screening tool, not a prediction or a recommendation.
Scores rank stocks *relative to each other* on characteristics associated
with future returns; they do not forecast prices.
"""
import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG - tune these
# --------------------------------------------------------------------------
CONFIG = {
    "min_price": 1.0,               # below $1 = delisting risk territory
    "max_price": 10.0,
    "min_market_cap": 300e6,        # excludes micro-caps
    "min_avg_dollar_volume": 2e6,   # 60-day avg daily $ traded
    "max_debt_to_equity": 3.0,
    "min_runway_months": 12,        # only applied when FCF is negative
    "min_analysts_for_target": 3,   # ignore price targets from 1-2 analysts
    "min_data_coverage": 0.6,       # share of metrics that must be available
    "exchanges": ["NMS", "NGM", "NCM", "NYQ", "ASE"],  # Nasdaq, NYSE, NYSE American
}

WEIGHTS = {
    "growth": 0.30,     # is the business actually growing?
    "momentum": 0.20,   # is the market starting to notice?
    "quality": 0.20,    # can it survive long enough to realize the growth?
    "upside": 0.20,     # is there valuation room / analyst upside?
    "sentiment": 0.10,  # insiders, analysts, short interest
}

FLAG_PENALTY = 5.0  # points deducted from composite (0-100) per soft red flag

# metric -> (factor, higher_is_better)
METRICS = {
    "revenue_growth":   ("growth", True),
    "earnings_growth":  ("growth", True),
    "fwd_eps_change":   ("growth", True),
    "mom_12_1":         ("momentum", True),
    "mom_6m":           ("momentum", True),
    "pct_above_200dma": ("momentum", True),
    "gross_margin":     ("quality", True),
    "operating_margin": ("quality", True),
    "fcf_yield":        ("quality", True),
    "current_ratio":    ("quality", True),
    "debt_to_equity":   ("quality", False),
    "analyst_upside":   ("upside", True),
    "ev_to_sales":      ("upside", False),
    "rec_mean":         ("sentiment", False),  # 1 = Strong Buy, 5 = Sell
    "insider_pct":      ("sentiment", True),
    "short_pct_float":  ("sentiment", False),
}


# --------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------
def screen_universe(max_n):
    """Pull candidate symbols from Yahoo's screener."""
    import yfinance as yf
    q = yf.EquityQuery("and", [
        yf.EquityQuery("gt", ["intradayprice", CONFIG["min_price"]]),
        yf.EquityQuery("lt", ["intradayprice", CONFIG["max_price"]]),
        yf.EquityQuery("gt", ["intradaymarketcap", CONFIG["min_market_cap"]]),
        yf.EquityQuery("eq", ["region", "us"]),
        yf.EquityQuery("is-in", ["exchange", *CONFIG["exchanges"]]),
    ])
    symbols, offset = [], 0
    while len(symbols) < max_n:
        res = yf.screen(q, size=250, offset=offset,
                        sortField="intradaymarketcap", sortAsc=False)
        quotes = res.get("quotes", [])
        if not quotes:
            break
        symbols += [x["symbol"] for x in quotes if "symbol" in x]
        offset += len(quotes)
        if offset >= res.get("total", 0):
            break
        time.sleep(0.5)
    return list(dict.fromkeys(symbols))[:max_n]


def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def fetch_one(sym):
    """Fetch fundamentals + price history for one ticker and compute metrics."""
    import yfinance as yf
    t = yf.Ticker(sym)
    info = t.info or {}
    hist = t.history(period="2y", auto_adjust=True)
    if hist is None or hist.empty or len(hist) < 200:
        return None

    close = hist["Close"]
    price = float(close.iloc[-1])

    def ret(days_back, skip=0):
        if len(close) <= days_back:
            return np.nan
        return close.iloc[-1 - skip] / close.iloc[-1 - days_back] - 1

    ma200 = close.rolling(200).mean().iloc[-1]
    hi52 = close.tail(252).max()

    # Reverse splits in the last 2 years (classic distressed-penny-stock tell)
    splits = hist.get("Stock Splits", pd.Series(dtype=float))
    reverse_split = bool(((splits > 0) & (splits < 1)).any())

    # Share dilution, YoY (quarterly balance sheet, newest column first)
    dilution = np.nan
    try:
        bs = t.quarterly_balance_sheet
        for row in ("Ordinary Shares Number", "Share Issued"):
            if row in bs.index:
                s = bs.loc[row].dropna()
                if len(s) >= 5:
                    dilution = s.iloc[0] / s.iloc[4] - 1
                elif len(s) >= 2:
                    dilution = s.iloc[0] / s.iloc[-1] - 1
                break
    except Exception:
        pass

    g = lambda k: _num(info.get(k))
    mcap = g("marketCap")
    fcf = g("freeCashflow")
    cash = g("totalCash")
    tr_eps, fw_eps = g("trailingEps"), g("forwardEps")
    target, n_analysts = g("targetMeanPrice"), g("numberOfAnalystOpinions")
    de = g("debtToEquity")

    runway = np.inf
    if fcf < 0 and cash >= 0:
        runway = cash / (-fcf / 12)

    return {
        "ticker": sym,
        "name": info.get("shortName") or info.get("longName") or sym,
        "sector": info.get("sector", ""),
        "industry": info.get("industry", ""),
        "exchange": info.get("exchange", ""),
        "price": price,
        "market_cap": mcap,
        "avg_dollar_volume": float((close * hist["Volume"]).tail(60).mean()),
        # growth
        "revenue_growth": g("revenueGrowth"),
        "earnings_growth": g("earningsGrowth"),
        "fwd_eps_change": (fw_eps - tr_eps) / abs(tr_eps) if tr_eps and np.isfinite(tr_eps) and tr_eps != 0 else np.nan,
        # momentum
        "mom_12_1": ret(252, 21),
        "mom_6m": ret(126),
        "pct_above_200dma": price / ma200 - 1 if ma200 else np.nan,
        "pct_off_52w_high": price / hi52 - 1 if hi52 else np.nan,
        # quality
        "gross_margin": g("grossMargins"),
        "operating_margin": g("operatingMargins"),
        "fcf_yield": fcf / mcap if mcap else np.nan,
        "current_ratio": g("currentRatio"),
        "debt_to_equity": de / 100 if np.isfinite(de) else np.nan,  # Yahoo reports as %
        "runway_months": runway,
        "dilution_yoy": dilution,
        "reverse_split_2y": reverse_split,
        # upside
        "analyst_target": target,
        "n_analysts": n_analysts,
        "analyst_upside": target / price - 1 if (np.isfinite(target) and n_analysts >= CONFIG["min_analysts_for_target"]) else np.nan,
        "ev_to_sales": g("enterpriseToRevenue"),
        # sentiment
        "rec_mean": g("recommendationMean"),
        "insider_pct": g("heldPercentInsiders"),
        "short_pct_float": g("shortPercentOfFloat"),
        "total_revenue": g("totalRevenue"),
    }


def fetch_all(symbols, workers=4):
    rows, failed = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one, s): s for s in symbols}
        for i, f in enumerate(as_completed(futs), 1):
            s = futs[f]
            try:
                r = f.result()
                (rows if r else failed).append(r or s)
            except Exception:
                failed.append(s)
            if i % 25 == 0 or i == len(symbols):
                print(f"  fetched {i}/{len(symbols)}", file=sys.stderr)
    if failed:
        print(f"  {len(failed)} tickers had insufficient data and were skipped", file=sys.stderr)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------
def apply_filters(df):
    """Hard filters -> (passed, excluded). Excluded rows keep their reasons."""
    c = CONFIG
    reasons = pd.Series([[] for _ in range(len(df))], index=df.index)

    def fail(mask, why):
        for i in df.index[mask.fillna(False)]:
            reasons[i].append(why)

    fail(~df["price"].between(c["min_price"], c["max_price"]), "price outside range")
    fail(df["market_cap"] < c["min_market_cap"], "market cap too small")
    fail(df["avg_dollar_volume"] < c["min_avg_dollar_volume"], "illiquid")
    fail(~(df["total_revenue"] > 0), "no revenue")
    fail(df["debt_to_equity"] > c["max_debt_to_equity"], "excessive leverage")
    fail(df["runway_months"] < c["min_runway_months"], "cash runway < 12 mo")

    metric_cols = list(METRICS)
    df = df.copy()
    df["data_coverage"] = df[metric_cols].notna().mean(axis=1)
    fail(df["data_coverage"] < c["min_data_coverage"], "insufficient data")

    df["exclusion_reasons"] = reasons.apply("; ".join)
    ok = df["exclusion_reasons"] == ""
    return df[ok].copy(), df[~ok].copy()


def red_flags(row):
    flags = []
    if row.get("dilution_yoy", 0) > 0.10:
        flags.append(f"dilution {row['dilution_yoy']:.0%} YoY")
    if row.get("reverse_split_2y"):
        flags.append("reverse split <2y")
    if row.get("runway_months", np.inf) < 24:
        flags.append("runway < 24 mo")
    if row.get("short_pct_float", 0) > 0.20:
        flags.append("short interest > 20%")
    if row.get("operating_margin", 0) < -0.25:
        flags.append("deep operating losses")
    if row.get("pct_off_52w_high", 0) < -0.60:
        flags.append("down >60% from 52w high")
    return flags


def score(df):
    """Percentile-rank each metric within the universe, average into factors,
    weight into a 0-100 composite, then subtract red-flag penalties."""
    df = df.copy()
    by_factor = {f: [] for f in WEIGHTS}
    for m, (f, higher_better) in METRICS.items():
        col = f"_r_{m}"
        df[col] = df[m].rank(pct=True, ascending=higher_better) * 100
        by_factor[f].append(col)

    for f, cols in by_factor.items():
        # Missing factor -> neutral 50 so absent data neither helps nor hurts
        df[f"{f}_score"] = df[cols].mean(axis=1).fillna(50).round(1)

    df["raw_score"] = sum(df[f"{f}_score"] * w for f, w in WEIGHTS.items())
    df["red_flags"] = df.apply(red_flags, axis=1)
    df["n_flags"] = df["red_flags"].apply(len)
    df["composite"] = (df["raw_score"] - FLAG_PENALTY * df["n_flags"]).clip(lower=0).round(1)
    df["red_flags"] = df["red_flags"].apply("; ".join)

    df = df.drop(columns=[c for c in df.columns if c.startswith("_r_")])
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


# --------------------------------------------------------------------------
# OUTPUT
# --------------------------------------------------------------------------
DISPLAY_COLS = [
    "rank", "ticker", "name", "sector", "price", "composite",
    "growth_score", "momentum_score", "quality_score", "upside_score", "sentiment_score",
    "red_flags", "revenue_growth", "mom_12_1", "gross_margin", "fcf_yield",
    "analyst_upside", "n_analysts", "runway_months", "dilution_yoy",
    "market_cap", "avg_dollar_volume", "data_coverage", "industry",
]

METHODOLOGY = [
    ("Universe", f"US stocks on Nasdaq/NYSE/NYSE American, ${CONFIG['min_price']:.0f}-${CONFIG['max_price']:.0f}, market cap >= ${CONFIG['min_market_cap']/1e6:.0f}M"),
    ("Hard filters", "Removes: illiquid (<$2M/day), no revenue, debt/equity > 3, <12 months cash runway, <60% data coverage"),
    ("Scoring", "Each metric is percentile-ranked (0-100) against the other survivors, averaged into 5 factors, then weighted"),
    ("Growth 30%", "Revenue growth YoY, earnings growth YoY, forward vs trailing EPS"),
    ("Momentum 20%", "12-1 month return (skips last month), 6-month return, % above 200-day MA"),
    ("Quality 20%", "Gross margin, operating margin, FCF yield, current ratio, debt/equity (lower better)"),
    ("Upside 20%", "Analyst mean target vs price (3+ analysts only), EV/Sales (lower better)"),
    ("Sentiment 10%", "Analyst rec mean (lower better), insider ownership, short % float (lower better)"),
    ("Red flags", f"-{FLAG_PENALTY:.0f} pts each: dilution >10% YoY, reverse split in 2y, runway <24mo, short >20%, op margin < -25%, >60% off 52w high"),
    ("Caveats", "Relative ranking, not a price forecast. Yahoo data can be stale or missing. Not backtested. Verify filings (10-K/10-Q) before acting."),
]


def save(ranked, excluded, prefix, top):
    stamp = datetime.now().strftime("%Y-%m-%d")
    cols = [c for c in DISPLAY_COLS if c in ranked.columns]
    ranked[cols].to_csv(f"{prefix}_{stamp}.csv", index=False)
    xlsx = f"{prefix}_{stamp}.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
            ranked[cols].to_excel(w, sheet_name="Ranked", index=False)
            ex_cols = ["ticker", "name", "price", "exclusion_reasons"]
            excluded[[c for c in ex_cols if c in excluded.columns]].to_excel(w, sheet_name="Excluded", index=False)
            pd.DataFrame(METHODOLOGY, columns=["Component", "Detail"]).to_excel(w, sheet_name="Methodology", index=False)
            for ws in w.book.worksheets:
                ws.freeze_panes = "A2"
                for col in ws.columns:
                    width = max(len(str(c.value or "")) for c in col[:50])
                    ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 60)
        print(f"Saved {xlsx}")
    except ImportError:
        print("openpyxl not installed - CSV only")
    print(f"Saved {prefix}_{stamp}.csv")

    show = ranked[["rank", "ticker", "price", "composite", "growth_score",
                   "momentum_score", "quality_score", "upside_score", "red_flags"]].head(top)
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(f"\nTop {top} of {len(ranked)} candidates ({len(excluded)} excluded by filters):\n")
        print(show.to_string(index=False))


# --------------------------------------------------------------------------
# DEMO DATA (no network) - fake tickers, for checking the pipeline only
# --------------------------------------------------------------------------
def demo_data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    fcf = rng.normal(5e6, 40e6, n)
    cash = rng.uniform(20e6, 400e6, n)
    df = pd.DataFrame({
        "ticker": [f"DEMO{i:03d}" for i in range(n)],
        "name": [f"Demo Co {i}" for i in range(n)],
        "sector": rng.choice(["Technology", "Healthcare", "Industrials", "Energy", "Financials"], n),
        "industry": "Synthetic", "exchange": "NMS",
        "price": rng.uniform(0.5, 11, n),
        "market_cap": rng.uniform(1e8, 3e9, n),
        "avg_dollar_volume": rng.uniform(5e5, 5e7, n),
        "revenue_growth": rng.normal(0.12, 0.25, n),
        "earnings_growth": rng.normal(0.05, 0.5, n),
        "fwd_eps_change": rng.normal(0.2, 0.6, n),
        "mom_12_1": rng.normal(0.05, 0.4, n),
        "mom_6m": rng.normal(0.03, 0.3, n),
        "pct_above_200dma": rng.normal(0.0, 0.2, n),
        "pct_off_52w_high": -rng.uniform(0, 0.8, n),
        "gross_margin": rng.uniform(0.05, 0.8, n),
        "operating_margin": rng.normal(0.0, 0.2, n),
        "fcf_yield": fcf / 1e9,
        "current_ratio": rng.uniform(0.5, 5, n),
        "debt_to_equity": rng.uniform(0, 4, n),
        "runway_months": np.where(fcf < 0, cash / (-fcf / 12), np.inf),
        "dilution_yoy": rng.normal(0.04, 0.08, n),
        "reverse_split_2y": rng.random(n) < 0.05,
        "analyst_target": np.nan, "n_analysts": rng.integers(0, 12, n),
        "analyst_upside": rng.normal(0.3, 0.3, n),
        "ev_to_sales": rng.uniform(0.3, 12, n),
        "rec_mean": rng.uniform(1.3, 3.5, n),
        "insider_pct": rng.uniform(0, 0.4, n),
        "short_pct_float": rng.uniform(0, 0.35, n),
        "total_revenue": rng.choice([0, 1], n, p=[0.05, 0.95]) * rng.uniform(1e7, 2e9, n),
    })
    for m in METRICS:  # sprinkle missing data like the real feed has
        df.loc[rng.random(n) < 0.08, m] = np.nan
    return df


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", help="text file with one ticker per line (skips market screen)")
    ap.add_argument("--max-universe", type=int, default=750, help="cap on symbols pulled from screener")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--workers", type=int, default=4, help="parallel fetches (keep low to avoid rate limits)")
    ap.add_argument("--out", default="sub10_screen")
    ap.add_argument("--demo", action="store_true", help="run on synthetic data, no network")
    a = ap.parse_args()

    if a.demo:
        raw = demo_data()
    else:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            sys.exit("Install dependencies first:  pip install yfinance pandas numpy openpyxl")
        if a.tickers:
            with open(a.tickers) as f:
                symbols = [s.strip().upper() for s in f if s.strip() and not s.startswith("#")]
        else:
            print("Screening US market for $1-$10 stocks...", file=sys.stderr)
            try:
                symbols = screen_universe(a.max_universe)
            except Exception as e:
                sys.exit(f"Market screen failed ({e}). Update yfinance (pip install -U yfinance) "
                         "or pass --tickers my_list.txt")
        print(f"Fetching data for {len(symbols)} symbols (a few minutes)...", file=sys.stderr)
        raw = fetch_all(symbols, a.workers)
        if raw.empty:
            sys.exit("No data returned.")

    passed, excluded = apply_filters(raw)
    if passed.empty:
        sys.exit("Nothing passed the hard filters - try loosening CONFIG.")
    save(score(passed), excluded, a.out, a.top)


if __name__ == "__main__":
    main()
