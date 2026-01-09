"""
This script analyzes data for unit price funds provided by HSBC 
at https://www.hsbc.com.hk/mpf/tool/unit-prices/

The reason is because in Hong Kong all employees aged 18 to 64 employed 
for a continuous period of 60 days or more, and self-employed persons, 
must participate in an MPF (Mandatory Provident Fund) scheme

The goal is to determine if a person in their 30s should accept the default 
DIS investment plan or decide on a customized investment plan

Default Investment Strategy (DIS) is a ready-made investment arrangement 
mainly designed for those members who are not interested or do not wish to make 
an investment choice
It focuses on 2 funds (CAF, APF) with percentages varying based on the investor's plan and goal
CAF is more focused on potential gain but risker, while APF is more focused on stability
Before age 50         -> DIS = 100% CAF (Core accumulation fund)
After age 65          -> DIS = 100% APF (Age 64 plus fund)
Between age 50 and 65 -> DIS is a mixture of the two

To find a better investment strategy we investigate the available funds' behavior between 2020 and 2025

This script does the following three things:

A) Data engineering (MPF .csv files -> clean merged per-fund histories)
   - Merge all partial .csv downloads for each fund code (e.g., VEEF_1.csv, VEEF_2.csv, ... -> VEEF.csv)
   - Sort chronologically, drop duplicate dates
   - Filter to a requested window (X_START..X_END)
   - Write one clean .csv per fund into OUTPUT_DIR

B) Visualization
   - Plot each group of funds:
       * Top panel: normalized performance index with a baseline at 100
       * Bottom panel: moving average + Bollinger bands (same color as top line) + baseline at 100
   - Plot ROI comparison portfolios with the same two-panel style

C) Portfolio search (grid search over weights)
   - Build a "DIS proxy" portfolio (for a ~35 year-old: effectively CAF-only in this simplified proxy)
   - Search portfolios that can "beat DIS" on performance while staying "close enough" on risk
   - Avoid fake diversification by discarding combinations whose funds are too similar
     (high correlation, low effective number of bets)

---------------------------------------------------------------------------
RULEBOOK (what's considered "better than DIS")
---------------------------------------------------------------------------

The script currently uses a two-stage decision system:

Stage 1: Hard filters ("must pass")
  1) Return > DIS:
       cum_return(port) > cum_return(DIS)

  2) Risk must be comparable to DIS:
       ann_vol(port) <= ann_vol(DIS) + VOL_SLACK
       max_drawdown(port) >= max_drawdown(DIS) - MDD_SLACK
     Notes:
       - max_drawdown is negative. ">=" means "not more negative than"

  3) Diversification is considered as follows:
       max_pair_corr < SIMILARITY_CORR_THRESHOLD
       effective_bets >= k * MIN_EFFECTIVE_BETS_RATIO
     (k = number of funds in the combination, up to 4 funds are considered in a given combination)

Stage 2: Ranking / tie-break ("best among survivors")
  The simplest rule is:
     pick the mix with the highest cum_return within the combination

  You can switch between a pure-return selector and a more complex investor-grade tie-break system (default) via:
      SELECTION_METHOD = "return"  or  "lexi"


"investor-grade" rule set for ranking within a combination:

  Primary objective (what you want more of):
    - Maximize terminal value / cum_return 

  Secondary objectives (don't win by taking stupid risk):
    - Prefer higher Sharpe and Sortino (more return per unit of volatility,
      and more return per unit of *downside* volatility)
    - Prefer higher Calmar (more CAGR per unit of max drawdown)
    - Prefer less negative CVaR (better behaviour on the worst days)
    - Prefer lower Ulcer Index and shorter drawdown duration (less time underwater)
    - Prefer downside capture < 1 (lose less than DIS in DIS-down months)
    - Prefer upside capture >= 1 (keep up or exceed DIS in DIS-up months)

Practical "beat DIS" interpretation for these metrics:
  - Sharpe / Sortino: higher than DIS is "better risk-adjusted"
  - Calmar: higher than DIS means you earned return without paying huge drawdown
  - CVaR(95%): closer to 0 (less negative) than DIS means better tail protection
  - Ulcer Index: lower than DIS means less prolonged pain
  - Downside capture: below 1 is a feature, not a bug (you fall less than DIS)
  - Upside capture: above 1 means you participate more in good months

---------------------------------------------------------------------------
Dependencies:
  pip install pandas numpy matplotlib
---------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# User configuration
INPUT_DIR   = Path("/mnt/c/Users/scana/Desktop/MPF") # folder containing .csv files downloaded from https://www.hsbc.com.hk/mpf/tool/unit-prices/
OUTPUT_DIR  = Path("./merged")                       # folder for outputs
RESULTS_CSV = OUTPUT_DIR / "grid_search_results.csv"

X_START = pd.Timestamp("2020-01-01") # start time to consider
X_END   = pd.Timestamp("2025-12-31") # end time to consider

# Lower panel stats
MA_WINDOW     = 20   # moving average window size
BB_K          = 2.0  # number of sigma to consider for Bollinger bands
BB_LINE_ALPHA = 0.75 # Bollinger bands transparency line
BB_FILL_ALPHA = 0.12 # Bollinger bands transparency fill color

# Portfolio search config
WEIGHT_STEP       = 0.1 # weight delta for each fund in grid search
MAX_FUNDS_IN_PORT = 5   # a search is done for each combination of funds between 1 and MAX_FUNDS_IN_PORT 

# Enforce each fund weight >= 0.1, prevents situation with (A, B) funds with (0.0, 1.0) weights
MIN_WEIGHT = 0.1

# Softened risk budget vs DIS (absolute percentage points)
VOL_SLACK   = 0.02 # allow up to 2% worse annualized volatility
MDD_SLACK   = 0.02 # allow up to 2% worse drawdown
COMPARE_TOL = 0.05

# Similarity / effective bets filter (a priori, per fund combination)
SIMILARITY_CORR_THRESHOLD = 0.92 # if any pair corr >= SIMILARITY_CORR_THRESHOLD is considered too similar 
MIN_EFFECTIVE_BETS_RATIO  = 0.70 


# Best-mix selection logic (within a fixed fund tuple)
#   Stage 1: FILTER (must beat DIS + risk + similarity constraints)
#   Stage 2: RANK survivors (return first, then reward "good risk shape")
#
# SELECTION_METHOD controls Stage 2 ranking within each fund tuple:
#   - "return": keep behaviour exactly as before (highest cum_return among survivors)
#   - "lexi":   lexicographic tie-breaks after cum_return using additional metrics

SELECTION_METHOD = "lexi"  # "return" or "lexi"
# require ENB >= k * ratio (k = #funds)

# DIS proxy assumption
DIS_WEIGHTS        = {"CAF": 1.0, "APF": 0.0} # funds making up the DIS investment plan for someone in their 30s
INITIAL_INVESTMENT = 100.0

# Risk-free rate assumption (annualized) for Sharpe/Sortino
RISK_FREE_ANNUAL = 0.0

# CVaR confidence level: 95% => average of worst 5% days
CVAR_ALPHA = 0.95

# Fund grouping metadata
# Grouping is done on a simple basis of how the fund is broadly defined
@dataclass(frozen=True)
class FundMeta:
    code:  str
    name:  str
    group: str

FUND_META: Dict[str, FundMeta] = {
    "CPF":  FundMeta("CPF",  "MPF Conservative Fund", "Money Market"),
 
    "GBF":  FundMeta("GBF",  "Global Bond Fund", "Bonds / Guaranteed"),
    "GTF":  FundMeta("GTF",  "Guaranteed Fund", "Bonds / Guaranteed"),

    "CAF":  FundMeta("CAF",  "Core Accumulation Fund", "Lifecycle (DIS Multi-Asset)"),
    "APF":  FundMeta("APF",  "Age 65 Plus Fund", "Lifecycle (DIS Multi-Asset)"),

    "SBF":  FundMeta("SBF",  "Stable Fund", "Mixed Asset"),
    "BLF":  FundMeta("BLF",  "Balanced Fund", "Mixed Asset"),
    "GRF":  FundMeta("GRF",  "Growth Fund", "Mixed Asset"),
    "VBLF": FundMeta("VBLF", "ValueChoice Balanced Fund", "Mixed Asset"),

    "GEF":  FundMeta("GEF",  "Global Equity Fund", "Equity (Active / Regional)"),
    "NAEF": FundMeta("NAEF", "North American Equity Fund", "Equity (Active / Regional)"),
    "EUEF": FundMeta("EUEF", "European Equity Fund", "Equity (Active / Regional)"),
    "ANEF": FundMeta("ANEF", "Asia Pacific Equity Fund", "Equity (Active / Regional)"),
    "HKEF": FundMeta("HKEF", "Hong Kong and Chinese Equity Fund", "Equity (Active / Regional)"),
    "CNEF": FundMeta("CNEF", "Chinese Equity Fund", "Equity (Active / Regional)"),

    "VUEF": FundMeta("VUEF", "ValueChoice North America Equity Tracker Fund", "Equity (Index / Tracker)"),
    "VEEF": FundMeta("VEEF", "ValueChoice Europe Equity Tracker Fund", "Equity (Index / Tracker)"),
    "VAEF": FundMeta("VAEF", "ValueChoice Asia Pacific Equity Tracker Fund", "Equity (Index / Tracker)"),
    "HSIF": FundMeta("HSIF", "Hang Seng Index Tracking Fund", "Equity (Index / Tracker)"),
    "HSHF": FundMeta("HSHF", "Hang Seng China Enterprises Index Tracking Fund", "Equity (Index / Tracker)"),
}


# Formatting settings
def fmt(x: object) -> str:
    """Format numbers to max 3 decimals; keep strings readable"""
    if x is None:
        return "None"
    if isinstance(x, (float, int, np.floating, np.integer)):
        if pd.isna(x):
            return "nan"
        return f"{float(x):.3f}"
    return str(x)


def fmt_tuple(t: Tuple[float, ...]) -> str:
    """Format a tuple of floats (e.g., weights) to 3 decimals each"""
    return "(" + ", ".join(f"{float(v):.3f}" for v in t) + ")"


# Parsing + merging
def fund_code_from_filename(path: Path) -> str:
    """Extract the symbol (fund code) from filenames (e.g. extract VEEF from VEEF_1.csv)"""
    m = re.match(r"^(?P<code>[A-Za-z0-9]+)_", path.stem)
    if not m:
        raise ValueError(f"Cannot extract fund code from filename: {path.name}")
    return m.group("code").upper()


def load_hsbc_csv(path: Path) -> pd.DataFrame:
    """Load HSBC unit price .csv file and prepare a pandas.DataFrame"""
    df = pd.read_csv(path)
    first_col = df.columns[0]

    # Convert first col to date
    out = pd.DataFrame()
    out["date"] = (
        df[first_col]
        .astype(str)
        .str.replace("\t", "", regex=False)
        .str.strip()
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    out["bid"] = pd.to_numeric(df["BID"], errors="coerce") if "BID" in df.columns else pd.NA
    out["offer"] = pd.to_numeric(df["OFFER"], errors="coerce") if "OFFER" in df.columns else pd.NA
    
    # Clean and sort
    out = out.dropna(subset=["date"]).sort_values("date")

    out["price"] = out["bid"]
    out.loc[out["price"].isna(), "price"] = out["offer"]

    out = out.dropna(subset=["price"])
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return out[["date", "bid", "offer", "price"]]


def merge_all_funds(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Merge all .csv files in input_dir by code, sort, dedupe"""
    csvs = sorted([p for p in input_dir.glob("*.csv*") if p.is_file()])

    buckets: Dict[str, List[pd.DataFrame]] = {}
    for path in csvs:
        code = fund_code_from_filename(path)
        buckets.setdefault(code, []).append(load_hsbc_csv(path))

    merged: Dict[str, pd.DataFrame] = {}
    for code, parts in buckets.items():
        df = pd.concat(parts, ignore_index=True)
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        merged[code] = df

    return merged


def filter_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Filter data to only consider the dates in interval [start, end]"""
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def save_per_fund(merged: Dict[str, pd.DataFrame], out_dir: Path) -> Dict[str, pd.DataFrame]:
    """Save filtered merged funds into out_dir as CODE.csv and return filtered dict"""
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered: Dict[str, pd.DataFrame] = {}
    for code, df in merged.items():
        w = filter_window(df, X_START, X_END)
        if w.empty:
            continue
        w.to_csv(out_dir / f"{code}.csv", index=False)
        filtered[code] = w

    return filtered


def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize 'price' to predefined value of 100"""
    s          = df.sort_values("date").copy()
    base       = float(s["price"].iloc[0])
    s["index"] = (s["price"] / base) * 100.0 if base != 0 else pd.NA
    return s


def add_ma_bollinger(df_norm: pd.DataFrame, window: int, k: float) -> pd.DataFrame:
    """Compute rolling MA and Bollinger bands on normalized index"""
    s             = df_norm.sort_values("date").copy()
    s["ma"]       = s["index"].rolling(window=window, min_periods=window).mean()
    s["std"]      = s["index"].rolling(window=window, min_periods=window).std()
    s["bb_upper"] = s["ma"] + k * s["std"]
    s["bb_lower"] = s["ma"] - k * s["std"]
    return s


# Plotting helpers
def plot_groups(data: Dict[str, pd.DataFrame]) -> None:
    """Grouped plots: top normalized index, bottom MA + Bollinger; consistent colors; xlim fixed by [start, end]"""
    group_map: Dict[str, List[FundMeta]] = {}
    for code in data.keys():
        meta = FUND_META.get(code, FundMeta(code, code, "Other / Unknown"))
        group_map.setdefault(meta.group, []).append(meta)

    for group in sorted(group_map.keys()):
        metas = sorted(group_map[group], key=lambda m: m.code)

        fig, (ax_top, ax_bot) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(11, 7),
            gridspec_kw={"height_ratios": [2.2, 1.4]},
        )

        # Baseline at 100 on both panels
        ax_top.axhline(100.0, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)
        ax_bot.axhline(100.0, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)

        plotted = False

        for meta in metas:
            df_raw = data.get(meta.code)
            if df_raw is None or df_raw.empty:
                continue

            df_norm  = normalize_to_100(df_raw)
            df_stats = add_ma_bollinger(df_norm, window=MA_WINDOW, k=BB_K)

            # Top: normalized line (capture color so bottom matches)
            top_line = ax_top.plot(
                df_stats["date"], df_stats["index"],
                label=f"{meta.code} - {meta.name}"
            )[0]
            color = top_line.get_color()
            plotted = True

            # Bottom: MA + Bollinger using same color
            ax_bot.plot(df_stats["date"], df_stats["ma"], color=color, label=f"{meta.code} MA")
            ax_bot.plot(df_stats["date"], df_stats["bb_upper"], color=color, alpha=BB_LINE_ALPHA, linewidth=0.9)
            ax_bot.plot(df_stats["date"], df_stats["bb_lower"], color=color, alpha=BB_LINE_ALPHA, linewidth=0.9)
            ax_bot.fill_between(
                df_stats["date"], df_stats["bb_lower"], df_stats["bb_upper"],
                color=color, alpha=BB_FILL_ALPHA
            )

        if not plotted:
            plt.close(fig)
            continue

        ax_bot.set_xlim([X_START, X_END])

        ax_top.set_title(f"{group} (Normalized Performance, Start=100)")
        ax_top.set_ylabel("Index")
        ax_bot.set_title(f"Rolling MA ({MA_WINDOW}) + Bollinger Bands (±{BB_K}σ)")
        ax_bot.set_xlabel("Date")
        ax_bot.set_ylabel("Index")

        ax_top.legend(fontsize=8)
        fig.tight_layout()
        plt.show()


# Portfolio math (ROI series + metrics)
def build_price_matrix(data: Dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Create a daily calendar matrix of prices for all funds, forward-filled"""
    idx    = pd.date_range(start, end, freq="D")
    prices = pd.DataFrame(index=idx)

    for code, df in data.items():
        s = df[["date", "price"]].drop_duplicates("date", keep="last").set_index("date")["price"]
        s = s[(s.index >= start) & (s.index <= end)]
        prices[code] = s

    prices = prices.ffill()
    return prices


def portfolio_value_series(prices: pd.DataFrame, weights: Dict[str, float], initial: float = 100.0) -> pd.Series:
    """
    Value(t) = initial * sum_i w_i * (price_i(t) / price_i(t0))

    Notes:
    - t0 is the first date where ALL selected funds have non-NaN prices
    - That makes comparisons fair when some funds start later
    """
    codes = list(weights.keys())
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()

    p = prices[codes].copy()
    valid = p.notna().all(axis=1)
    if not valid.any():
        raise RuntimeError(f"No overlapping date range with prices for: {codes}")

    t0 = valid.idxmax()
    p  = p.loc[t0:].ffill()

    base = p.loc[t0, codes]
    rel  = p[codes] / base
    val  = initial * (rel.mul(w, axis=1).sum(axis=1))
    val.name = "value"
    return val


def _max_drawdown_and_duration(v: pd.Series) -> Tuple[float, int]:
    """
    Returns:
      max_drawdown (decimal, negative)
      max_drawdown_duration (in business days)

    Why duration matters:
    - Two portfolios can both have -15% max drawdown
      But for examples one might recovers in 2 months, the other might takes 2 years
    """
    peak   = v.cummax()
    dd     = v / peak - 1.0
    max_dd = float(dd.min())

    underwater = dd < 0
    max_len    = 0
    current    = 0
    for flag in underwater.to_numpy(dtype=bool):
        if flag:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0

    return max_dd, int(max_len)


def _ulcer_index(v: pd.Series) -> float:
    """
    Ulcer Index: sqrt(mean(drawdown^2)), where drawdown is measured from prior peak

    Behavior:
    - Lower is better
    - Punishes long and deep underwater periods
    """
    peak = v.cummax()
    dd   = v / peak - 1.0
    return float(np.sqrt(np.mean(np.square(dd.to_numpy(dtype=float)))))


def _cvar_expected_shortfall(r: pd.Series, alpha: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall at confidence alpha (e.g., 0.95):
    - Compute the (1-alpha) quantile (e.g., 5% worst daily return threshold)
    - Return the mean return of days worse than that threshold

    Behavior:
    - Higher (less negative) is better
    - Captures tail risk that volatility misses
    """
    if r.empty:
        return np.nan
    q    = float(r.quantile(1.0 - alpha))
    tail = r[r <= q]
    if tail.empty:
        return float(q)
    return float(tail.mean())


def _rolling_worst_12m(v: pd.Series, window_bdays: int = 252) -> float:
    """
    Worst rolling ~12-month return (business-day window)

    Behavior:
    - Higher is better (less negative worst-year stretch)
    """
    if len(v) <= window_bdays:
        return np.nan
    roll = v / v.shift(window_bdays) - 1.0
    return float(roll.min())


def _monthly_returns(v: pd.Series) -> pd.Series:
    """Month-end sampled returns (useful for capture/hit rate; less noisy than daily)"""
    m = v.resample("ME").last()
    return m.pct_change().dropna()


def perf_metrics(
    val: pd.Series,
    benchmark_val: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    cvar_alpha: float = 0.95,
) -> Dict[str, float]:
    """
    Compute a richer set of metrics on business days

    'How to beat DIS' guideline:
    - Higher return (cum_return, cagr)
    - Higher Sharpe/Sortino/Calmar (more return per unit risk/pain)
    - Less negative CVaR (better worst-case days)
    - Lower Ulcer index and shorter drawdown duration
    - If comparing vs DIS:
        information_ratio > 0
        downside_capture < 1
        upside_capture >= 1 (ideally)
Metric interpretation quick guide:
  - cum_return: total return over the aligned window.
  - cagr: annualized growth rate (approx). Useful for comparing periods of different length.
  - ann_vol: annualized volatility from daily returns (std * sqrt(252)).
  - max_drawdown: worst peak-to-trough decline. More negative is worse.
  - max_dd_duration_bdays: longest continuous time spent below a prior peak.
  - sharpe: excess return per unit volatility (assumes symmetric risk).
  - sortino: excess return per unit *downside* volatility (penalizes only negative swings).
  - calmar: CAGR / |max_drawdown| (reward return, punish deep drawdown).
  - cvar_95: expected shortfall at 95% (average of worst 5% daily returns). Tail-risk measure.
  - ulcer_index: RMS of drawdowns. Penalizes deep and long underwater periods.
  - hit_rate_monthly: fraction of positive months. Stability proxy.
  - worst_rolling_12m: worst one-year rolling return. Regime pain proxy.

Relative-to-DIS metrics (only when benchmark_val is provided):
  - tracking_error: volatility of active returns (portfolio - DIS).
  - information_ratio: mean active return / tracking error (higher is better).
  - upside_capture: how much you capture in DIS-up months (>=1 is good).
  - downside_capture: how much you capture in DIS-down months (<=1 is good).
  - corr_with_dis: correlation of daily returns vs DIS.
  - beta_vs_dis: sensitivity to DIS moves.

    """
    v = val[val.index.dayofweek < 5].dropna().copy()

    if benchmark_val is not None:
        b = benchmark_val[benchmark_val.index.dayofweek < 5].dropna().copy()
        common_idx = v.index.intersection(b.index)
        v = v.loc[common_idx]
        b = b.loc[common_idx]
    else:
        b = None

    r = v.pct_change().dropna()
    if len(r) == 0:
        base = {
            "cum_return": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "max_drawdown": np.nan,
            "max_dd_duration_bdays": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "calmar": np.nan,
            "cvar_95": np.nan,
            "ulcer_index": np.nan,
            "hit_rate_monthly": np.nan,
            "worst_rolling_12m": np.nan,
        }
        if b is not None:
            base.update({
                "tracking_error": np.nan,
                "information_ratio": np.nan,
                "upside_capture": np.nan,
                "downside_capture": np.nan,
                "corr_with_dis": np.nan,
                "beta_vs_dis": np.nan,
            })
        return base

    cum     = float(v.iloc[-1] / v.iloc[0] - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252))
    cagr    = float((v.iloc[-1] / v.iloc[0]) ** (252 / len(r)) - 1.0)

    mdd, mdd_dur = _max_drawdown_and_duration(v)

    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    mean_excess_daily = float((r - rf_daily).mean())

    sharpe = np.nan
    if r.std(ddof=1) > 0:
        sharpe = float(mean_excess_daily * 252.0 / (r.std(ddof=1) * np.sqrt(252)))

    downside = (r - rf_daily).copy()
    downside = downside[downside < 0]
    sortino  = np.nan
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = float(mean_excess_daily * 252.0 / (downside.std(ddof=1) * np.sqrt(252)))

    calmar = np.nan
    if mdd < 0:
        calmar = float(cagr / abs(mdd))

    cvar_95 = _cvar_expected_shortfall(r, alpha=cvar_alpha)
    ulcer   = _ulcer_index(v)

    mret     = _monthly_returns(v)
    hit_rate = float((mret > 0).mean()) if len(mret) > 0 else np.nan

    worst_12m = _rolling_worst_12m(v, window_bdays=252)

    out: Dict[str, float] = {
        "cum_return": cum,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "max_drawdown": float(mdd),
        "max_dd_duration_bdays": float(mdd_dur),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "cvar_95": float(cvar_95),
        "ulcer_index": float(ulcer),
        "hit_rate_monthly": float(hit_rate),
        "worst_rolling_12m": float(worst_12m),
    }

    if b is not None:
        rb = b.pct_change().dropna()
        common_r_idx = r.index.intersection(rb.index)
        r2  = r.loc[common_r_idx]
        rb2 = rb.loc[common_r_idx]

        active = (r2 - rb2)
        te = float(active.std(ddof=1) * np.sqrt(252)) if active.std(ddof=1) > 0 else np.nan
        ir = np.nan
        if active.std(ddof=1) > 0:
            ir = float(active.mean() * 252.0 / (active.std(ddof=1) * np.sqrt(252)))

        corr = float(r2.corr(rb2)) if len(r2) > 1 else np.nan

        beta = np.nan
        if len(rb2) > 1 and float(rb2.var(ddof=1)) > 0:
            beta = float(np.cov(r2, rb2, ddof=1)[0, 1] / rb2.var(ddof=1))

        mret_p   = _monthly_returns(v)
        mret_b   = _monthly_returns(b)
        common_m = mret_p.index.intersection(mret_b.index)
        mret_p   = mret_p.loc[common_m]
        mret_b   = mret_b.loc[common_m]

        upside_capture   = np.nan
        downside_capture = np.nan
        if len(common_m) > 0:
            up_mask = mret_b > 0
            down_mask = mret_b < 0

            if up_mask.any():
                port_up = float(np.prod(1.0 + mret_p[up_mask]) - 1.0)
                bench_up = float(np.prod(1.0 + mret_b[up_mask]) - 1.0)
                if bench_up != 0:
                    upside_capture = port_up / bench_up

            if down_mask.any():
                port_down = float(np.prod(1.0 + mret_p[down_mask]) - 1.0)
                bench_down = float(np.prod(1.0 + mret_b[down_mask]) - 1.0)
                if bench_down != 0:
                    downside_capture = port_down / bench_down

        out.update({
            "tracking_error": float(te),
            "information_ratio": float(ir),
            "upside_capture": float(upside_capture),
            "downside_capture": float(downside_capture),
            "corr_with_dis": float(corr),
            "beta_vs_dis": float(beta),
        })

    return out


# ROI plotting (two panels + baseline line)
def _series_to_bdays(s: pd.Series) -> pd.Series:
    s2 = s.copy()
    s2 = s2[s2.index.dayofweek < 5]
    return s2.dropna()


def _ma_bb_on_series(v: pd.Series, window: int, k: float) -> pd.DataFrame:
    df = pd.DataFrame({"v": v})
    df["ma"] = df["v"].rolling(window=window, min_periods=window).mean()
    df["std"] = df["v"].rolling(window=window, min_periods=window).std()
    df["bb_upper"] = df["ma"] + k * df["std"]
    df["bb_lower"] = df["ma"] - k * df["std"]
    return df


def plot_roi_comparison(series_map: Dict[str, pd.Series], title: str) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(11, 7),
        gridspec_kw={"height_ratios": [2.2, 1.4]},
    )

    ax_top.axhline(100.0, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)
    ax_bot.axhline(100.0, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)

    for label, s in series_map.items():
        s_bd = _series_to_bdays(s)
        s_bd = s_bd[(s_bd.index >= X_START) & (s_bd.index <= X_END)]
        if s_bd.empty:
            continue

        top_line = ax_top.plot(s_bd.index, s_bd.values, label=label)[0]
        color = top_line.get_color()

        stats = _ma_bb_on_series(s_bd, window=MA_WINDOW, k=BB_K)
        ax_bot.plot(stats.index, stats["ma"], color=color, label=f"{label} MA")
        ax_bot.plot(stats.index, stats["bb_upper"], color=color, alpha=BB_LINE_ALPHA, linewidth=0.9)
        ax_bot.plot(stats.index, stats["bb_lower"], color=color, alpha=BB_LINE_ALPHA, linewidth=0.9)
        ax_bot.fill_between(stats.index, stats["bb_lower"], stats["bb_upper"], color=color, alpha=BB_FILL_ALPHA)

    ax_bot.set_xlim([X_START, X_END])

    ax_top.set_title(title)
    ax_top.set_ylabel("Portfolio Value (Initial = 100)")
    ax_top.legend(fontsize=8)

    ax_bot.set_title(f"Rolling MA ({MA_WINDOW}) + Bollinger Bands (±{BB_K}σ)")
    ax_bot.set_xlabel("Date")
    ax_bot.set_ylabel("Value")

    fig.tight_layout()
    plt.show()


# Similarity / effective bets (a priori filter for combinations)
def _effective_bets_from_corr(corr: pd.DataFrame) -> float:
    """
    Effective number of bets (ENB) from correlation matrix eigenvalues
    If ENB is different from k which is number of funds in the combination, then the funds are somewhat related to each other

    Idea:
    - If funds are identical, corr has 1 big eigenvalue and the rest tiny => ENB ~ 1
    - If funds are independent, eigenvalues are more even => ENB approaches k

    Computation:
      p_i = λ_i / sum(λ)
      ENB = 1 / sum(p_i^2)
    """
    if corr.empty:
        return np.nan
    c = corr.to_numpy(dtype=float)
    vals = np.linalg.eigvalsh(c)
    vals = np.clip(vals, 0.0, None)
    s = float(vals.sum())
    if s <= 0:
        return np.nan
    p = vals / s
    return float(1.0 / np.sum(p * p))


def combo_similarity_stats(rel: pd.DataFrame, funds: Tuple[str, ...]) -> Tuple[float, float]:
    """
    Returns:
      (effective_bets, max_pair_corr)

    Uses daily returns of rel prices (aligned by construction)
    """
    k = len(funds)
    if k <= 1:
        return 1.0, 0.0

    r = rel[list(funds)].pct_change().dropna(how="any")
    if r.empty or r.shape[0] < 10:
        # Not enough data to judge similarity; be conservative and keep it (don't discard).
        return float(k), 0.0

    corr = r.corr()

    # max off-diagonal correlation (positive similarity we want to avoid)
    corr_np = corr.to_numpy(dtype=float)
    mask = ~np.eye(k, dtype=bool)
    max_corr = float(np.nanmax(corr_np[mask])) if mask.any() else 0.0

    enb = _effective_bets_from_corr(corr)
    return float(enb), float(max_corr)


# Grid search: compositions with min weight + keep best per fund combination
def generate_weight_compositions(step: float, k: int, min_weight: float = 0.1) -> List[Tuple[float, ...]]:
    S = int(round(1.0 / step))
    min_i = int(round(min_weight / step))

    if k == 1:
        return [(1.0,)]

    comps: List[Tuple[float, ...]] = []

    def rec(remaining: int, parts_left: int) -> Iterable[Tuple[int, ...]]:
        if parts_left == 1:
            yield (remaining,)
            return
        for x in range(0, remaining + 1):
            for rest in rec(remaining - x, parts_left - 1):
                yield (x,) + rest

    for ints in rec(S, k):
        if any(i < min_i for i in ints):
            continue
        w = tuple(i / S for i in ints)
        if max(w) == 1.0:
            continue
        comps.append(w)

    return comps


# Ranking helpers (Stage 2: "best" among survivors within a fund tuple)
def _rank_key_within_combo(m: Dict[str, float]) -> Tuple:
    """
    Return a tuple used to pick the "best" mix *within a fixed fund combination*

    IMPORTANT: The overall grid-search already enforces "must pass" filters:
      - cum_return > DIS
      - ann_vol <= DIS + slack
      - max_drawdown >= DIS - slack
      - similarity / effective bets filter

    So the ranking below is only deciding among *survivors*

    Reasoning we discussed (investor lens):
      1) Primary: higher return
      2) Then prefer portfolios that achieve the return with "better risk shape":
           - higher Calmar (return vs drawdown)
           - higher Sortino (return vs downside volatility)
           - better tail risk (less-negative CVaR)
           - less prolonged pain (lower Ulcer Index, shorter drawdown duration)
           - better behaviour relative to DIS: downside capture <= 1, upside capture >= 1

    Implementation approach:
      - We use a lexicographic tuple so we don't have to argue about exact weights
      - Each component is oriented so "higher is better" for sorting
    """
    # Helper to safely fetch floats; missing metrics become very bad so they lose ties
    def g(key: str, default: float) -> float:
        v = m.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    cum = g("cum_return", -np.inf)

    # Calmar: higher is better. If missing, treat as -inf
    calmar = g("calmar", -np.inf)

    # Sortino: higher is better (downside risk-adjusted)
    sortino = g("sortino", -np.inf)

    # CVaR: typically negative. Less negative (closer to 0) is better => higher is better
    cvar = g("cvar_95", -np.inf)

    # Ulcer index: lower is better => negate to make "higher is better"
    ulcer = -g("ulcer_index", np.inf)

    # Drawdown duration: lower is better => negate
    dd_dur = -g("max_dd_duration_bdays", np.inf)

    # Downside capture: <=1 is preferred (lose less than DIS in DIS-down months)
    # Smaller is better => negate. If missing, make it terrible
    downside_cap = -g("downside_capture", np.inf)

    # Upside capture: >=1 is preferred (keep up or exceed DIS in DIS-up months)
    upside_cap = g("upside_capture", -np.inf)

    # Information ratio: higher is better if available (consistency vs DIS)
    ir = g("information_ratio", -np.inf)

    # Add more here if you like: hit_rate_monthly (higher), worst_rolling_12m (higher), etc
    hit = g("hit_rate_monthly", -np.inf)
    worst12 = g("worst_rolling_12m", -np.inf)

    return (cum, calmar, sortino, cvar, ulcer, dd_dur, downside_cap, upside_cap, ir, hit, worst12)


def _print_combo_best(funds: Tuple[str, ...], weights: Tuple[float, ...], row: Dict[str, object]) -> None:
    print("\nBest mix for combination:", funds, "weights:", fmt_tuple(weights))
    print("  effective_bets:", fmt(row.get("effective_bets")))
    print("  max_pair_corr :", fmt(row.get("max_pair_corr")))
    keys = [
        "cum_return", "cagr", "ann_vol", "max_drawdown", "max_dd_duration_bdays",
        "sharpe", "sortino", "calmar", "cvar_95", "ulcer_index",
        "hit_rate_monthly", "worst_rolling_12m",
        "tracking_error", "information_ratio",
        "upside_capture", "downside_capture",
    ]
    for k in keys:
        if k in row:
            print(f"  {k:>22}: {fmt(row[k])}")


def search_non_dis_portfolios(
    prices_bday: pd.DataFrame,
    exclude_codes: List[str],
    max_funds: int = 4,
    step: float = 0.1,
    min_weight: float = 0.1,
    vol_slack: float = 0.02,
    mdd_slack: float = 0.02,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    """
    Grid search over 1..max_funds fund combinations (long-only constant weights)

    A priori discard filter:
    - Compute (effective_bets, max_pair_corr) per combination
    - Discard if:
        max_pair_corr >= SIMILARITY_CORR_THRESHOLD
        OR effective_bets < k * MIN_EFFECTIVE_BETS_RATIO
    """
    dis_codes = list(DIS_WEIGHTS.keys())
    universe = [c for c in prices_bday.columns if c not in exclude_codes]

    needed_cols = sorted(set(universe + dis_codes))
    p_all = prices_bday[needed_cols].dropna()
    if p_all.empty:
        return pd.DataFrame(), {}, pd.Series(dtype=float)

    dis_val = portfolio_value_series(p_all, DIS_WEIGHTS, initial=INITIAL_INVESTMENT)
    dis_m = perf_metrics(dis_val, benchmark_val=None, rf_annual=RISK_FREE_ANNUAL, cvar_alpha=CVAR_ALPHA)

    dis_cum = float(dis_m["cum_return"])
    dis_vol = float(dis_m["ann_vol"])
    dis_mdd = float(dis_m["max_drawdown"])

    vol_limit = dis_vol + vol_slack
    mdd_limit = dis_mdd - mdd_slack

    t0 = p_all.index[0]
    rel = p_all / p_all.loc[t0]
    rel_last = rel.iloc[-1]

    results: List[Dict[str, object]] = []
    discarded_similar = 0

    # k=1
    for a in universe:
        terminal_rel = float(rel_last[a])
        if terminal_rel - 1.0 <= dis_cum:
            continue

        val = INITIAL_INVESTMENT * rel[a]
        m = perf_metrics(val, benchmark_val=dis_val, rf_annual=RISK_FREE_ANNUAL, cvar_alpha=CVAR_ALPHA)

        beats = (m["cum_return"] > dis_cum)
        risk_ok = (
            # Volatility: lower is better. We allow some slack vs DIS.
            (m["ann_vol"] <= vol_limit)
            and
            # Drawdown is negative. ">= mdd_limit" means "not *more* negative than the limit".
            (m["max_drawdown"] >= mdd_limit)
        )

        if beats and risk_ok:
            row: Dict[str, object] = {
                "k": 1, "funds": (a,), "weights": (1.0,),
                "effective_bets": 1.0, "max_pair_corr": 0.0,
                **m,
            }
            results.append(row)
            if verbose:
                _print_combo_best((a,), (1.0,), row)

    # k>=2: best-per-combination with similarity filter
    for k in range(2, max_funds + 1):
        comps = generate_weight_compositions(step, k, min_weight=min_weight)
        required_enb = k * MIN_EFFECTIVE_BETS_RATIO

        for funds in combinations(universe, k):
            enb, max_corr = combo_similarity_stats(rel, funds)
            if (
                # If any pair is extremely correlated, it is likely the same risk factor expressed twice.
                (max_corr >= SIMILARITY_CORR_THRESHOLD)
                or
                # ENB below threshold implies "effective diversification" is too low for k funds.
                (enb < required_enb)
            ):
                discarded_similar += 1
                continue

            best_row: Optional[Dict[str, object]] = None
            best_cum = -np.inf
            best_key: Optional[Tuple] = None
            last_vec = rel_last[list(funds)].to_numpy(dtype=float)

            for w in comps:
                wv = np.asarray(w, dtype=float)

                terminal_rel = float(np.dot(wv, last_vec))
                if terminal_rel - 1.0 <= dis_cum:
                    continue

                series_rel = rel[list(funds)].to_numpy(dtype=float) @ wv
                val = pd.Series(INITIAL_INVESTMENT * series_rel, index=rel.index, name="value")

                m = perf_metrics(val, benchmark_val=dis_val, rf_annual=RISK_FREE_ANNUAL, cvar_alpha=CVAR_ALPHA)

                beats = (m["cum_return"] > dis_cum)
                risk_ok = (
            # Volatility: lower is better. We allow some slack vs DIS.
            (m["ann_vol"] <= vol_limit)
            and
            # Drawdown is negative. ">= mdd_limit" means "not *more* negative than the limit".
            (m["max_drawdown"] >= mdd_limit)
        )
                if not (beats and risk_ok):
                    continue

                if SELECTION_METHOD == "return":
                    choose = (m["cum_return"] > best_cum)
                else:
                    # Lexicographic ranking: return first, then better risk-shape metrics.
                    key = _rank_key_within_combo(m)
                    choose = (best_key is None) or (key > best_key)

                if choose:
                # Selection rule within a fixed fund tuple:
                #   Primary objective: maximize return among mixes that beat DIS + meet risk constraints.
                # In a more advanced setup, you would add tie-breaks here (e.g., Calmar, CVaR, Ulcer).

                    best_cum = float(m["cum_return"])
                    best_key = _rank_key_within_combo(m)
                    best_row = {
                        "k": k, "funds": funds, "weights": w,
                        "effective_bets": enb, "max_pair_corr": max_corr,
                        **m,
                    }

            if best_row is not None:
                results.append(best_row)
                if verbose:
                    _print_combo_best(best_row["funds"], best_row["weights"], best_row)

    if verbose and discarded_similar > 0:
        print(f"\nDiscarded {discarded_similar} combinations as 'too similar' "
              f"(corr>={SIMILARITY_CORR_THRESHOLD} or ENB<{MIN_EFFECTIVE_BETS_RATIO:.2f}*k).")

    df = pd.DataFrame(results)
    if df.empty:
        return df, dis_m, dis_val

    df = df.sort_values("cum_return", ascending=False).reset_index(drop=True)
    return df, dis_m, dis_val


# ---------------------------------------------------------------------
# Console reporting with DIS comparison
# ---------------------------------------------------------------------
# Print the best models (overall, and best per k):
#    portfolio_value | DIS_value | label(better/worse/similar)
# where "similar" means within ±2% of DIS (or ±0.02 absolute when DIS is ~0)
#
# This requires a direction map: which way is "better"?
# - Return-like metrics: higher is better
# - Risk/pain metrics: lower is better
# - Drawdown is negative; "higher" means "less negative" => better
METRIC_DIRECTION = {
    "cum_return": "higher",
    "cagr": "higher",
    "ann_vol": "lower",
    "max_drawdown": "higher",
    "max_dd_duration_bdays": "lower",
    "sharpe": "higher",
    "sortino": "higher",
    "calmar": "higher",
    "cvar_95": "higher",
    "ulcer_index": "lower",
    "hit_rate_monthly": "higher",
    "worst_rolling_12m": "higher",
    "tracking_error": "lower",
    "information_ratio": "higher",
    "upside_capture": "higher",
    "downside_capture": "lower",
    "corr_with_dis": "lower",
    "beta_vs_dis": "lower",
}


def _cmp_label(port_val: float, dis_val: float, direction: str, tol: float = COMPARE_TOL) -> str:
    """Return 'better'/'worse'/'similar' given a direction and tolerance"""
    if pd.isna(port_val) or pd.isna(dis_val):
        return "n/a"

    denom = abs(float(dis_val))

    # If DIS is ~0, percent comparisons explode. Use absolute tolerance
    if denom < 1e-12:
        diff = float(port_val) - float(dis_val)
        if abs(diff) <= tol:
            return "similar"
        if direction == "higher":
            return "better" if diff > tol else "worse"
        if direction == "lower":
            return "better" if diff < -tol else "worse"
        return "n/a"

    rel = (float(port_val) - float(dis_val)) / denom
    if abs(rel) <= tol:
        return "similar"

    if direction == "higher":
        return "better" if rel > tol else "worse"
    if direction == "lower":
        return "better" if rel < -tol else "worse"
    return "n/a"


def print_row_with_dis_comparison(title: str, row: pd.Series, dis_ref: Dict[str, float]) -> None:
    """Print a portfolio row and compare each metric to DIS"""
    print(title)
    print("  funds         :", row.get("funds"))
    print("  weights       :", row.get("weights"))
    print("  effective_bets:", fmt(row.get("effective_bets")))
    print("  max_pair_corr :", fmt(row.get("max_pair_corr")))

    for metric, direction in METRIC_DIRECTION.items():
        if metric not in row.index or metric not in dis_ref:
            continue
        pv = float(row[metric]) if not pd.isna(row[metric]) else np.nan
        dv = float(dis_ref[metric]) if not pd.isna(dis_ref[metric]) else np.nan
        label = _cmp_label(pv, dv, direction, tol=COMPARE_TOL)
        print(f"  {metric:>22}: {fmt(pv)}   | DIS: {fmt(dv)}   => {label}")

def _print_key_metrics(title: str, m: Dict[str, float]) -> None:
    keys = [
        "cum_return", "cagr", "ann_vol", "max_drawdown", "max_dd_duration_bdays",
        "sharpe", "sortino", "calmar", "cvar_95", "ulcer_index",
        "hit_rate_monthly", "worst_rolling_12m",
        "tracking_error", "information_ratio",
        "upside_capture", "downside_capture",
    ]
    print(title)
    for k in keys:
        if k in m:
            print(f"  {k:>22}: {fmt(m[k])}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {INPUT_DIR.resolve()}")

    merged = merge_all_funds(INPUT_DIR)
    filtered = save_per_fund(merged, OUTPUT_DIR)

    # 1) Grouped plots
    plot_groups(filtered)

    # 2) Build price matrix + DIS series
    prices = build_price_matrix(filtered, X_START, X_END)
    prices_b = prices[prices.index.dayofweek < 5].copy()

    dis_val_full = portfolio_value_series(prices_b, DIS_WEIGHTS, initial=INITIAL_INVESTMENT)
    dis_m_full = perf_metrics(dis_val_full, benchmark_val=None, rf_annual=RISK_FREE_ANNUAL, cvar_alpha=CVAR_ALPHA)

    print("\nDIS proxy metrics (CAF-only, ~35 y/o) on its available window:")
    _print_key_metrics("DIS:", dis_m_full)

    # 3) Search for portfolios that beat DIS under slack + similarity constraints
    # df_best contains ONE ROW PER FUND TUPLE (funds=...) corresponding to the best mix for that tuple
    df_best, dis_m_aligned, dis_val_aligned = search_non_dis_portfolios(
        prices_bday=prices_b,
        exclude_codes=list(DIS_WEIGHTS.keys()),
        max_funds=MAX_FUNDS_IN_PORT,
        step=WEIGHT_STEP,
        min_weight=MIN_WEIGHT,
        vol_slack=VOL_SLACK,
        mdd_slack=MDD_SLACK,
        verbose=False,
    )

    if df_best.empty:
        print(
            f"\nNo portfolio beats DIS while staying within risk limits "
            f"(vol +{VOL_SLACK*100:.0f}pp, drawdown -{MDD_SLACK*100:.0f}pp), "
            f"min weight {MIN_WEIGHT:.1f}, and similarity filter."
        )
        return


    # ------------------------------------------------------------
    # Save ALL grid-search results to CSV
    # ------------------------------------------------------------
    # df_best has ONE ROW PER FUND TUPLE (funds=...) corresponding to the best weight mix
    # found for that tuple under the filters + ranking method.
    #
    # Saving results is better than printing everything because:
    #   - it is reproducible
    #   - you can filter/sort without re-running the grid search
    #   - it plays nicely with Excel/pandas
    df_out = df_best.copy()
    if not df_out.empty:
        # Make tuple columns human-friendly in CSV
        df_out["funds"] = df_out["funds"].apply(
            lambda x: ";".join(list(x)) if isinstance(x, (tuple, list)) else str(x)
        )
        df_out["weights"] = df_out["weights"].apply(
            lambda x: ";".join([f"{float(w):.3f}" for w in x]) if isinstance(x, (tuple, list)) else str(x)
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(RESULTS_CSV, index=False)
        print(f"\nSaved grid-search results to: {RESULTS_CSV.resolve()}")

        # DIS reference metrics for comparison printing.
    # Benchmark DIS against itself so relative-to-DIS fields are defined (capture ~ 1, tracking error ~ 0).
    dis_ref = perf_metrics(dis_val_aligned, benchmark_val=dis_val_aligned, rf_annual=RISK_FREE_ANNUAL, cvar_alpha=CVAR_ALPHA)

    print("\nDIS metrics aligned to the search window (constraints reference):")
    _print_key_metrics("DIS (aligned):", dis_m_aligned)

    best_overall = df_best.iloc[0]
    print_row_with_dis_comparison("\nBest overall (among best-per-combination candidates):", best_overall, dis_ref)

    best_k: Dict[int, pd.Series] = {}
    for k in range(2, MAX_FUNDS_IN_PORT+1, 1):
        df_k = df_best[df_best["k"] == k]
        if df_k.empty:
            print(f"\nNo {k}-fund combination beats DIS under the constraints.")
            continue
        best_k[k] = df_k.iloc[0]
        print_row_with_dis_comparison(f"\nBest k={k} (best mix per combination):", best_k[k], dis_ref)

    # 4) ROI comparison plot (two panels, baseline at 100)
    series_map: Dict[str, pd.Series] = {"DIS proxy (CAF-only)": dis_val_full}

    funds_o = list(best_overall["funds"])
    weights_o = dict(zip(funds_o, best_overall["weights"]))
    series_map[f"Best overall (non-DIS): {funds_o}"] = portfolio_value_series(prices, weights_o, INITIAL_INVESTMENT)

    for k, row in best_k.items():
        funds = list(row["funds"])
        weights = dict(zip(funds, row["weights"]))
        series_map[f"Best k={k}: {funds}"] = portfolio_value_series(prices, weights, INITIAL_INVESTMENT)

    plot_roi_comparison(series_map, f"ROI Comparison (Initial=100), {X_START.date()} to {X_END.date()}")


if __name__ == "__main__":
    main()
