"""
fund_analysis_clean.py

Analysis of HSBC MPF unit-price funds with a "can we beat DIS?" backtest.

This is a cleaned-up version of the original `fund_analysis.py`:
- keeps the same methodology,
- improves structure (config + CLI),
- reduces duplicated logic,
- tightens up documentation and comments,
- uses Business Day calendars where appropriate to cut noise.

Data source: HSBC MPF unit price tool
https://www.hsbc.com.hk/mpf/tool/unit-prices/

Notes:
- MPF and DIS details are described in the README and the accompanying presentation.
- This code is for research/education only, not investment advice.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """User-tunable parameters for the analysis."""
    input_dir: Path
    output_dir: Path
    results_csv: Path

    x_start: pd.Timestamp
    x_end: pd.Timestamp

    # Lower panel stats
    ma_window: int = 20
    bb_k: float = 2.0
    bb_line_alpha: float = 0.75
    bb_fill_alpha: float = 0.12

    # Portfolio search config
    weight_step: float = 0.1
    max_funds_in_port: int = 5
    min_weight: float = 0.1

    # Risk slack vs DIS (absolute percentage points)
    vol_slack: float = 0.02
    mdd_slack: float = 0.02
    compare_tol: float = 0.05

    # Similarity / effective bets
    similarity_corr_threshold: float = 0.92
    min_effective_bets_ratio: float = 0.70

    # Selection method for "best mix in a given fund tuple"
    selection_method: str = "lexi"  # "return" or "lexi"

    # DIS proxy assumption (for ~35 y/o: CAF-only)
    dis_weights: Dict[str, float] = None  # set in __post_init__ replacement below

    initial_investment: float = 100.0
    risk_free_annual: float = 0.0
    cvar_alpha: float = 0.95


def _with_default_dis(cfg: Config) -> Config:
    """Inject default DIS proxy weights into an immutable Config."""
    if cfg.dis_weights is not None:
        return cfg
    return Config(
        input_dir=cfg.input_dir,
        output_dir=cfg.output_dir,
        results_csv=cfg.results_csv,
        x_start=cfg.x_start,
        x_end=cfg.x_end,
        ma_window=cfg.ma_window,
        bb_k=cfg.bb_k,
        bb_line_alpha=cfg.bb_line_alpha,
        bb_fill_alpha=cfg.bb_fill_alpha,
        weight_step=cfg.weight_step,
        max_funds_in_port=cfg.max_funds_in_port,
        min_weight=cfg.min_weight,
        vol_slack=cfg.vol_slack,
        mdd_slack=cfg.mdd_slack,
        compare_tol=cfg.compare_tol,
        similarity_corr_threshold=cfg.similarity_corr_threshold,
        min_effective_bets_ratio=cfg.min_effective_bets_ratio,
        selection_method=cfg.selection_method,
        dis_weights={"CAF": 1.0, "APF": 0.0},
        initial_investment=cfg.initial_investment,
        risk_free_annual=cfg.risk_free_annual,
        cvar_alpha=cfg.cvar_alpha,
    )


# Fund grouping metadata (same as original, kept for plotting readability)
@dataclass(frozen=True)
class FundMeta:
    code: str
    name: str
    group: str


FUND_META: Dict[str, FundMeta] = {
    "CPF": FundMeta("CPF", "MPF Conservative Fund", "Money Market"),
    "GBF": FundMeta("GBF", "Global Bond Fund", "Bonds / Guaranteed"),
    "GTF": FundMeta("GTF", "Guaranteed Fund", "Bonds / Guaranteed"),
    "CAF": FundMeta("CAF", "Core Accumulation Fund", "Lifecycle (DIS Multi-Asset)"),
    "APF": FundMeta("APF", "Age 65 Plus Fund", "Lifecycle (DIS Multi-Asset)"),
    "SBF": FundMeta("SBF", "Stable Fund", "Mixed Asset"),
    "BLF": FundMeta("BLF", "Balanced Fund", "Mixed Asset"),
    "GRF": FundMeta("GRF", "Growth Fund", "Mixed Asset"),
    "VBLF": FundMeta("VBLF", "ValueChoice Balanced Fund", "Mixed Asset"),
    "GEF": FundMeta("GEF", "Global Equity Fund", "Equity (Active / Regional)"),
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


# ---------------------------------------------------------------------
# IO / Parsing
# ---------------------------------------------------------------------
def fund_code_from_filename(path: Path) -> str:
    """
    Extract the fund code from filenames like:
      VEEF_1.csv -> VEEF

    HSBC downloads usually include this pattern; adapt if your filenames differ.
    """
    match = re.match(r"^(?P<code>[A-Za-z0-9]+)_", path.stem)
    if not match:
        raise ValueError(f"Cannot extract fund code from filename: {path.name}")
    return match.group("code").upper()


def load_hsbc_csv(path: Path) -> pd.DataFrame:
    """
    Load an HSBC unit price CSV.

    The site sometimes changes formatting slightly, so this loader is defensive:
    - parse the first column as date,
    - read BID/OFFER if present,
    - choose BID else OFFER as 'price'.
    """
    df = pd.read_csv(path)
    first_col = df.columns[0]

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

    out = out.dropna(subset=["date"]).sort_values("date")

    # Prefer BID; fall back to OFFER where BID is missing.
    out["price"] = out["bid"]
    out.loc[out["price"].isna(), "price"] = out["offer"]

    out = out.dropna(subset=["price"])
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return out[["date", "bid", "offer", "price"]]


def merge_all_funds(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Merge all CSV files in input_dir by fund code."""
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
    """Filter [start, end] inclusive."""
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()


def save_per_fund(
    merged: Dict[str, pd.DataFrame],
    out_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Save filtered merged funds into out_dir as CODE.csv and return the filtered dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    filtered: Dict[str, pd.DataFrame] = {}
    for code, df in merged.items():
        w = filter_window(df, start, end)
        if w.empty:
            continue
        w.to_csv(out_dir / f"{code}.csv", index=False)
        filtered[code] = w

    return filtered


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
def normalize_to_100(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize 'price' to start at 100."""
    s = df.sort_values("date").copy()
    base = float(s["price"].iloc[0])
    s["index"] = (s["price"] / base) * 100.0 if base != 0 else pd.NA
    return s


def add_ma_bollinger(df_norm: pd.DataFrame, window: int, k: float) -> pd.DataFrame:
    """Compute rolling MA and Bollinger bands on normalized index."""
    s = df_norm.sort_values("date").copy()
    s["ma"] = s["index"].rolling(window=window, min_periods=window).mean()
    s["std"] = s["index"].rolling(window=window, min_periods=window).std()
    s["bb_upper"] = s["ma"] + k * s["std"]
    s["bb_lower"] = s["ma"] - k * s["std"]
    return s


def plot_groups(cfg: Config, data: Dict[str, pd.DataFrame]) -> None:
    """Plot each fund group as a two-panel chart."""
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

            df_norm = normalize_to_100(df_raw)
            df_stats = add_ma_bollinger(df_norm, window=cfg.ma_window, k=cfg.bb_k)

            # Top panel: normalized index
            top_line = ax_top.plot(
                df_stats["date"], df_stats["index"],
                label=f"{meta.code} - {meta.name}",
            )[0]
            color = top_line.get_color()
            plotted = True

            # Bottom panel: MA + Bollinger (match color)
            ax_bot.plot(df_stats["date"], df_stats["ma"], color=color, label=f"{meta.code} MA")
            ax_bot.plot(df_stats["date"], df_stats["bb_upper"], color=color, alpha=cfg.bb_line_alpha, linewidth=0.9)
            ax_bot.plot(df_stats["date"], df_stats["bb_lower"], color=color, alpha=cfg.bb_line_alpha, linewidth=0.9)
            ax_bot.fill_between(
                df_stats["date"], df_stats["bb_lower"], df_stats["bb_upper"],
                color=color, alpha=cfg.bb_fill_alpha,
            )

        if not plotted:
            plt.close(fig)
            continue

        ax_bot.set_xlim([cfg.x_start, cfg.x_end])

        ax_top.set_title(f"{group} (Normalized Performance, Start=100)")
        ax_top.set_ylabel("Index")
        ax_bot.set_title(f"Rolling MA ({cfg.ma_window}) + Bollinger Bands (±{cfg.bb_k}σ)")
        ax_bot.set_xlabel("Date")
        ax_bot.set_ylabel("Index")

        ax_top.legend(fontsize=8)
        fig.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# Portfolio math
# ---------------------------------------------------------------------
def build_price_matrix(
    data: Dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Create a Business Day calendar matrix of prices for all funds, forward-filled.

    Using business days keeps the dataset smaller and avoids weekend placeholders
    in the subsequent metrics.
    """
    idx = pd.date_range(start, end, freq="B")
    prices = pd.DataFrame(index=idx)

    for code, df in data.items():
        s = df[["date", "price"]].drop_duplicates("date", keep="last").set_index("date")["price"]
        s = s[(s.index >= start) & (s.index <= end)]
        prices[code] = s

    return prices.ffill()


def portfolio_value_series(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    initial: float = 100.0,
) -> pd.Series:
    """
    Portfolio value series (buy-and-hold, constant weights).

    Value(t) = initial * sum_i w_i * (price_i(t) / price_i(t0))

    - t0 is the first date where ALL selected funds have non-NaN prices.
    - That makes comparisons fair when some funds start later.
    """
    codes = list(weights.keys())
    w = pd.Series(weights, dtype=float)
    w = w / w.sum()

    p = prices[codes].copy()
    valid = p.notna().all(axis=1)
    if not valid.any():
        raise RuntimeError(f"No overlapping date range with prices for: {codes}")

    t0 = valid.idxmax()
    p = p.loc[t0:].ffill()

    base = p.loc[t0, codes]
    rel = p[codes] / base
    val = initial * (rel.mul(w, axis=1).sum(axis=1))
    val.name = "value"
    return val


def _max_drawdown_and_duration(v: pd.Series) -> Tuple[float, int]:
    """
    Returns:
      - max_drawdown (decimal, negative)
      - max_drawdown_duration (in business days)

    Duration matters because:
      same drawdown depth, very different "time underwater".
    """
    peak = v.cummax()
    dd = v / peak - 1.0
    max_dd = float(dd.min())

    underwater = dd < 0
    max_len = 0
    current = 0
    for flag in underwater.to_numpy(dtype=bool):
        if flag:
            current += 1
            max_len = max(max_len, current)
        else:
            current = 0

    return max_dd, int(max_len)


def _ulcer_index(v: pd.Series) -> float:
    """Ulcer Index: sqrt(mean(drawdown^2)). Lower is better."""
    peak = v.cummax()
    dd = v / peak - 1.0
    return float(np.sqrt(np.mean(np.square(dd.to_numpy(dtype=float)))))


def _cvar_expected_shortfall(r: pd.Series, alpha: float = 0.95) -> float:
    """
    CVaR / Expected Shortfall at confidence alpha (e.g., 0.95):
    mean of the worst (1-alpha) fraction of returns.
    """
    if r.empty:
        return np.nan

    q = float(r.quantile(1.0 - alpha))
    tail = r[r <= q]
    return float(tail.mean()) if not tail.empty else float(q)


def _rolling_worst_12m(v: pd.Series, window_bdays: int = 252) -> float:
    """Worst rolling ~12-month return. Higher is better (less negative)."""
    if len(v) <= window_bdays:
        return np.nan
    roll = v / v.shift(window_bdays) - 1.0
    return float(roll.min())


def _monthly_returns(v: pd.Series) -> pd.Series:
    """Month-end sampled returns (less noisy than daily for capture/hit-rate)."""
    m = v.resample("ME").last()
    return m.pct_change().dropna()


def perf_metrics(
    val: pd.Series,
    benchmark_val: Optional[pd.Series] = None,
    rf_annual: float = 0.0,
    cvar_alpha: float = 0.95,
) -> Dict[str, float]:
    """
    Compute a richer metric set.

    If benchmark_val is provided, also compute relative-to-benchmark metrics
    (tracking error, information ratio, capture ratios, corr, beta).
    """
    v = val.dropna().copy()

    if benchmark_val is not None:
        b = benchmark_val.dropna().copy()
        common_idx = v.index.intersection(b.index)
        v = v.loc[common_idx]
        b = b.loc[common_idx]
    else:
        b = None

    r = v.pct_change().dropna()
    if len(r) == 0:
        return {k: np.nan for k in [
            "cum_return", "cagr", "ann_vol", "max_drawdown", "max_dd_duration_bdays",
            "sharpe", "sortino", "calmar", "cvar_95", "ulcer_index",
            "hit_rate_monthly", "worst_rolling_12m",
            "tracking_error", "information_ratio",
            "upside_capture", "downside_capture",
            "corr_with_dis", "beta_vs_dis",
        ]}

    cum = float(v.iloc[-1] / v.iloc[0] - 1.0)
    ann_vol = float(r.std(ddof=1) * np.sqrt(252))
    cagr = float((v.iloc[-1] / v.iloc[0]) ** (252 / len(r)) - 1.0)

    mdd, mdd_dur = _max_drawdown_and_duration(v)

    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    mean_excess_daily = float((r - rf_daily).mean())

    sharpe = np.nan
    if r.std(ddof=1) > 0:
        sharpe = float(mean_excess_daily * 252.0 / (r.std(ddof=1) * np.sqrt(252)))

    downside = (r - rf_daily).copy()
    downside = downside[downside < 0]
    sortino = np.nan
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = float(mean_excess_daily * 252.0 / (downside.std(ddof=1) * np.sqrt(252)))

    calmar = np.nan
    if mdd < 0:
        calmar = float(cagr / abs(mdd))

    cvar_95 = _cvar_expected_shortfall(r, alpha=cvar_alpha)
    ulcer = _ulcer_index(v)
    mret = _monthly_returns(v)
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

    if b is None:
        return out

    rb = b.pct_change().dropna()
    common_r_idx = r.index.intersection(rb.index)
    r2 = r.loc[common_r_idx]
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

    mret_p = _monthly_returns(v)
    mret_b = _monthly_returns(b)
    common_m = mret_p.index.intersection(mret_b.index)
    mret_p = mret_p.loc[common_m]
    mret_b = mret_b.loc[common_m]

    upside_capture = np.nan
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


# ---------------------------------------------------------------------
# Similarity / effective bets
# ---------------------------------------------------------------------
def _effective_bets_from_corr(corr: pd.DataFrame) -> float:
    """
    Effective Number of Bets (ENB) from correlation eigenvalues.

    If funds are identical:
      one big eigenvalue + tiny rest => ENB ~ 1
    If independent:
      eigenvalues more even => ENB approaches k
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

    Uses daily returns of rel prices.
    """
    k = len(funds)
    if k <= 1:
        return 1.0, 0.0

    r = rel[list(funds)].pct_change().dropna(how="any")
    if r.empty or r.shape[0] < 10:
        # Not enough data; don't discard.
        return float(k), 0.0

    corr = r.corr()

    corr_np = corr.to_numpy(dtype=float)
    mask = ~np.eye(k, dtype=bool)
    max_corr = float(np.nanmax(corr_np[mask])) if mask.any() else 0.0

    enb = _effective_bets_from_corr(corr)
    return float(enb), float(max_corr)


# ---------------------------------------------------------------------
# Grid search utilities
# ---------------------------------------------------------------------
def generate_weight_compositions(step: float, k: int, min_weight: float = 0.1) -> List[Tuple[float, ...]]:
    """
    Generate long-only weight vectors of length k that:
    - sum to 1.0
    - each weight >= min_weight
    - step is a discrete grid (e.g. 0.1)

    Implementation uses integer compositions to avoid floating accumulation error.
    """
    s = int(round(1.0 / step))
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

    for ints in rec(s, k):
        if any(i < min_i for i in ints):
            continue
        w = tuple(i / s for i in ints)

        # Skip "degenerate" vectors like (0,0,1) once min_weight is 0.0.
        if max(w) == 1.0:
            continue

        comps.append(w)

    return comps


def _rank_key_within_combo(m: Dict[str, float]) -> Tuple:
    """
    Lexicographic ranking among survivors (all already beat DIS and pass constraints).
    Higher tuple values are better.

    Order: return, then risk-shape/tail behaviour.
    """
    def g(key: str, default: float) -> float:
        v = m.get(key, default)
        try:
            return float(v)
        except Exception:
            return default

    return (
        g("cum_return", -np.inf),
        g("calmar", -np.inf),
        g("sortino", -np.inf),
        g("cvar_95", -np.inf),          # less negative is higher (better)
        -g("ulcer_index", np.inf),      # lower is better
        -g("max_dd_duration_bdays", np.inf),
        -g("downside_capture", np.inf), # lower is better
        g("upside_capture", -np.inf),
        g("information_ratio", -np.inf),
        g("hit_rate_monthly", -np.inf),
        g("worst_rolling_12m", -np.inf),
    )


def search_non_dis_portfolios(
    cfg: Config,
    prices_bday: pd.DataFrame,
    exclude_codes: List[str],
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.Series]:
    """
    Grid search over 1..cfg.max_funds_in_port fund combinations (long-only constant weights).

    Returns:
      df_best: ONE ROW PER FUND TUPLE (funds=...) with best weights for that tuple
      dis_metrics: metrics for DIS series on the aligned window
      dis_series: DIS value series aligned to the price matrix
    """
    cfg = _with_default_dis(cfg)
    dis_codes = list(cfg.dis_weights.keys())
    universe = [c for c in prices_bday.columns if c not in exclude_codes]

    needed_cols = sorted(set(universe + dis_codes))
    p_all = prices_bday[needed_cols].dropna(how="all")
    if p_all.empty:
        return pd.DataFrame(), {}, pd.Series(dtype=float)

    dis_val = portfolio_value_series(p_all, cfg.dis_weights, initial=cfg.initial_investment)
    dis_m = perf_metrics(dis_val, rf_annual=cfg.risk_free_annual, cvar_alpha=cfg.cvar_alpha)

    dis_cum = float(dis_m["cum_return"])
    dis_vol = float(dis_m["ann_vol"])
    dis_mdd = float(dis_m["max_drawdown"])

    vol_limit = dis_vol + cfg.vol_slack
    mdd_limit = dis_mdd - cfg.mdd_slack  # drawdown is negative

    # Precompute relative prices once
    t0 = p_all.index[0]
    rel = p_all / p_all.loc[t0]
    rel_last = rel.iloc[-1]

    def passes_constraints(metrics: Dict[str, float]) -> bool:
        """Stage 1 filter: must beat DIS + stay within risk slack."""
        return (
            (metrics["cum_return"] > dis_cum)
            and (metrics["ann_vol"] <= vol_limit)
            and (metrics["max_drawdown"] >= mdd_limit)
        )

    results: List[Dict[str, object]] = []
    discarded_similar = 0

    # k=1: single-fund candidates
    for a in universe:
        terminal_rel = float(rel_last[a])
        if terminal_rel - 1.0 <= dis_cum:
            continue

        val = cfg.initial_investment * rel[a]
        m = perf_metrics(val, benchmark_val=dis_val, rf_annual=cfg.risk_free_annual, cvar_alpha=cfg.cvar_alpha)

        if passes_constraints(m):
            results.append({
                "k": 1,
                "funds": (a,),
                "weights": (1.0,),
                "effective_bets": 1.0,
                "max_pair_corr": 0.0,
                **m,
            })

    # k>=2: best-per-combination with similarity filter
    for k in range(2, cfg.max_funds_in_port + 1):
        comps = generate_weight_compositions(cfg.weight_step, k, min_weight=cfg.min_weight)
        required_enb = k * cfg.min_effective_bets_ratio

        for funds in combinations(universe, k):
            enb, max_corr = combo_similarity_stats(rel, funds)
            if (max_corr >= cfg.similarity_corr_threshold) or (enb < required_enb):
                discarded_similar += 1
                continue

            best_row: Optional[Dict[str, object]] = None
            best_key: Optional[Tuple] = None
            best_cum = -np.inf

            last_vec = rel_last[list(funds)].to_numpy(dtype=float)
            series_mat = rel[list(funds)].to_numpy(dtype=float)

            for w in comps:
                wv = np.asarray(w, dtype=float)

                # Quick terminal filter before doing full metrics
                terminal_rel = float(np.dot(wv, last_vec))
                if terminal_rel - 1.0 <= dis_cum:
                    continue

                val = pd.Series(cfg.initial_investment * (series_mat @ wv), index=rel.index, name="value")
                m = perf_metrics(val, benchmark_val=dis_val, rf_annual=cfg.risk_free_annual, cvar_alpha=cfg.cvar_alpha)

                if not passes_constraints(m):
                    continue

                if cfg.selection_method == "return":
                    choose = (m["cum_return"] > best_cum)
                else:
                    key = _rank_key_within_combo(m)
                    choose = (best_key is None) or (key > best_key)

                if choose:
                    best_cum = float(m["cum_return"])
                    best_key = _rank_key_within_combo(m)
                    best_row = {
                        "k": k,
                        "funds": funds,
                        "weights": w,
                        "effective_bets": enb,
                        "max_pair_corr": max_corr,
                        **m,
                    }

            if best_row is not None:
                results.append(best_row)

    if verbose and discarded_similar > 0:
        print(
            f"Discarded {discarded_similar} combinations as 'too similar' "
            f"(corr>={cfg.similarity_corr_threshold} or ENB<{cfg.min_effective_bets_ratio:.2f}*k)."
        )

    df = pd.DataFrame(results)
    if df.empty:
        return df, dis_m, dis_val

    df = df.sort_values("cum_return", ascending=False).reset_index(drop=True)
    return df, dis_m, dis_val


# ---------------------------------------------------------------------
# Reporting + ROI plot
# ---------------------------------------------------------------------
def _fmt(x: object) -> str:
    """Pretty formatting for console output."""
    if x is None:
        return "None"
    if isinstance(x, (float, int, np.floating, np.integer)):
        if pd.isna(x):
            return "nan"
        return f"{float(x):.3f}"
    return str(x)


def _cmp_label(port_val: float, dis_val: float, direction: str, tol: float) -> str:
    """Return better/worse/similar given a direction and tolerance."""
    if pd.isna(port_val) or pd.isna(dis_val):
        return "n/a"

    denom = abs(float(dis_val))
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


METRIC_DIRECTION = {
    "cum_return": "higher",
    "cagr": "higher",
    "ann_vol": "lower",
    "max_drawdown": "higher",           # less negative is better
    "max_dd_duration_bdays": "lower",
    "sharpe": "higher",
    "sortino": "higher",
    "calmar": "higher",
    "cvar_95": "higher",                # less negative is better
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


def print_row_with_dis_comparison(title: str, row: pd.Series, dis_ref: Dict[str, float], tol: float) -> None:
    """Print a portfolio row and compare each metric to DIS."""
    print(title)
    print("  funds         :", row.get("funds"))
    print("  weights       :", row.get("weights"))
    print("  effective_bets:", _fmt(row.get("effective_bets")))
    print("  max_pair_corr :", _fmt(row.get("max_pair_corr")))

    for metric, direction in METRIC_DIRECTION.items():
        if metric not in row.index or metric not in dis_ref:
            continue
        pv = float(row[metric]) if not pd.isna(row[metric]) else np.nan
        dv = float(dis_ref[metric]) if not pd.isna(dis_ref[metric]) else np.nan
        label = _cmp_label(pv, dv, direction, tol=tol)
        print(f"  {metric:>22}: {_fmt(pv)}   | DIS: {_fmt(dv)}   => {label}")


def plot_roi_comparison(
    cfg: Config,
    series_map: Dict[str, pd.Series],
    title: str,
) -> None:
    """Two-panel ROI comparison plot, with baseline at 100."""
    fig, (ax_top, ax_bot) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(11, 7),
        gridspec_kw={"height_ratios": [2.2, 1.4]},
    )

    ax_top.axhline(cfg.initial_investment, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)
    ax_bot.axhline(cfg.initial_investment, linestyle="--", color="grey", alpha=0.5, linewidth=1.0)

    for label, s in series_map.items():
        s = s.dropna()
        if s.empty:
            continue

        top_line = ax_top.plot(s.index, s.values, label=label)[0]
        color = top_line.get_color()

        df = pd.DataFrame({"v": s})
        df["ma"] = df["v"].rolling(window=cfg.ma_window, min_periods=cfg.ma_window).mean()
        df["std"] = df["v"].rolling(window=cfg.ma_window, min_periods=cfg.ma_window).std()
        df["bb_upper"] = df["ma"] + cfg.bb_k * df["std"]
        df["bb_lower"] = df["ma"] - cfg.bb_k * df["std"]

        ax_bot.plot(df.index, df["ma"], color=color, label=f"{label} MA")
        ax_bot.plot(df.index, df["bb_upper"], color=color, alpha=cfg.bb_line_alpha, linewidth=0.9)
        ax_bot.plot(df.index, df["bb_lower"], color=color, alpha=cfg.bb_line_alpha, linewidth=0.9)
        ax_bot.fill_between(df.index, df["bb_lower"], df["bb_upper"], color=color, alpha=cfg.bb_fill_alpha)

    ax_top.set_title(title)
    ax_top.set_ylabel("Portfolio Value")
    ax_top.legend(fontsize=8)

    ax_bot.set_title(f"Rolling MA ({cfg.ma_window}) + Bollinger Bands (±{cfg.bb_k}σ)")
    ax_bot.set_xlabel("Date")
    ax_bot.set_ylabel("Value")

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """CLI interface so the script isn't welded to one filesystem."""
    parser = argparse.ArgumentParser(description="HSBC MPF unit-price analysis and DIS comparison.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing downloaded HSBC CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("./merged"), help="Folder for merged outputs.")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD).")
    parser.add_argument("--weight-step", type=float, default=0.1, help="Grid step for weights (e.g. 0.1).")
    parser.add_argument("--max-funds", type=int, default=5, help="Max funds in a candidate portfolio.")
    parser.add_argument("--min-weight", type=float, default=0.1, help="Minimum weight per fund.")
    parser.add_argument("--selection-method", choices=["return", "lexi"], default="lexi", help="Tie-break method.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    results_csv = output_dir / "grid_search_results.csv"

    cfg = Config(
        input_dir=input_dir,
        output_dir=output_dir,
        results_csv=results_csv,
        x_start=pd.Timestamp(args.start),
        x_end=pd.Timestamp(args.end),
        weight_step=float(args.weight_step),
        max_funds_in_port=int(args.max_funds),
        min_weight=float(args.min_weight),
        selection_method=str(args.selection_method),
    )
    cfg = _with_default_dis(cfg)

    if not cfg.input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {cfg.input_dir.resolve()}")

    merged = merge_all_funds(cfg.input_dir)
    filtered = save_per_fund(merged, cfg.output_dir, cfg.x_start, cfg.x_end)

    if not args.no_plots:
        plot_groups(cfg, filtered)

    prices_b = build_price_matrix(filtered, cfg.x_start, cfg.x_end)

    # DIS series on aligned window
    dis_val_full = portfolio_value_series(prices_b, cfg.dis_weights, initial=cfg.initial_investment)
    dis_m_full = perf_metrics(dis_val_full, rf_annual=cfg.risk_free_annual, cvar_alpha=cfg.cvar_alpha)

    print("\nDIS proxy metrics (CAF-only, ~35 y/o) on its available window:")
    for k, v in dis_m_full.items():
        print(f"  {k:>22}: {_fmt(v)}")

    df_best, dis_m_aligned, dis_val_aligned = search_non_dis_portfolios(
        cfg=cfg,
        prices_bday=prices_b,
        exclude_codes=list(cfg.dis_weights.keys()),
        verbose=False,
    )

    if df_best.empty:
        print(
            f"\nNo portfolio beats DIS while staying within risk limits "
            f"(vol +{cfg.vol_slack*100:.0f}pp, drawdown -{cfg.mdd_slack*100:.0f}pp), "
            f"min weight {cfg.min_weight:.1f}, and similarity filter."
        )
        return

    # Save results to CSV (best-per-combination)
    df_out = df_best.copy()
    df_out["funds"] = df_out["funds"].apply(
        lambda x: ";".join(list(x)) if isinstance(x, (tuple, list)) else str(x)
    )
    df_out["weights"] = df_out["weights"].apply(
        lambda x: ";".join([f"{float(w):.3f}" for w in x]) if isinstance(x, (tuple, list)) else str(x)
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(cfg.results_csv, index=False)
    print(f"\nSaved grid-search results to: {cfg.results_csv.resolve()}")

    # DIS reference metrics for comparison printing: benchmark DIS against itself
    dis_ref = perf_metrics(
        dis_val_aligned,
        benchmark_val=dis_val_aligned,
        rf_annual=cfg.risk_free_annual,
        cvar_alpha=cfg.cvar_alpha,
    )

    print("\nDIS metrics aligned to the search window (constraints reference):")
    for k, v in dis_m_aligned.items():
        print(f"  {k:>22}: {_fmt(v)}")

    # -----------------------------------------------------------------
    # Report best candidates by k (number of funds)
    # -----------------------------------------------------------------
    best_by_k: List[pd.Series] = []
    for k in sorted(df_best["k"].unique()):
        bucket = df_best[df_best["k"] == k]
        if bucket.empty:
            continue

        if cfg.selection_method == "return":
            best_idx = bucket["cum_return"].astype(float).idxmax()
        else:
            # Lexicographic ranking among survivors (more robust than raw return)
            best_idx = None
            best_key = None
            for idx, row in bucket.iterrows():
                key = _rank_key_within_combo(row.to_dict())
                if (best_key is None) or (key > best_key):
                    best_key = key
                    best_idx = idx

        best_by_k.append(df_best.loc[best_idx])

    # Print a concise overview first
    print("\nBest portfolio per k (number of funds):")
    for row in best_by_k:
        kk = int(row["k"])
        funds = row["funds"]
        weights = row["weights"]
        print(f"  k={kk}: funds={funds} weights={weights} cum_return={_fmt(row.get('cum_return'))}")

    # Then print full metric-by-metric comparisons
    for row in best_by_k:
        kk = int(row["k"])
        print_row_with_dis_comparison(f"\nBest for k={kk}:", row, dis_ref, tol=cfg.compare_tol)

    # Keep the single best overall for convenience
    best_overall = df_best.sort_values("cum_return", ascending=False).iloc[0]
    print_row_with_dis_comparison("\nBest overall (across all k):", best_overall, dis_ref, tol=cfg.compare_tol)

    # -----------------------------------------------------------------
    # Optional ROI comparison plot: DIS + best-per-k
    # -----------------------------------------------------------------
    if not args.no_plots:
        series_map: Dict[str, pd.Series] = {"DIS proxy (CAF-only)": dis_val_full}

        # Add best-per-k portfolios (kept intentionally small to avoid plot soup)
        for row in best_by_k:
            kk = int(row["k"])
            funds_k = list(row["funds"])
            weights_k = dict(zip(funds_k, row["weights"]))
            label = f"Best k={kk}: {funds_k}"
            series_map[label] = portfolio_value_series(prices_b, weights_k, cfg.initial_investment)

        plot_roi_comparison(cfg=cfg, series_map=series_map, title=f"ROI Comparison (Initial={cfg.initial_investment})")



if __name__ == "__main__":
    main()
