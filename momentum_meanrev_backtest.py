"""
Momentum & Mean Reversion Backtest on S&P 500 stocks
=====================================================

Strategies (per-stock, long/short, equal-weighted across the universe):
  1. Momentum       : moving-average crossover (fast MA vs slow MA)
  2. Mean reversion : rolling Z-score of price (Bollinger-band style)

Pipeline:
  - Pull daily adjusted closes (yfinance) or generate synthetic data (--synthetic)
  - Log returns, signals lagged by 1 day (no look-ahead)
  - Transaction costs + slippage charged on turnover
  - Parameter sensitivity sweeps (lookbacks 10..200)
  - Alpha decay vs. cost level
  - Summary table (CAGR, Vol, Sharpe, Sortino, Max DD, Calmar, Turnover)
  - Equity curves + heatmaps saved to ./output

Usage:
  python momentum_meanrev_backtest.py                      # 50 liquid S&P names, 2010-today
  python momentum_meanrev_backtest.py --n-tickers 500      # full current constituents (slow)
  python momentum_meanrev_backtest.py --synthetic          # offline test
  python momentum_meanrev_backtest.py --cost-bps 10 --slip-bps 5

Backtesting-bias caveats (read before trusting results):
  * Survivorship bias: current S&P 500 constituents are, by definition, survivors.
    Results are optimistically biased. Use point-in-time constituents for real work.
  * Look-ahead bias: avoided here by lagging signals one bar.
  * Data-snooping: the parameter sweep is in-sample. The script reports a
    train/test split (best params chosen on first 60%, evaluated on last 40%).
  * Dividends/splits: handled via adjusted close.
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
OUT_DIR = "output"

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "UNH", "HD", "MA", "XOM", "CVX", "KO", "PEP", "ABBV", "MRK", "WMT",
    "BAC", "PFE", "CSCO", "ORCL", "INTC", "T", "VZ", "DIS", "MCD", "NKE",
    "ADBE", "CRM", "TXN", "QCOM", "HON", "UNP", "IBM", "CAT", "GS", "AXP",
    "MMM", "BA", "GE", "LOW", "SBUX", "AMGN", "MDT", "COST", "LLY", "TMO",
]

def get_sp500_tickers(n: int) -> list[str]:
    try:
        tbl = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = tbl["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        tickers = FALLBACK_TICKERS
    return tickers[:n]


def load_prices_yf(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.dropna(axis=1, thresh=int(0.9 * len(df)))  # drop names with lots of missing data
    return df.ffill().dropna(how="all")


def make_synthetic_prices(n_tickers: int = 40, n_days: int = 2500, seed: int = 7) -> pd.DataFrame:
    """GBM with a market factor plus a weak AR(1)/OU component so signals have something to find."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    mkt = rng.normal(0.0004, 0.01, n_days)
    cols = {}
    for i in range(n_tickers):
        beta = rng.uniform(0.6, 1.4)
        idio = rng.normal(0, 0.012, n_days)
        r = beta * mkt + idio
        for t in range(1, n_days):
            r[t] += -0.03 * r[t - 1]
        cols[f"SYN{i:03d}"] = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame(cols, index=idx)

def momentum_signal(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Long when fast MA > slow MA, short otherwise."""
    if fast >= slow:
        return pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    ma_f = prices.rolling(fast).mean()
    ma_s = prices.rolling(slow).mean()
    sig = np.sign(ma_f - ma_s)
    return sig.where(ma_s.notna())

def meanrev_signal(prices: pd.DataFrame, window: int, entry_z: float = 2.0,
                   exit_z: float = 0.5) -> pd.DataFrame:
    """
    Bollinger/Z-score reversion with hysteresis:
      z < -entry -> long;  z > +entry -> short;  |z| < exit -> flat; otherwise hold.
    """
    mu = prices.rolling(window).mean()
    sd = prices.rolling(window).std()
    z = (prices - mu) / sd

    zv = z.values
    pos = np.zeros_like(zv)
    prev = np.zeros(zv.shape[1])
    for t in range(zv.shape[0]):
        zt = zv[t]
        cur = prev.copy()
        cur[zt < -entry_z] = 1.0
        cur[zt > entry_z] = -1.0
        cur[np.abs(zt) < exit_z] = 0.0
        cur[np.isnan(zt)] = np.nan
        pos[t] = cur
        prev = np.nan_to_num(cur)
    return pd.DataFrame(pos, index=prices.index, columns=prices.columns)

@dataclass
class Result:
    gross: pd.Series
    net: pd.Series
    turnover: pd.Series 

def backtest(prices: pd.DataFrame, signal: pd.DataFrame, cost_bps: float,
             slip_bps: float) -> Result:
    """
    Equal-weight across names that have a valid signal; weights sum to gross exposure 1.
    Costs = (commission + slippage) bps * |change in weight|.
    """
    rets = np.log(prices).diff()  
    pos = signal.shift(1) 
    valid = pos.notna() & rets.notna()
    n_active = valid.sum(axis=1).replace(0, np.nan)

    w = pos.fillna(0.0).div(n_active, axis=0).fillna(0.0)
    gross = (w * rets.fillna(0.0)).sum(axis=1)

    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * (cost_bps + slip_bps) / 1e4
    net = gross - cost
    first = signal.dropna(how="all").index[0] if signal.notna().any().any() else prices.index[0]
    return Result(gross.loc[first:], net.loc[first:], turnover.loc[first:])

def equity_curve(log_ret: pd.Series) -> pd.Series:
    return np.exp(log_ret.cumsum())

def max_drawdown(log_ret: pd.Series) -> float:
    eq = equity_curve(log_ret)
    return float((eq / eq.cummax() - 1).min())

def summarize(r: Result, name: str) -> dict:
    out = {}
    for label, s in (("Gross", r.gross), ("Net", r.net)):
        n = len(s)
        ann_ret = s.mean() * TRADING_DAYS
        ann_vol = s.std() * np.sqrt(TRADING_DAYS)
        downside = s[s < 0].std() * np.sqrt(TRADING_DAYS)
        mdd = max_drawdown(s)
        cagr = float(np.exp(s.sum()) ** (TRADING_DAYS / n) - 1)
        out[label] = {
            "CAGR": cagr,
            "Ann.Vol": ann_vol,
            "Sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
            "Sortino": ann_ret / downside if downside > 0 else np.nan,
            "MaxDD": mdd,
            "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        }
    row = {"Strategy": name}
    for label in ("Gross", "Net"):
        for k, v in out[label].items():
            if label == "Gross" and k not in ("Sharpe", "CAGR"):
                continue
            row[f"{k} ({label})"] = v
    row["Ann.Turnover"] = r.turnover.mean() * TRADING_DAYS
    gs, ns = out["Gross"]["Sharpe"], out["Net"]["Sharpe"]
    row["Sharpe decay"] = (gs - ns) / abs(gs) if gs and not np.isnan(gs) else np.nan
    return row


def sharpe(s: pd.Series) -> float:
    sd = s.std()
    return float(s.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 1e-12 else np.nan

def sweep_momentum(prices, fasts, slows, cost, slip) -> pd.DataFrame:
    grid = pd.DataFrame(index=fasts, columns=slows, dtype=float)
    for f in fasts:
        for s in slows:
            if f >= s:
                continue
            r = backtest(prices, momentum_signal(prices, f, s), cost, slip)
            grid.loc[f, s] = sharpe(r.net)
    return grid

def sweep_meanrev(prices, windows, entries, cost, slip) -> pd.DataFrame:
    grid = pd.DataFrame(index=windows, columns=entries, dtype=float)
    for w in windows:
        for z in entries:
            r = backtest(prices, meanrev_signal(prices, w, z, exit_z=0.5), cost, slip)
            grid.loc[w, z] = sharpe(r.net)
    return grid

def alpha_decay(prices, sig, cost_levels) -> pd.DataFrame:
    rows = []
    for c in cost_levels:
        r = backtest(prices, sig, c, 0.0)
        rows.append({"Cost (bps)": c, "Net Sharpe": sharpe(r.net),
                     "Net CAGR": float(np.exp(r.net.sum()) ** (TRADING_DAYS / len(r.net)) - 1)})
    return pd.DataFrame(rows)

def plot_equity(curves: dict[str, Result], bench: pd.Series, path: str):
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    for name, r in curves.items():
        ax[0].plot(equity_curve(r.net), label=f"{name} (net)")
        ax[0].plot(equity_curve(r.gross), ls=":", alpha=0.5, label=f"{name} (gross)")
    ax[0].plot(equity_curve(bench.reindex(next(iter(curves.values())).net.index).fillna(0)),
               color="k", lw=1, alpha=0.6, label="Equal-wt buy & hold")
    ax[0].set_yscale("log")
    ax[0].set_title("Equity curves (growth of $1, log scale)")
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].grid(alpha=0.3)
    for name, r in curves.items():
        eq = equity_curve(r.net)
        ax[1].plot(eq / eq.cummax() - 1, label=name)
    ax[1].set_title("Drawdown (net)")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def plot_heatmap(grid: pd.DataFrame, title: str, xlabel: str, ylabel: str, path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid.values.astype(float), aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(grid.columns)), [str(c) for c in grid.columns])
    ax.set_yticks(range(len(grid.index)), [str(i) for i in grid.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, label="Net Sharpe")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def plot_decay(dec: dict[str, pd.DataFrame], path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, d in dec.items():
        ax.plot(d["Cost (bps)"], d["Net Sharpe"], marker="o", label=name)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Round-trip-side cost (bps per unit turnover)")
    ax.set_ylabel("Net Sharpe")
    ax.set_title("Alpha decay vs. transaction costs")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tickers", type=int, default=50)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cost-bps", type=float, default=5.0, help="commission/fees per unit turnover")
    ap.add_argument("--slip-bps", type=float, default=2.0, help="slippage per unit turnover")
    ap.add_argument("--synthetic", action="store_true", help="use simulated prices (offline)")
    ap.add_argument("--fast", type=int, default=50)
    ap.add_argument("--slow", type=int, default=200)
    ap.add_argument("--mr-window", type=int, default=20)
    ap.add_argument("--mr-entry", type=float, default=2.0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.synthetic:
        prices = make_synthetic_prices(args.n_tickers)
        print(f"[data] synthetic: {prices.shape[1]} tickers, {len(prices)} days")
    else:
        tickers = get_sp500_tickers(args.n_tickers)
        prices = load_prices_yf(tickers, args.start, args.end)
        print(f"[data] yfinance: {prices.shape[1]} tickers, "
              f"{prices.index[0].date()} -> {prices.index[-1].date()}")

    bench = np.log(prices).diff().mean(axis=1)  

    cost, slip = args.cost_bps, args.slip_bps
    mom_sig = momentum_signal(prices, args.fast, args.slow)
    mr_sig = meanrev_signal(prices, args.mr_window, args.mr_entry)
    mom = backtest(prices, mom_sig, cost, slip)
    mr = backtest(prices, mr_sig, cost, slip)
    combo = Result(
        gross=(mom.gross + mr.gross) / 2,
        net=(mom.net + mr.net) / 2,
        turnover=(mom.turnover + mr.turnover) / 2,
    )

    curves = {
        f"Momentum MA({args.fast}/{args.slow})": mom,
        f"MeanRev Z({args.mr_window},{args.mr_entry})": mr,
        "50/50 Combo": combo,
    }
    rows = [summarize(r, n) for n, r in curves.items()]
    bench_res = Result(bench.dropna(), bench.dropna(), pd.Series(0.0, index=bench.dropna().index))
    rows.append(summarize(bench_res, "Equal-wt Buy&Hold"))
    table = pd.DataFrame(rows).set_index("Strategy")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(f"\n=== Summary (cost={cost}bps, slippage={slip}bps) ===")
    print(table.T)
    table.to_csv(f"{OUT_DIR}/summary_stats.csv")

    plot_equity(curves, bench, f"{OUT_DIR}/equity_curves.png")

    fasts = [10, 20, 30, 50, 75, 100]
    slows = [50, 75, 100, 150, 200]
    mom_grid = sweep_momentum(prices, fasts, slows, cost, slip)
    plot_heatmap(mom_grid, "Momentum: net Sharpe (fast vs slow MA)", "Slow MA", "Fast MA",
                 f"{OUT_DIR}/sweep_momentum.png")
    mom_grid.to_csv(f"{OUT_DIR}/sweep_momentum.csv")

    windows = [10, 20, 40, 60, 100, 150, 200]
    entries = [1.0, 1.5, 2.0, 2.5, 3.0]
    mr_grid = sweep_meanrev(prices, windows, entries, cost, slip)
    plot_heatmap(mr_grid, "Mean reversion: net Sharpe (window vs entry Z)", "Entry |Z|",
                 "Lookback (days)", f"{OUT_DIR}/sweep_meanrev.png")
    mr_grid.to_csv(f"{OUT_DIR}/sweep_meanrev.csv")

    levels = [0, 2, 5, 7.5, 10, 15, 20]
    dec = {"Momentum": alpha_decay(prices, mom_sig, levels),
           "MeanRev": alpha_decay(prices, mr_sig, levels)}
    plot_decay(dec, f"{OUT_DIR}/alpha_decay.png")
    print("\n=== Alpha decay (net Sharpe vs cost bps) ===")
    print(pd.concat({k: v.set_index("Cost (bps)")["Net Sharpe"] for k, v in dec.items()}, axis=1))

    split = prices.index[int(len(prices) * 0.6)]
    p_tr, p_te = prices.loc[:split], prices.loc[split:]
    print(f"\n=== Walk-forward sanity check (train ends {split.date()}) ===")
    mr_tr = sweep_meanrev(p_tr, windows, entries, cost, slip)
    bw, bz = mr_tr.stack().idxmax()
    te = backtest(p_te, meanrev_signal(p_te, bw, bz), cost, slip)
    print(f"MeanRev best in-sample: window={bw}, entry_z={bz}, "
          f"train Sharpe={mr_tr.loc[bw, bz]:.2f} -> test Sharpe={sharpe(te.net):.2f}")
    mom_tr = sweep_momentum(p_tr, fasts, slows, cost, slip)
    bf, bs = mom_tr.stack().idxmax()
    te = backtest(p_te, momentum_signal(p_te, bf, bs), cost, slip)
    print(f"Momentum best in-sample: fast={bf}, slow={bs}, "
          f"train Sharpe={mom_tr.loc[bf, bs]:.2f} -> test Sharpe={sharpe(te.net):.2f}")

    print(f"\nSaved outputs to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
