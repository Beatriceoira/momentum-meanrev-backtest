"""
Tests for momentum_meanrev_backtest.py

Run from the project root:
    pip install pytest
    pytest -v

All tests are offline (synthetic or hand-built prices); no network needed.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import momentum_meanrev_backtest as bt  # noqa: E402

def make_prices(n=300, k=4, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    r = rng.normal(0.0003, 0.01, (n, k))
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=[f"T{i}" for i in range(k)])


@pytest.fixture
def prices():
    return make_prices()

def test_synthetic_prices_shape_and_determinism():
    a = bt.make_synthetic_prices(n_tickers=5, n_days=200, seed=1)
    b = bt.make_synthetic_prices(n_tickers=5, n_days=200, seed=1)
    assert a.shape == (200, 5)
    assert (a > 0).all().all()
    pd.testing.assert_frame_equal(a, b)

def test_momentum_signal_values_and_warmup(prices):
    sig = bt.momentum_signal(prices, 10, 50)
    assert sig.iloc[:49].isna().all().all()  
    vals = np.unique(sig.dropna().values)
    assert set(vals) <= {-1.0, 0.0, 1.0}


def test_momentum_signal_long_in_uptrend_short_in_downtrend():
    idx = pd.bdate_range("2020-01-01", periods=120)
    up = pd.DataFrame({"A": np.linspace(100, 200, 120)}, index=idx)
    down = pd.DataFrame({"A": np.linspace(200, 100, 120)}, index=idx)
    assert bt.momentum_signal(up, 5, 20)["A"].iloc[-1] == 1.0
    assert bt.momentum_signal(down, 5, 20)["A"].iloc[-1] == -1.0


def test_momentum_signal_invalid_windows_returns_nan(prices):
    assert bt.momentum_signal(prices, 50, 50).isna().all().all()
    assert bt.momentum_signal(prices, 100, 50).isna().all().all()


def _spiky_series(spike):
    """30 days of tiny alternating noise around 100, then a level shift."""
    base = 100 + 0.1 * np.array([1, -1] * 15)
    return np.concatenate([base, [spike] * 5])


def test_meanrev_shorts_after_upward_spike():
    idx = pd.bdate_range("2020-01-01", periods=35)
    p = pd.DataFrame({"A": _spiky_series(110.0)}, index=idx)
    sig = bt.meanrev_signal(p, window=20, entry_z=2.0, exit_z=0.5)
    assert sig["A"].iloc[30] == -1.0


def test_meanrev_longs_after_downward_spike():
    idx = pd.bdate_range("2020-01-01", periods=35)
    p = pd.DataFrame({"A": _spiky_series(90.0)}, index=idx)
    sig = bt.meanrev_signal(p, window=20, entry_z=2.0, exit_z=0.5)
    assert sig["A"].iloc[30] == 1.0


def test_meanrev_signal_values_valid(prices):
    sig = bt.meanrev_signal(prices, 20, 2.0)
    assert set(np.unique(sig.dropna().values)) <= {-1.0, 0.0, 1.0}
    assert sig.iloc[:19].isna().all().all()


def test_meanrev_hysteresis_holds_between_exit_and_entry():
    """Z between exit (0.5) and entry (2.0) must keep the existing position."""
    idx = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(3)
    p = pd.DataFrame({"A": 100 + rng.normal(0, 1, 60).cumsum() * 0.2}, index=idx)
    strict = bt.meanrev_signal(p, 20, 1.5, exit_z=1.4)   
    wide = bt.meanrev_signal(p, 20, 1.5, exit_z=0.2)     
    flips = lambda s: s["A"].dropna().diff().abs().sum()
    assert flips(wide) <= flips(strict)

def test_no_lookahead_signal_is_lagged_one_day(prices):
    """A signal built from TODAY's return (sign of r_t) must not earn r_t.
    Without the one-day lag it would capture |r_t| every day (Sharpe >> 10);
    with the lag it only sees tomorrow's independent return (Sharpe ~ 0)."""
    rets = np.log(prices).diff()
    same_day = np.sign(rets)
    res = bt.backtest(prices, same_day, 0.0, 0.0)
    assert abs(bt.sharpe(res.gross)) < 3

    leaked = np.sign(rets.shift(-1))
    leak_res = bt.backtest(prices, leaked, 0.0, 0.0)
    assert bt.sharpe(leak_res.gross) > 10


def test_constant_long_matches_equal_weight_buy_and_hold(prices):
    sig = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    res = bt.backtest(prices, sig, 0.0, 0.0)
    bench = np.log(prices).diff().mean(axis=1).loc[res.gross.index]
    common = res.gross.index[1:]
    np.testing.assert_allclose(res.gross.loc[common], bench.loc[common], atol=1e-12)


def test_zero_costs_make_net_equal_gross(prices):
    sig = bt.momentum_signal(prices, 10, 50)
    res = bt.backtest(prices, sig, 0.0, 0.0)
    pd.testing.assert_series_equal(res.gross, res.net)


def test_costs_never_increase_returns(prices):
    sig = bt.meanrev_signal(prices, 20, 1.5)
    res = bt.backtest(prices, sig, 5.0, 2.0)
    assert (res.net <= res.gross + 1e-15).all()
    assert res.net.sum() < res.gross.sum()


def test_cost_equals_turnover_times_bps(prices):
    sig = bt.momentum_signal(prices, 10, 50)
    res = bt.backtest(prices, sig, 5.0, 3.0)
    expected = res.turnover * 8.0 / 1e4
    np.testing.assert_allclose((res.gross - res.net).values, expected.values, atol=1e-15)


def test_constant_position_has_single_entry_turnover(prices):
    sig = pd.DataFrame(1.0, index=prices.index, columns=prices.columns)
    res = bt.backtest(prices, sig, 5.0, 0.0)
    assert res.turnover.sum() == pytest.approx(1.0, abs=1e-9)


def test_doubling_costs_doubles_cost_drag(prices):
    sig = bt.meanrev_signal(prices, 20, 1.5)
    r1 = bt.backtest(prices, sig, 5.0, 0.0)
    r2 = bt.backtest(prices, sig, 10.0, 0.0)
    drag1 = (r1.gross - r1.net).sum()
    drag2 = (r2.gross - r2.net).sum()
    assert drag2 == pytest.approx(2 * drag1, rel=1e-9)

def test_max_drawdown_known_value():
    log_ret = pd.Series(np.log([1.10, 0.80, 1.05]))
    assert bt.max_drawdown(log_ret) == pytest.approx(-0.2, abs=1e-9)


def test_max_drawdown_monotonic_gain_is_zero():
    assert bt.max_drawdown(pd.Series([0.01] * 50)) == pytest.approx(0.0)


def test_sharpe_zero_vol_is_nan():
    assert np.isnan(bt.sharpe(pd.Series([0.001] * 100)))


def test_sharpe_known_value():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.001, 0.01, 5000))
    expected = s.mean() / s.std() * np.sqrt(252)
    assert bt.sharpe(s) == pytest.approx(expected)


def test_equity_curve_starts_near_one_and_compounds():
    r = pd.Series([0.01, 0.01, -0.02])
    eq = bt.equity_curve(r)
    assert eq.iloc[0] == pytest.approx(np.exp(0.01))
    assert eq.iloc[-1] == pytest.approx(np.exp(0.0))


def test_summarize_contains_expected_columns_and_decay(prices):
    sig = bt.meanrev_signal(prices, 20, 1.5)
    res = bt.backtest(prices, sig, 5.0, 2.0)
    row = bt.summarize(res, "test")
    for key in ("Strategy", "Sharpe (Gross)", "Sharpe (Net)", "Sortino (Net)",
                "MaxDD (Net)", "CAGR (Net)", "Ann.Turnover", "Sharpe decay"):
        assert key in row
    assert row["MaxDD (Net)"] <= 0
    assert row["Ann.Turnover"] > 0


def test_summarize_net_sharpe_not_above_gross_when_positive_costs(prices):
    sig = bt.meanrev_signal(prices, 20, 1.5)
    res = bt.backtest(prices, sig, 10.0, 5.0)
    row = bt.summarize(res, "t")
    assert row["Sharpe (Net)"] <= row["Sharpe (Gross)"]

def test_sweep_momentum_shape_and_invalid_cells_are_nan(prices):
    grid = bt.sweep_momentum(prices, [10, 50], [20, 50, 100], 5.0, 2.0)
    assert grid.shape == (2, 3)
    assert np.isnan(grid.loc[50, 20])   # fast >= slow
    assert np.isnan(grid.loc[50, 50])   # fast == slow
    assert not np.isnan(grid.loc[10, 50])


def test_sweep_meanrev_shape(prices):
    grid = bt.sweep_meanrev(prices, [10, 20], [1.0, 2.0], 5.0, 2.0)
    assert grid.shape == (2, 2)
    assert grid.notna().all().all()


def test_alpha_decay_net_sharpe_falls_as_costs_rise(prices):
    sig = bt.meanrev_signal(prices, 20, 1.0)
    dec = bt.alpha_decay(prices, sig, [0, 5, 10, 20])
    sharpes = dec["Net Sharpe"].values
    assert (np.diff(sharpes) <= 1e-12).all()
    assert list(dec["Cost (bps)"]) == [0, 5, 10, 20]

def test_main_synthetic_end_to_end(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["prog", "--synthetic", "--n-tickers", "12"])
    bt.main()
    out = tmp_path / "output"
    for name in ("summary_stats.csv", "equity_curves.png", "sweep_momentum.png",
                 "sweep_meanrev.png", "alpha_decay.png",
                 "sweep_momentum.csv", "sweep_meanrev.csv"):
        assert (out / name).exists(), f"missing {name}"
    summary = pd.read_csv(out / "summary_stats.csv", index_col=0)
    assert len(summary) == 4  
    assert "Summary" in capsys.readouterr().out
