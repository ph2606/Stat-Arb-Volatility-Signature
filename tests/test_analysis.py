"""Independent checks of estimator conventions and executable trade accounting."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import annualized_vol, backtest, daily_rolling, intraday_analysis, intraday_returns


def price_series(log_returns, start="2020-01-02"):
    """A controlled positive path with a known sequence of log returns."""
    values = 100.0 * np.exp(np.r_[0.0, np.cumsum(log_returns)])
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_sample_standard_deviation_and_annualization_by_hand():
    prices = pd.Series([100.0, 110.0, 99.0, 108.9])
    returns = np.log([1.1, 0.9, 1.1])
    mean = sum(returns) / 3
    expected = np.sqrt(sum((returns - mean) ** 2) / 2) * np.sqrt(252)
    assert annualized_vol(prices) == pytest.approx(expected, rel=1e-12)


def test_multiday_sampling_uses_nonoverlapping_returns_and_correct_clock():
    prices = pd.Series([100.0, 101.0, 110.0, 107.0, 99.0, 101.0, 108.9])
    returns = np.log([1.1, 0.9, 1.1])
    expected = np.std(returns, ddof=1) * np.sqrt(252 / 2)
    assert annualized_vol(prices, step=2) == pytest.approx(expected, rel=1e-12)


def test_flat_prices_have_zero_volatility():
    prices = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=401))
    for step in [1, 5, 30]:
        assert annualized_vol(prices, step=step) == pytest.approx(0.0, abs=1e-15)


def test_rolling_estimates_do_not_use_future_prices():
    rng = np.random.default_rng(4711)
    prices = price_series(rng.normal(0.0002, 0.01, 420))
    prefix = prices.iloc[:350]
    whole = daily_rolling(prices, window=252, max_step=30)
    partial = daily_rolling(prefix, window=252, max_step=30)
    pd.testing.assert_frame_equal(whole.reindex(partial.index), partial)


def test_constant_price_backtest_never_creates_pnl_or_cost():
    prices = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=320))
    result = backtest(prices, cost_bps=10)
    assert np.allclose(result["gross_pnl"], 0.0)
    assert np.allclose(result["cost"], 0.0)
    assert np.allclose(result["net_pnl"], 0.0)
    assert np.allclose(result["quantity"], 0.0)
    assert np.allclose(result["equity"], 100_000.0)


def test_positions_and_pnl_are_unchanged_by_future_prices():
    rng = np.random.default_rng(9001)
    prices = price_series(rng.normal(0.0001, 0.013, 430))
    prefix = prices.iloc[:380]
    whole = backtest(prices, lookback=252, cost_bps=2)
    partial = backtest(prefix, lookback=252, cost_bps=2)
    # The shorter run liquidates at its own final observation. Its earlier
    # decisions and gains must agree with the unrestricted run exactly.
    columns = ["quantity", "signal", "gross_pnl", "cost", "net_pnl", "equity"]
    pd.testing.assert_frame_equal(
        whole.reindex(partial.index).iloc[:-1][columns],
        partial.iloc[:-1][columns],
    )


def test_reversed_orientation_has_opposite_gross_pnl():
    rng = np.random.default_rng(501)
    prices = price_series(rng.normal(0.0004, 0.012, 500))
    first = backtest(prices, orientation=1, cost_bps=3)
    opposite = backtest(prices, orientation=-1, cost_bps=3)
    assert first["gross_pnl"].abs().sum() > 0
    np.testing.assert_allclose(first["gross_pnl"], -opposite["gross_pnl"], atol=1e-10)
    np.testing.assert_allclose(first["cost"], opposite["cost"], atol=1e-10)
    np.testing.assert_allclose(
        first["net_pnl"] + opposite["net_pnl"], -2 * first["cost"], atol=1e-10
    )


@pytest.mark.parametrize("fast, slow", [(1, 5), (2, 6)])
def test_complete_blocks_equal_difference_of_simple_return_sums(fast, slow):
    """Test the exact discrete identity, independently of trade-by-trade code.

    This catches a tempting use of the paper's noncausal equation (15), whose
    same-interval reciprocal-price change has an entirely different payoff.
    """
    rng = np.random.default_rng(1901)
    prices = price_series(rng.normal(0.0003, 0.015, 160))
    notional = 25_000.0
    result = backtest(prices, fast=fast, slow=slow, lookback=60,
                      notional=notional, cost_bps=0)
    executed_prices = prices.reindex(result.index)
    tested = 0
    for start in range(0, len(result) - slow, slow):
        block_prices = executed_prices.iloc[start:start + slow + 1:fast].to_numpy()
        fine_simple_returns = block_prices[1:] / block_prices[:-1] - 1
        coarse_simple_return = block_prices[-1] / block_prices[0] - 1
        expected = (result["signal"].iloc[start] * notional
                    * (fine_simple_returns.sum() - coarse_simple_return))
        actual = result["gross_pnl"].iloc[start + 1:start + slow + 1].sum()
        assert actual == pytest.approx(expected, abs=2e-10)
        tested += 1
    assert tested > 5


def test_self_financing_costs_and_terminal_liquidation():
    rng = np.random.default_rng(321)
    prices = price_series(rng.normal(0.0002, 0.01, 318))
    initial_equity = 100_000.0
    cost_bps = 4.0
    result = backtest(prices, cost_bps=cost_bps, initial_equity=initial_equity)
    executed_prices = prices.reindex(result.index)
    previous_quantity = result["quantity"].shift(1, fill_value=0)
    expected_cost = ((result["quantity"] - previous_quantity).abs()
                     * executed_prices * cost_bps / 10_000)
    np.testing.assert_allclose(result["cost"], expected_cost, atol=1e-12)
    assert result["quantity"].iloc[-1] == 0.0
    assert abs(result["quantity"].iloc[-2]) > 0.0
    assert result["cost"].iloc[-1] == pytest.approx(
        abs(result["quantity"].iloc[-2]) * executed_prices.iloc[-1] * cost_bps / 10_000
    )
    np.testing.assert_allclose(
        result["equity"], initial_equity + result["net_pnl"].cumsum(), atol=1e-8
    )
    np.testing.assert_allclose(
        result["net_pnl"], result["gross_pnl"] - result["cost"], atol=1e-12
    )


def equity_hours(day, starting_price=100.0):
    timestamps = pd.date_range(day + " 09:30", periods=7, freq="h",
                               tz="America/New_York")
    # Deliberately unrealistic final close makes incorrect inclusion of the
    # 15:30--16:00 half-hour endpoint easy to detect.
    return pd.DataFrame({"open": starting_price * np.arange(100, 107) / 100,
                         "close": starting_price * 3.0}, index=timestamps)


def test_intraday_equity_excludes_overnight_jump_and_partial_final_hour():
    first = equity_hours("2025-01-06", 100)
    second = equity_hours("2025-01-07", 300)
    actual = intraday_returns(pd.concat([first, second]),
                              {"symbol": "SPY", "kind": "equity", "hours_per_day": 6.5},
                              step_hours=1)
    expected_one_day = np.log(np.arange(101, 107) / np.arange(100, 106))
    assert len(actual) == 12
    np.testing.assert_allclose(actual.to_numpy(), np.tile(expected_one_day, 2), atol=1e-14)


def test_missing_intraday_hour_is_never_bridged():
    broken = equity_hours("2025-01-06").drop(
        pd.Timestamp("2025-01-06 12:30", tz="America/New_York")
    )
    complete = equity_hours("2025-01-07", 300)
    actual = intraday_returns(pd.concat([broken, complete]),
                              {"symbol": "SPY", "kind": "equity"}, step_hours=2)
    expected = np.log(np.array([102 / 100, 104 / 102, 106 / 104]))
    assert list(actual.index.unique()) == [pd.Timestamp("2025-01-07")]
    np.testing.assert_allclose(actual.to_numpy(), expected, atol=1e-14)


def test_fx_four_hour_returns_end_at_midnight_without_weekend_bridging():
    days = []
    for day, level in [("2025-01-03", 1.0), ("2025-01-06", 1.5)]:
        index = pd.date_range(day, periods=24, freq="h", tz="UTC")
        days.append(pd.DataFrame({"open": level * np.exp(np.arange(24) * 0.001),
                                  "close": level * np.exp(np.arange(1, 25) * 0.001)},
                                 index=index))
    actual = intraday_returns(pd.concat(days),
                              {"symbol": "EURUSD=X", "kind": "fx", "hours_per_day": 24},
                              step_hours=4)
    assert len(actual) == 12
    np.testing.assert_allclose(actual.to_numpy(), 0.004, atol=1e-14)


def test_signal_coarse_grid_ends_at_latest_completed_observation():
    # Twelve historical returns: ten zeros followed by two 10% log returns.
    # A left-aligned five-day grid misses BOTH recent moves and says fast > slow.
    # Ending both grids yesterday gives coarse returns [0, .20], so slow > fast.
    historical_returns = np.r_[np.zeros(10), 0.1, 0.1]
    prices = price_series(np.r_[historical_returns, 0.001, 0.002, -0.001])
    fast_vol = np.std(historical_returns, ddof=1) * np.sqrt(252)
    slow_vol = np.std([0.0, 0.2], ddof=1) * np.sqrt(252 / 5)
    assert slow_vol > fast_vol
    result = backtest(prices, lookback=12, fast=1, slow=5, cost_bps=0)
    assert result["signal"].iloc[0] == -1.0
    # Today's execution price may affect sizing, but not yesterday's signal.
    changed = prices.copy()
    changed.iloc[13:] *= 2
    changed_result = backtest(changed, lookback=12, fast=1, slow=5, cost_bps=0)
    assert changed_result["signal"].iloc[0] == result["signal"].iloc[0]


def test_intraday_rolling_uses_exact_six_months_without_future_information():
    profiles = {}
    frames = []
    for day in pd.bdate_range("2025-01-02", "2025-07-10"):
        # Fewer than 126 retained sessions must not extend a six-month window.
        if day.dayofweek == 4:
            continue
        returns = np.array([0.002, -0.001, 0.003, -0.002, 0.001, -0.003])
        returns *= 1 + day.day / 31
        if day == pd.Timestamp("2025-01-02"):
            returns *= 100  # Exactly on the open left boundary at July 2.
        if day > pd.Timestamp("2025-07-02"):
            returns *= 50  # A large future shock must not leak backwards.
        frame = equity_hours(day.strftime("%Y-%m-%d"))
        frame["open"] = 100 * np.exp(np.r_[0.0, np.cumsum(returns)])
        frames.append(frame)
        profiles[day] = returns
    hourly = pd.concat(frames)
    asset = {"symbol": "SPY", "kind": "equity", "hours_per_day": 6.5}
    _, whole, _ = intraday_analysis(hourly, asset)
    end = pd.Timestamp("2025-07-02")
    prefix = hourly[hourly.index.tz_localize(None).normalize() <= end]
    _, partial, _ = intraday_analysis(prefix, asset)
    assert whole.index[0] == end
    assert len(profiles) < 126
    pd.testing.assert_frame_equal(whole.reindex(partial.index), partial)
    observed = np.concatenate([r for day, r in profiles.items()
                               if pd.Timestamp("2025-01-02") < day <= end])
    expected = np.std(observed, ddof=1) * np.sqrt(252 * 6.5)
    assert whole.loc[end, 1] == pytest.approx(expected, rel=1e-12)


def test_fx_common_eighteen_hour_window_retains_short_friday():
    frames = []
    for day, bars in [("2025-01-02", 24), ("2025-01-03", 22)]:
        index = pd.date_range(day, periods=bars, freq="h", tz="UTC")
        frames.append(pd.DataFrame({"open": np.exp(np.arange(bars) * 0.001),
                                     "close": np.exp(np.arange(1, bars + 1) * 0.001)},
                                    index=index))
    actual = intraday_returns(pd.concat(frames),
                              {"kind": "fx", "hours_per_day": 24,
                               "session_timezone": "UTC", "session_hours": 18},
                              step_hours=6)
    assert len(actual) == 6
    assert list(actual.index.unique()) == [pd.Timestamp("2025-01-02"),
                                          pd.Timestamp("2025-01-03")]
    np.testing.assert_allclose(actual.to_numpy(), 0.006, atol=1e-14)
