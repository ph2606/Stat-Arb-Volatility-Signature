"""Independent checks of clock-aligned minute returns and sample comparisons."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from minute_analysis import (matched_five_minute_comparison, minute_blocks,
                             minute_signature)


SPY = {"symbol": "SPY", "kind": "equity", "hours_per_day": 6.5}
FX = {"symbol": "EURUSD=X", "kind": "fx", "hours_per_day": 24}


def minute_day(day, returns=None, *, asset=SPY, base=1, level=100.0):
    equity = asset["kind"] == "equity"
    duration = 360 if equity else 1080
    count = duration // base
    if returns is None:
        returns = np.resize(np.array([0.001, -0.002, 0.003, -0.001]), count)
    assert len(returns) == count
    origin = day + (" 09:30" if equity else " 00:00")
    zone = "America/New_York" if equity else "UTC"
    index = pd.date_range(origin, periods=count + 1, freq=f"{base}min", tz=zone)
    return pd.DataFrame({"open": level * np.exp(np.r_[0.0, np.cumsum(returns)]),
                         "close": level * 1000}, index=index)


def sample_vol_by_hand(returns, minutes, hours):
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    return np.sqrt(variance * 15120 * hours / minutes)


@pytest.mark.parametrize("asset", [SPY, FX], ids=["equity", "fx"])
@pytest.mark.parametrize("base", [1, 5])
def test_minute_signature_matches_independent_log_return_formula(asset, base):
    count = (360 if asset["kind"] == "equity" else 1080) // base
    returns = np.resize(np.array([0.001, -0.002, 0.003, -0.001]), count)
    frame = pd.concat([minute_day("2025-01-06", returns, asset=asset, base=base),
                       minute_day("2025-01-07", returns * 2, asset=asset,
                                  base=base, level=500)])
    signature, audits = minute_signature(frame, asset, base)
    row = signature.set_index("interval_minutes").loc[base]
    expected_returns = np.r_[returns, returns * 2]
    assert row.annualized_vol == pytest.approx(
        sample_vol_by_hand(expected_returns, base, asset["hours_per_day"]), rel=1e-11
    )
    assert row.n_returns == 2 * count
    assert row.n_sessions == 2
    assert row.rejected_blocks == 0
    assert (audits.candidate_blocks == audits.valid_blocks + audits.rejected_blocks).all()
    # Five base intervals aggregate by their log-return sum, not their mean.
    coarse = signature.set_index("interval_minutes").loc[5 if base == 1 else 15]
    width = (5 if base == 1 else 15) // base
    coarse_returns = np.r_[returns.reshape(-1, width).sum(axis=1),
                           (returns * 2).reshape(-1, width).sum(axis=1)]
    assert coarse.annualized_vol == pytest.approx(
        sample_vol_by_hand(coarse_returns, width * base, asset["hours_per_day"]), rel=1e-11
    )


def test_missing_interior_minute_rejects_path_with_present_endpoints():
    complete = minute_day("2025-01-06")
    broken = complete.drop(complete.index[2])
    blocks, audit = minute_blocks(broken, SPY, 1, 5)
    # The missing observation is strictly inside 09:30--09:35. Both endpoints
    # exist, but the coarse return must still fail the source-completeness rule.
    assert complete.index[0] in broken.index and complete.index[5] in broken.index
    assert len(blocks) == 71
    assert blocks.start.iloc[0] == complete.index[5].tz_convert("UTC")
    assert blocks.end.iloc[0] == complete.index[10].tz_convert("UTC")
    expected = np.log(complete.open.iloc[10] / complete.open.iloc[5])
    assert blocks.log_return.iloc[0] == pytest.approx(expected)
    assert audit.valid_blocks.iloc[0] == 71
    assert audit.rejected_blocks.iloc[0] == 1
    assert audit.observed_grid_marks.iloc[0] == 360
    assert audit.expected_grid_marks.iloc[0] == 361


def test_missing_shared_endpoint_rejects_both_adjacent_blocks_without_close_substitution():
    complete = minute_day("2025-01-06")
    broken = complete.drop(complete.index[5])
    # Even an available prior close cannot replace the missing open endpoint.
    broken.loc[complete.index[4], "close"] = complete.open.iloc[5]
    blocks, audit = minute_blocks(broken, SPY, 1, 5)
    assert len(blocks) == 70
    assert blocks.start.iloc[0] == complete.index[10].tz_convert("UTC")
    assert audit.rejected_blocks.iloc[0] == 2


@pytest.mark.parametrize("asset", [SPY, FX], ids=["equity", "fx"])
def test_overnight_and_weekend_jumps_never_enter_returns(asset):
    count = 360 if asset["kind"] == "equity" else 1080
    frames = [minute_day(day, np.zeros(count), asset=asset, level=level)
              for day, level in [("2025-01-03", 100), ("2025-01-04", 10000),
                                 ("2025-01-06", 800)]]
    blocks, _ = minute_blocks(pd.concat(frames), asset, 1, 60)
    assert list(blocks.session.unique()) == [pd.Timestamp("2025-01-03"),
                                            pd.Timestamp("2025-01-06")]
    assert len(blocks) == 2 * count // 60
    np.testing.assert_allclose(blocks.log_return, 0.0, atol=0)
    assert ((blocks.end - blocks.start) == pd.Timedelta(hours=1)).all()


def test_spy_endpoint_is_1530_and_final_half_hour_is_excluded():
    frame = minute_day("2025-01-06")
    extra = pd.DataFrame({"open": 1e8, "close": 1e8}, index=pd.date_range(
        "2025-01-06 15:31", "2025-01-06 16:00", freq="min", tz="America/New_York"))
    blocks, audit = minute_blocks(pd.concat([frame, extra]), SPY, 1, 30)
    assert len(blocks) == 12
    assert blocks.end.iloc[-1] == pd.Timestamp("2025-01-06 15:30", tz="America/New_York")
    expected = np.log(frame.open.iloc[-1] / frame.open.iloc[-31])
    assert blocks.log_return.iloc[-1] == pytest.approx(expected)
    assert audit.observed_grid_marks.iloc[0] == audit.expected_grid_marks.iloc[0] == 361


def test_spy_clock_stays_at_0930_local_across_daylight_saving_transition():
    frames = [minute_day("2025-03-07"), minute_day("2025-03-10")]
    # Inputs commonly arrive as UTC, so convert before constructing sessions.
    frame = pd.concat(frames).tz_convert("UTC")
    blocks, _ = minute_blocks(frame, SPY, 1, 360)
    assert list(blocks.start) == [pd.Timestamp("2025-03-07 14:30", tz="UTC"),
                                  pd.Timestamp("2025-03-10 13:30", tz="UTC")]
    assert ((blocks.end - blocks.start) == pd.Timedelta(hours=6)).all()


@pytest.mark.parametrize("asset, outside_time, zone", [
    (SPY, "16:30", "America/New_York"), (FX, "21:10", "UTC")
], ids=["equity", "fx"])
def test_day_observed_only_outside_selected_span_does_not_inflate_rejections(asset, outside_time, zone):
    complete = minute_day("2025-01-07", asset=asset)
    outside = pd.DataFrame({"open": [100000.0], "close": [100000.0]},
                           index=pd.DatetimeIndex([pd.Timestamp(
                               f"2025-01-06 {outside_time}", tz=zone)]))
    expected_blocks, expected_audit = minute_blocks(complete, asset, 1, 60)
    actual_blocks, actual_audit = minute_blocks(pd.concat([outside, complete]), asset, 1, 60)
    # A request can begin after the sampled session has ended. That source
    # date is outside our sampling support, not a session of missing bars.
    pd.testing.assert_frame_equal(actual_blocks, expected_blocks)
    pd.testing.assert_frame_equal(actual_audit, expected_audit)
    assert actual_audit.rejected_blocks.sum() == 0


def test_observed_terminal_endpoint_keeps_incomplete_session_in_availability_audit():
    complete = minute_day("2025-01-07", asset=FX)
    terminal_only = minute_day("2025-01-06", asset=FX).iloc[[-1]]
    blocks, audit = minute_blocks(pd.concat([terminal_only, complete]), FX, 1, 60)
    assert list(audit.session) == ["2025-01-06", "2025-01-07"]
    first = audit.iloc[0]
    assert first.observed_grid_marks == 1
    assert first.valid_blocks == 0
    assert first.candidate_blocks == first.rejected_blocks == 18
    assert list(blocks.session.unique()) == [pd.Timestamp("2025-01-07")]


def test_duplicate_minute_timestamps_are_rejected():
    frame = minute_day("2025-01-06")
    with pytest.raises(ValueError, match="Duplicate"):
        minute_blocks(pd.concat([frame, frame.iloc[[0]]]), SPY, 1, 5)


@pytest.mark.parametrize("as_column", [False, True])
def test_timezoneless_minute_timestamps_are_rejected(as_column):
    frame = minute_day("2025-01-06").tz_localize(None)
    if as_column:
        frame = frame.rename_axis("timestamp").reset_index()
        frame["timestamp"] = frame.timestamp.astype(str)
    with pytest.raises(ValueError, match="timezone"):
        minute_blocks(frame, SPY, 1, 5)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_invalid_observed_minute_prices_are_rejected(invalid):
    frame = minute_day("2025-01-06")
    frame.iloc[3, frame.columns.get_loc("open")] = invalid
    with pytest.raises(ValueError, match="finite and positive"):
        minute_blocks(frame, SPY, 1, 5)


def test_resolution_comparison_uses_only_identical_valid_return_intervals():
    one = minute_day("2025-01-06")
    native = one.iloc[::5].copy()
    # Alter native marks so comparing a provider's unmatched full series would
    # produce different volatilities from the true common-interval comparison.
    native["open"] *= np.exp(np.resize([0.0, 0.002, -0.001], len(native)))
    one = one.drop(one.index[2])  # Exclude 09:30--09:35 in 1-minute data only.
    native = native.drop(native.index[2])  # Exclude both 09:35--09:45 blocks.
    extra = minute_day("2025-01-07", base=5, level=500)
    actual = matched_five_minute_comparison(one, pd.concat([native, extra]), SPY)
    # The surviving intersection starts at 09:45 and runs to 15:30. The extra
    # day in the native feed must have no effect on either comparison sample.
    starts = pd.date_range("2025-01-06 09:45", "2025-01-06 15:25",
                           freq="5min", tz="America/New_York")
    ends = starts + pd.Timedelta(minutes=5)
    one_returns = np.log(one.open.loc[ends].to_numpy() / one.open.loc[starts].to_numpy())
    native_returns = np.log(native.open.loc[ends].to_numpy() / native.open.loc[starts].to_numpy())
    assert actual["matched_returns"] == 69
    assert actual["matched_sessions"] == 1
    assert actual["vol_from_1m"] == pytest.approx(sample_vol_by_hand(one_returns, 5, 6.5))
    assert actual["vol_native_5m"] == pytest.approx(sample_vol_by_hand(native_returns, 5, 6.5))
    assert actual["mean_abs_return_difference_bps"] == pytest.approx(
        np.mean(np.abs(one_returns - native_returns)) * 10000
    )
    assert actual["first_start_utc"] == starts[0].tz_convert("UTC").isoformat()
    assert actual["last_end_utc"] == ends[-1].tz_convert("UTC").isoformat()
