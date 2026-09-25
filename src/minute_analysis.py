"""Minute-resolution estimates on exact, observed, within-session price paths."""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTE_STEPS = {1: (1, 2, 3, 5, 10, 15, 30, 60, 120, 180, 360),
                5: (5, 10, 15, 30, 60, 120, 180, 360)}


def minute_blocks(frame, asset, base_minutes, interval_minutes):
    """Return valid nonoverlapping blocks and daily availability diagnostics.

    Yahoo timestamps mark bar starts; the price at each endpoint is its open.
    Require every base-grid open in the block, including both endpoints. A gap
    invalidates that block, not the entire day; later blocks keep their original
    session-open anchor. No filling, overnight bridging or partial end blocks.
    """
    if base_minutes not in MINUTE_STEPS or interval_minutes < base_minutes or interval_minutes % base_minutes:
        raise ValueError('Interval must be an integer multiple of the supported base resolution.')
    df = frame.copy()
    if 'timestamp' in df:
        # Reject timezone-less source strings rather than assuming a timezone.
        parsed = pd.to_datetime(df.pop('timestamp'))
        if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
            raise ValueError('Minute timestamps must explicitly include a timezone.')
        df.index = pd.DatetimeIndex(parsed)
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise ValueError('Minute timestamps must be timezone aware.')
    if df.index.has_duplicates:
        raise ValueError('Duplicate minute timestamps.')
    if not np.isfinite(df['open']).all() or (df['open'] <= 0).any():
        raise ValueError('Observed minute opens must be finite and positive.')
    equity = asset.get('kind') == 'equity' or asset.get('symbol') == 'SPY'
    zone = 'America/New_York' if equity else 'UTC'
    duration = 360 if equity else 1080
    if duration % interval_minutes:
        raise ValueError('Sampling interval must divide the selected session span exactly.')
    df = df.sort_index().tz_convert(zone)
    records, audits = [], []
    stride = interval_minutes//base_minutes
    for day, group in df.groupby(df.index.date):
        if pd.Timestamp(day).dayofweek >= 5:
            continue
        start = pd.Timestamp(f'{day} '+('09:30' if equity else '00:00'), tz=zone)
        grid = pd.date_range(start, periods=duration//base_minutes+1, freq=f'{base_minutes}min')
        prices = group['open'].reindex(grid).to_numpy(dtype=float)
        # Bars entirely outside the selected span do not establish an observed
        # session (for example a first FX date beginning at 21:10 UTC).
        if not np.isfinite(prices).any():
            continue
        valid_count = 0
        for left in range(0, len(prices)-1, stride):
            right = left+stride
            path = prices[left:right+1]
            if not np.isfinite(path).all():
                continue
            records.append({'session': pd.Timestamp(day), 'start': grid[left].tz_convert('UTC'),
                            'end': grid[right].tz_convert('UTC'),
                            'log_return': float(np.log(prices[right]/prices[left]))})
            valid_count += 1
        candidates = duration//interval_minutes
        audits.append({'session': str(day), 'candidate_blocks': candidates,
                       'valid_blocks': valid_count, 'rejected_blocks': candidates-valid_count,
                       'observed_grid_marks': int(np.isfinite(prices).sum()),
                       'expected_grid_marks': len(prices)})
    return pd.DataFrame(records, columns=['session','start','end','log_return']), pd.DataFrame(audits)


def minute_signature(frame, asset, base_minutes):
    rows, audits = [], []
    for minutes in MINUTE_STEPS[base_minutes]:
        blocks, audit = minute_blocks(frame, asset, base_minutes, minutes)
        if len(blocks) < 2:
            raise ValueError(f'Not enough valid {minutes}-minute returns for this sample.')
        vol = blocks.log_return.std(ddof=1)*np.sqrt(15120*asset['hours_per_day']/minutes)
        rows.append({'base_minutes': base_minutes, 'interval_minutes': minutes,
                     'annualized_vol': float(vol), 'n_returns': len(blocks),
                     'n_sessions': blocks.session.nunique(),
                     'candidate_blocks': int(audit.candidate_blocks.sum()),
                     'rejected_blocks': int(audit.rejected_blocks.sum()),
                     'first_return_start_utc': blocks.start.min().isoformat(),
                     'last_return_end_utc': blocks.end.max().isoformat()})
        audits.append(audit.assign(base_minutes=base_minutes, interval_minutes=minutes))
    return pd.DataFrame(rows), pd.concat(audits, ignore_index=True)


def matched_five_minute_comparison(one_minute, five_minute, asset):
    """Compare equal 5-minute intervals on the intersection of valid blocks."""
    one, _ = minute_blocks(one_minute, asset, 1, 5)
    five, _ = minute_blocks(five_minute, asset, 5, 5)
    matched = one.merge(five, on=['start','end'], suffixes=('_from_1m','_native_5m'), validate='one_to_one')
    if len(matched) < 2:
        raise ValueError('Insufficient common valid blocks for the source-resolution comparison.')
    scale = np.sqrt(15120*asset['hours_per_day']/5)
    a, b = matched.log_return_from_1m, matched.log_return_native_5m
    return {'matched_returns':len(matched), 'matched_sessions':matched.session_from_1m.nunique(),
            'first_start_utc':matched.start.min().isoformat(), 'last_end_utc':matched.end.max().isoformat(),
            'vol_from_1m':a.std(ddof=1)*scale, 'vol_native_5m':b.std(ddof=1)*scale,
            'correlation':a.corr(b), 'mean_abs_return_difference_bps':(a-b).abs().mean()*10000}
