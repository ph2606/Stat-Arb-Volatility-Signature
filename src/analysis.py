"""Explicit estimators; all returns are decimal and all costs are one-way bps."""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {'2000–2007': ('2000-01-01', '2007-12-31'),
           '2007–2014': ('2007-01-01', '2014-12-31'),
           '2020–2025': ('2020-01-01', '2025-12-31')}


def _prices(prices):
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError('Prices must be a finite, positive one-dimensional array.')
    return values


def annualized_vol(prices, step=1, periods_per_year=252):
    """Nonoverlapping returns, first observation as sampling origin, ddof=1."""
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError('step must be a positive integer')
    returns = np.diff(np.log(_prices(prices)[::step]))
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year / step)) if len(returns) >= 2 else np.nan


def daily_signature(prices, max_step=30):
    rows = []
    p = _prices(prices)
    for k in range(1, max_step + 1):
        phase = np.array([annualized_vol(p[j:], k) for j in range(k)])
        rows.append({'step': k, 'vol': phase[0], 'phase_min': np.nanmin(phase),
                     'phase_max': np.nanmax(phase), 'n_returns': (len(p)-1)//k})
    return pd.DataFrame(rows).set_index('step')


def daily_rolling(prices, window=252, max_step=30):
    """252-return windows; each k-grid ends at that window's latest observation.

    At k=30 use eight nonoverlapping returns over the most recent 240 sessions.
    Never use prices outside the 252-return window or future observations.
    """
    p = _prices(prices)
    logp = np.log(p)
    index = prices.index if isinstance(prices, pd.Series) else pd.RangeIndex(len(p))
    out = pd.DataFrame(index=index[window:])
    ends = np.arange(window, len(p))
    for k in range(1, max_step + 1):
        offsets = np.arange(-(window//k)*k, 1, k)
        r = np.diff(logp[ends[:, None] + offsets[None, :]], axis=1)
        out[k] = np.std(r, axis=1, ddof=1)*np.sqrt(252/k)
    return out


def distribution(frame):
    return pd.DataFrame({'mean': frame.mean(), 'median': frame.median(),
                         'q25': frame.quantile(.25), 'q75': frame.quantile(.75),
                         'n_windows': frame.count()})


def session_prices(hourly, asset):
    """Keep complete sessions and exact hourly boundaries, without filling gaps.

    SPY: Yahoo opens at 09:30,...,15:30 ET give six complete hours.
    FX: configurable observation span; 24-hour sessions use the last bar close.
    A session is kept only if every expected hourly bar/endpoint is present.
    SPY's 15:30–16:00 half hour is never misclassified as an hour.
    """
    df = hourly.copy()
    if 'timestamp' in df:
        df.index = pd.to_datetime(df.pop('timestamp'), utc=True)
    if df.index.tz is None:
        raise ValueError('Intraday timestamps must be timezone aware.')
    if df.index.has_duplicates:
        raise ValueError('Duplicate intraday timestamps.')
    equity = asset.get('kind') == 'equity' or asset.get('symbol') == 'SPY' or asset.get('ticker') == 'SPY'
    tz = 'America/New_York' if equity else asset.get('session_timezone', 'UTC')
    start_hour = 0 if equity else asset.get('session_start_hour', 0)
    sampled_hours = 6 if equity else asset.get('session_hours', 24)
    df = df.sort_index().tz_convert(tz)
    rows, audits = {}, []
    labels = (df.index.tz_localize(None)-pd.Timedelta(hours=start_hour)).date
    for origin, group in df.groupby(labels):
        day = pd.Timestamp(origin)+pd.Timedelta(days=1 if start_hour else 0)
        if pd.Timestamp(day).dayofweek >= 5:
            continue
        start = pd.Timestamp(f'{day.date()} 09:30', tz=tz) if equity else pd.Timestamp(f'{origin} {start_hour:02d}:00', tz=tz)
        expected = pd.date_range(start, periods=sampled_hours+1 if sampled_hours < 24 else 24, freq='h')
        part = group.reindex(expected)
        vals = part['open'].to_numpy(dtype=float)
        valid = np.isfinite(vals).all() and (vals > 0).all()
        if not equity and sampled_hours == 24:
            terminal = part['close'].iloc[-1]
            valid = valid and np.isfinite(terminal) and terminal > 0
            vals = np.r_[vals, terminal]
        audits.append({'date': str(day.date()), 'retained': bool(valid), 'observed_bars': len(group),
                       'expected_endpoints': len(vals)})
        if valid:
            rows[pd.Timestamp(day)] = vals
    return rows, pd.DataFrame(audits)


def intraday_returns(hourly, asset, step_hours):
    sessions, _ = session_prices(hourly, asset)
    records = []
    for day, p in sessions.items():
        r = np.diff(np.log(p[::step_hours]))
        records.extend((day, float(v)) for v in r)
    if not records:
        return pd.Series(dtype=float, name='log_return')
    return pd.Series([v for _, v in records], index=pd.DatetimeIndex([d for d, _ in records]), name='log_return')


def intraday_analysis(hourly, asset, steps=(1, 2, 3, 6), months=6):
    """Exact trailing six-calendar-month windows over complete sessions.

    A window is available only after six months of source history. The number
    of complete sessions varies; exclusions do not stretch the window's dates.
    """
    sessions, audit = session_prices(hourly, asset)
    dates = pd.DatetimeIndex(sorted(sessions))
    raw_dates = pd.to_datetime(audit['date'])
    eligible = dates[dates >= raw_dates.min()+pd.DateOffset(months=months)]
    signature, rolling = [], pd.DataFrame(index=eligible)
    hours = float(asset.get('hours_per_day', asset.get('hours', 6.5 if asset.get('kind') == 'equity' else 24)))
    for k in steps:
        blocks = [np.diff(np.log(sessions[d][::k])) for d in dates]
        all_r = np.concatenate(blocks) if blocks else np.array([])
        scale = np.sqrt(252*hours/k)
        signature.append({'hours': k, 'vol': all_r.std(ddof=1)*scale if len(all_r)>1 else np.nan,
                          'n_returns': len(all_r), 'n_sessions': len(dates)})
        estimates = []
        for end in eligible:
            select = (dates > end-pd.DateOffset(months=months)) & (dates <= end)
            r = np.concatenate([b for b, keep in zip(blocks, select) if keep])
            estimates.append(r.std(ddof=1)*scale if len(r)>1 else np.nan)
        rolling[k] = estimates
    return pd.DataFrame(signature).set_index('hours'), rolling, audit


def backtest(prices, fast=1, slow=5, lookback=252, orientation=1, cost_bps=1,
             notional=100000, initial_equity=100000, dividends=None, signal_prices=None):
    """Inverse-price holdings, lagged signals, fixed capital and netted turnover.

    A row is an execution time. quantity is held AFTER this row's execution;
    gross_pnl is earned by the PREVIOUS row's quantity. The current signal uses
    observations strictly before this execution row, including at slow resets.
    Terminal liquidation is charged. cash earns zero; no return compounding of N.
    """
    p = _prices(prices)
    if fast < 1 or slow <= fast or slow % fast or lookback < 2*slow:
        raise ValueError('Require slow > fast, slow divisible by fast, lookback >= 2*slow.')
    if cost_bps < 0 or initial_equity <= 0 or notional <= 0 or orientation not in (-1, 1):
        raise ValueError('Invalid cost, capital or orientation.')
    sigp = p if signal_prices is None else _prices(signal_prices)
    if len(sigp) != len(p):
        raise ValueError('Signal prices must align with execution prices.')
    div = np.zeros(len(p)) if dividends is None else np.asarray(dividends, dtype=float)
    if len(div) != len(p) or not np.isfinite(div).all():
        raise ValueError('Dividends must align and be finite.')
    if len(p) <= lookback+1:
        raise ValueError('Insufficient signal history.')
    q, sign, equity, slow_p, fast_p = 0., 0., initial_equity, p[lookback+1], p[lookback+1]
    rows = []
    start = lookback+1
    for t in range(start, len(p)):
        gross = q*(p[t]-p[t-1]+div[t]) if t > start else 0.
        old_q = q
        if (t-start) % slow == 0:
            # Exactly lookback returns ending at t-1, before current execution.
            hist = sigp[t-lookback-1:t]
            vf = annualized_vol(hist[-((lookback//fast)*fast+1):], fast)
            vs = annualized_vol(hist[-((lookback//slow)*slow+1):], slow)
            sign = float(orientation*np.sign(vf-vs))
            slow_p = p[t]
        if (t-start) % fast == 0:
            fast_p = p[t]
        q = sign*notional*(1/fast_p-1/slow_p)
        if t == len(p)-1:
            q = 0.  # Liquidate once; no artificial final re-hedge.
        traded = abs(q-old_q)*p[t]
        cost = traded*cost_bps/10000
        net = gross-cost
        ret = net/equity
        equity += net
        rows.append({'quantity': q, 'signal': sign, 'gross_pnl': gross, 'cost': cost,
                     'net_pnl': net, 'equity': equity, 'return': ret,
                     'turnover': traded/initial_equity, 'exposure': q*p[t]/initial_equity})
        if equity <= 0:
            raise RuntimeError('Strategy equity exhausted; stop rather than report invalid returns.')
    return pd.DataFrame(rows, index=prices.index[start:] if isinstance(prices, pd.Series) else pd.RangeIndex(start, len(p)))


def performance(frame, initial_equity=100000):
    r = frame['return'].iloc[1:]
    years = len(r)/252
    curve = np.r_[initial_equity, frame['equity'].to_numpy()]
    drawdown = curve/np.maximum.accumulate(curve)-1
    vol = r.std(ddof=1)*np.sqrt(252)
    return {'total_return': frame['equity'].iloc[-1]/initial_equity-1,
            'cagr': (frame['equity'].iloc[-1]/initial_equity)**(1/years)-1 if years else np.nan,
            'ann_vol': vol, 'sharpe_zero_rate': r.mean()*252/vol if vol else np.nan,
            'max_drawdown': float(drawdown.min()), 'annual_turnover': frame['turnover'].sum()/years,
            'cost_quote_units': frame['cost'].sum(), 'gross_pnl_quote_units': frame['gross_pnl'].sum(),
            'net_pnl_quote_units': frame['net_pnl'].sum(), 'max_abs_exposure': frame['exposure'].abs().max()}
