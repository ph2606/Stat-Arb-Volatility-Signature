"""Reproduce every numerical result and figure from downloaded local data."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from src.analysis import (PERIODS, daily_signature, daily_rolling, distribution,
                          intraday_analysis, backtest, performance)

OUT = ROOT/'outputs'
FIG = ROOT/'report'/'figures'
ASSETS = {'SPY': {'symbol':'SPY','kind':'equity','hours_per_day':6.5,'label':'SPY'},
          'USDJPY': {'symbol':'JPY=X','kind':'fx','hours_per_day':24,'label':'USD/JPY','session_hours':18},
          'EURUSD': {'symbol':'EURUSD=X','kind':'fx','hours_per_day':24,'label':'EUR/USD','session_hours':18}}
COLORS = {'SPY':'#167D8D', 'USDJPY':'#AD6A39', 'EURUSD':'#66589C'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
                         'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#C4CED5','axes.labelcolor':'#18354A','text.color':'#18354A',
                         'axes.grid':True,'grid.alpha':.2,'figure.facecolor':'white',
                         'savefig.facecolor':'white','legend.frameon':False})


def save(fig, name):
    fig.savefig(FIG/f'{name}.png', dpi=190, bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def verify_input_manifest(root=ROOT):
    """Reject changed or missing inputs before any result files are overwritten."""
    root = Path(root)
    manifest = json.loads((root/'data_manifest.json').read_text(encoding='utf-8'))
    required = [f'{asset}_{frequency}' for asset in ASSETS for frequency in ('daily', 'hourly')]
    required += ['USDJPY_FRED_daily', 'EURUSD_FRED_daily']
    for key in required:
        entry = manifest.get('files', {}).get(key)
        expected = f'data/raw/{key}.csv'
        if entry is None or entry.get('path') != expected:
            raise ValueError(f'Missing or inconsistent manifest entry: {key}')
        path = root/expected
        if not path.is_file():
            raise FileNotFoundError(f'Required input missing: {expected}; run scripts/download_data.py.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get('sha256'):
            raise ValueError(f'Input hash mismatch: {expected}; do not mix observations from different downloads.')
    return manifest


def daily_data(asset, root=ROOT):
    source = f'{asset}_FRED_daily.csv' if asset in ('EURUSD', 'USDJPY') else f'{asset}_daily.csv'
    path = Path(root)/'data'/'raw'/source
    frame = pd.read_csv(path, parse_dates=['date']).set_index('date').sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f'Duplicate dates: {asset}')
    if not np.isfinite(frame['adj_close']).all() or (frame['adj_close']<=0).any():
        raise ValueError(f'Invalid prices: {asset}')
    return frame, path


def load_hourly(asset, manifest, root=ROOT):
    """Clip provider-local request spillover to the stated UTC analysis interval."""
    entry = manifest['files'][f'{asset}_hourly']
    frame = pd.read_csv(Path(root)/'data'/'raw'/f'{asset}_hourly.csv')
    timestamps = pd.to_datetime(frame['timestamp'], utc=True)
    start = pd.Timestamp(entry['request']['start'], tz='UTC')
    end = pd.Timestamp(entry['request']['end_exclusive'], tz='UTC')
    frame = frame.loc[(timestamps >= start) & (timestamps < end)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f'No hourly observations inside the declared analysis interval for {asset}.')
    return frame


def rolling_plot(stats, intraday=False):
    fig, axes = plt.subplots(1,3,figsize=(11.2,3.15),sharey=False,layout='constrained')
    key = 'hours' if intraday else 'step'
    for ax,(asset,meta) in zip(axes,ASSETS.items()):
        part=stats[stats.asset==asset]
        if 'period' in part:
            part=part[part.period=='2000–2025']
        x=part[key].to_numpy()
        ax.fill_between(x,part.q25.to_numpy(),part.q75.to_numpy(),color=COLORS[asset],alpha=.16,label='25th–75th percentile')
        ax.plot(x,part['mean'],color=COLORS[asset],lw=2,label='Mean')
        ax.plot(x,part['median'],color='#18354A',ls='--',lw=1.5,label='Median')
        ax.set(title=meta['label'],xlabel='Sampling interval (hours)' if intraday else 'Sampling interval (trading days)')
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        if intraday: ax.set_xticks([1,2,3,6])
    axes[0].set_ylabel('Annualized volatility')
    axes[-1].legend(fontsize=7,loc='best')
    save(fig,'intraday_rolling_distribution' if intraday else 'daily_rolling_distribution')


def main():
    manifest = verify_input_manifest()
    OUT.mkdir(exist_ok=True); FIG.mkdir(parents=True,exist_ok=True); style()
    frames, signatures, distributions, audits = {}, [], [], []
    for asset in ASSETS:
        frame,path = daily_data(asset); frames[asset]=frame
        for label,(start,end) in PERIODS.items():
            p=frame.loc[start:end,'adj_close']
            if p.index.min().year != int(start[:4]) or p.index.max().year != int(end[:4]):
                raise ValueError(f'Incomplete required daily period for {asset} {label}')
            sig=daily_signature(p).reset_index().assign(asset=asset,period=label)
            signatures.append(sig)
            roll=daily_rolling(p)
            ds=distribution(roll).rename_axis('step').reset_index().assign(asset=asset,period=label)
            distributions.append(ds)
            audits.append({'asset':asset,'frequency':'daily','period':label,'first':str(p.index.min().date()),
                           'last':str(p.index.max().date()),'n_prices':len(p),'retained_sessions':len(p),
                           'rejected_sessions':0,'n_rolling_windows':len(roll),'source_file':path.name,
                           'min_window_days':(roll.index[0]-p.index[0]).days,
                           'max_window_days':np.nan})
        p=frame.loc['2000-01-01':'2025-12-31','adj_close']
        roll=daily_rolling(p)
        distributions.append(distribution(roll).rename_axis('step').reset_index().assign(asset=asset,period='2000–2025'))
    daily=pd.concat(signatures,ignore_index=True); ds=pd.concat(distributions,ignore_index=True)
    daily.to_csv(OUT/'daily_signatures.csv',index=False);ds.to_csv(OUT/'daily_distribution.csv',index=False)
    for label in PERIODS:
        fig,ax=plt.subplots(figsize=(9.6,3.7),layout='constrained')
        for asset,meta in ASSETS.items():
            part=daily[(daily.asset==asset)&(daily.period==label)]
            ax.plot(part.step,part.vol,lw=2,color=COLORS[asset],label=meta['label'])
            ax.fill_between(part.step,part.phase_min,part.phase_max,color=COLORS[asset],alpha=.13)
        ax.set(title=f'Daily volatility signature | {label}',xlabel='Sampling interval (trading days)',ylabel='Annualized volatility',xlim=(1,30))
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncols=3)
        save(fig,'daily_'+label.replace('–','_'))
    rolling_plot(ds)
    # Subperiod rolling summaries are separately plotted, as well as pooled results.
    fig,axes=plt.subplots(3,3,figsize=(11.2,8.2),layout='constrained')
    for row,period in enumerate(PERIODS):
        for col,asset in enumerate(ASSETS):
            ax=axes[row,col]; part=ds[(ds.asset==asset)&(ds.period==period)]
            ax.fill_between(part.step,part.q25,part.q75,color=COLORS[asset],alpha=.17)
            ax.plot(part.step,part['mean'],color=COLORS[asset],label='Mean')
            ax.plot(part.step,part['median'],'--',color='#18354A',label='Median')
            ax.set_title(f'{ASSETS[asset]["label"]} · {period}',fontsize=10)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            if row==2:ax.set_xlabel('Trading days')
    axes[0,0].legend(fontsize=7);save(fig,'daily_rolling_by_period')
    intraday, ids, session_sensitivity = [], [], []
    for asset,meta in ASSETS.items():
        hourly=load_hourly(asset,manifest)
        sig,roll,audit=intraday_analysis(hourly,meta)
        if len(roll)<2:
            raise ValueError(f'Not enough complete intraday sessions for {asset}')
        intraday.append(sig.reset_index().assign(asset=asset))
        session_sensitivity.append(sig.reset_index().assign(asset=asset,observed_hours=6 if asset=='SPY' else 18))
        if asset!='SPY':
            full,_,_=intraday_analysis(hourly,{**meta,'session_hours':24})
            session_sensitivity.append(full.reset_index().assign(asset=asset,observed_hours=24))
        ids.append(distribution(roll).rename_axis('hours').reset_index().assign(asset=asset))
        audit.to_csv(OUT/f'{asset}_session_audit.csv',index=False)
        valid=pd.to_datetime(audit.loc[audit.retained,'date'])
        counts=[int(((valid > end-pd.DateOffset(months=6)) & (valid <= end)).sum()) for end in roll.index]
        audits.append({'asset':asset,'frequency':'intraday','period':'2024-10–2025-12','first':str(valid.min().date()),
                       'last':str(valid.max().date()),'n_prices':len(hourly),'retained_sessions':int(audit.retained.sum()),
                       'rejected_sessions':int((~audit.retained).sum()),'n_rolling_windows':len(roll),
                       'source_file':f'{asset}_hourly.csv','min_window_days':min((d-(d-pd.DateOffset(months=6))).days for d in roll.index),
                       'max_window_days':max((d-(d-pd.DateOffset(months=6))).days for d in roll.index),
                       'min_window_sessions':min(counts),'max_window_sessions':max(counts)})
    intra=pd.concat(intraday,ignore_index=True);is_=pd.concat(ids,ignore_index=True)
    intra.to_csv(OUT/'intraday_signatures.csv',index=False);is_.to_csv(OUT/'intraday_distribution.csv',index=False)
    pd.concat(session_sensitivity,ignore_index=True).to_csv(OUT/'intraday_session_sensitivity.csv',index=False)
    audit=pd.DataFrame(audits);audit.to_csv(OUT/'data_audit.csv',index=False)
    fig,ax=plt.subplots(figsize=(9.6,3.7),layout='constrained')
    for asset,meta in ASSETS.items():
        part=intra[intra.asset==asset]
        ax.plot(part.hours,part.vol,'o-',lw=2,color=COLORS[asset],label=meta['label'])
    ax.set(title='Intraday volatility signature | October 2024–December 2025',xlabel='Sampling interval (hours)',ylabel='Annualized volatility',xticks=[1,2,3,6])
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(ncols=3);save(fig,'intraday_signature')
    rolling_plot(is_,True)
    metrics,sensitivity,curves=[],[],{}
    # Backtest uses Yahoo execution prices for each asset, not H.10 noon fixing.
    for asset in ASSETS:
        raw=pd.read_csv(ROOT/'data'/'raw'/f'{asset}_daily.csv',parse_dates=['date']).set_index('date')
        raw=raw.loc['2018-12-01':'2025-12-31']
        # Keep a fixed one-year warmup immediately before evaluation begins in 2020.
        first=np.flatnonzero(raw.index>='2020-01-01')[0]
        raw=raw.iloc[first-253:]
        if raw.splits.ne(0).any():raise ValueError('Split adjustment required for execution ledger.')
        for orientation,label in [(1,'LVD'),(-1,'HVD')]:
            result=backtest(raw.close,orientation=orientation,dividends=raw.dividends,signal_prices=raw.adj_close)
            result.to_csv(OUT/f'{asset}_{label}_ledger.csv',index_label='date')
            curves[asset,label]=result
            metrics.append(dict(asset=asset,strategy=label,cost_bps=1,**performance(result)))
        bh=raw.adj_close.reindex(curves[asset,'LVD'].index)
        bh_equity=100000*bh/bh.iloc[0]
        bh_frame=pd.DataFrame({'equity':bh_equity,'return':bh.pct_change().fillna(0),'turnover':0.,'cost':0.,'gross_pnl':bh_equity.diff().fillna(0),'net_pnl':bh_equity.diff().fillna(0),'exposure':1.})
        curves[asset,'Buy and hold']=bh_frame
        metrics.append(dict(asset=asset,strategy='Buy and hold',cost_bps=0,**performance(bh_frame)))
        for slow in [5,10,20]:
            for bps in [0,1,5,10]:
                r=backtest(raw.close,slow=slow,cost_bps=bps,dividends=raw.dividends,signal_prices=raw.adj_close)
                sensitivity.append(dict(asset=asset,slow_days=slow,cost_bps=bps,**performance(r)))
    metrics=pd.DataFrame(metrics);sensitivity=pd.DataFrame(sensitivity)
    metrics.to_csv(OUT/'strategy_metrics.csv',index=False);sensitivity.to_csv(OUT/'strategy_sensitivity.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.45),layout='constrained')
    for ax,asset in zip(axes,ASSETS):
        for label,color in [('LVD','#167D8D'),('HVD','#AD6A39'),('Buy and hold','#A4AEB6')]:
            c=curves[asset,label]
            ax.plot(c.index,c.equity/100000,color=color,label=label,lw=1.6)
        ax.axhline(1,color='#18354A',lw=.7,ls=':')
        ax.set(title=ASSETS[asset]['label'],ylabel='Wealth / initial capital')
        ax.tick_params(axis='x',labelrotation=30)
    axes[0].legend(fontsize=8);save(fig,'strategy_equity')
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.2),layout='constrained')
    for ax,asset in zip(axes,ASSETS):
        for slow,color in [(5,'#167D8D'),(10,'#66589C'),(20,'#AD6A39')]:
            part=sensitivity[(sensitivity.asset==asset)&(sensitivity.slow_days==slow)]
            ax.plot(part.cost_bps,part.total_return,'o-',color=color,label=f'1 vs {slow} days')
        ax.axhline(0,color='#18354A',lw=.6);ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(title=ASSETS[asset]['label'],xlabel='One-way transaction cost (bps)',ylabel='Total return, 2020–2025')
    axes[0].legend(fontsize=7);save(fig,'strategy_costs')
    # Reproducible summary for report generation and notebook display.
    summary={'data_audit':json.loads(audit.to_json(orient='records')),'strategy_metrics':json.loads(metrics.to_json(orient='records')),
             'settings':{'daily_window':252,'intraday_window_calendar_months':6,'daily_steps':list(range(1,31)),
                         'intraday_hours':[1,2,3,6],'year_sessions':252,'fixed_notional':100000,
                         'initial_equity':100000,'cost_bps':1,'signal_lag_sessions':1}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False,default=str),encoding='utf-8')
    print(audit.to_string(index=False))
    print(metrics[['asset','strategy','total_return','cagr','sharpe_zero_rate','max_drawdown']].to_string(index=False))


if __name__=='__main__':main()
