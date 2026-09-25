"""Reproduce the shorter-history minute-data section allowed by the clarification."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from src.minute_analysis import minute_signature, matched_five_minute_comparison

ASSETS = {'SPY':{'symbol':'SPY','kind':'equity','hours_per_day':6.5,'label':'SPY'},
          'USDJPY':{'symbol':'JPY=X','kind':'fx','hours_per_day':24,'label':'USD/JPY'},
          'EURUSD':{'symbol':'EURUSD=X','kind':'fx','hours_per_day':24,'label':'EUR/USD'}}
COLORS = {'SPY':'#167D8D','USDJPY':'#AD6A39','EURUSD':'#66589C'}


def verify_minute_inputs(root=ROOT):
    root=Path(root)
    manifest=json.loads((root/'minute_data_manifest.json').read_text(encoding='utf-8'))
    for asset in ASSETS:
        for base in [1,5]:
            key=f'{asset}_{base}m'; relative=f'data/raw/{key}.csv'
            entry=manifest.get('files',{}).get(key)
            if entry is None or entry.get('path')!=relative:
                raise ValueError(f'Missing or inconsistent minute manifest entry: {key}')
            path=root/relative
            if not path.is_file():raise FileNotFoundError(f'Required minute input missing: {relative}')
            if hashlib.sha256(path.read_bytes()).hexdigest()!=entry.get('sha256'):
                raise ValueError(f'Minute input hash mismatch: {relative}')
    return manifest


def main(root=ROOT):
    root=Path(root); manifest=verify_minute_inputs(root)
    out=root/'outputs';figdir=root/'report'/'figures'
    out.mkdir(exist_ok=True);figdir.mkdir(parents=True,exist_ok=True)
    data,coverage,signatures,audits,comparisons={},[],[],[],[]
    for asset,meta in ASSETS.items():
        for base in [1,5]:
            entry=manifest['files'][f'{asset}_{base}m']
            df=pd.read_csv(root/entry['path'])
            timestamps=pd.to_datetime(df.timestamp,utc=True)
            request=entry['request']
            start=pd.to_datetime(request['start'],utc=True)
            end=pd.to_datetime(request['end_exclusive'],utc=True)
            df=df.loc[(timestamps>=start)&(timestamps<end)].reset_index(drop=True)
            sig,audit=minute_signature(df,meta,base)
            data[asset,base]=df
            signatures.append(sig.assign(asset=asset))
            audits.append(audit.assign(asset=asset))
            coverage.append({'asset':asset,'base_minutes':base,'source':'Yahoo Finance',
                             'requested_start_utc':request['start'],'requested_end_exclusive_utc':request['end_exclusive'],
                             'observed_first_bar_utc':pd.to_datetime(df.timestamp,utc=True).min().isoformat(),
                             'observed_last_bar_utc':pd.to_datetime(df.timestamp,utc=True).max().isoformat(),
                             'n_bars':len(df),'base_valid_returns':int(sig.n_returns.iloc[0]),
                             'base_sessions':int(sig.n_sessions.iloc[0]),
                             'usable_first_return_utc':sig.first_return_start_utc.iloc[0],
                             'usable_last_return_utc':sig.last_return_end_utc.iloc[0],
                             'six_month_window_available':False})
        comparisons.append(dict(asset=asset,**matched_five_minute_comparison(data[asset,1],data[asset,5],meta)))
    sig=pd.concat(signatures,ignore_index=True); cov=pd.DataFrame(coverage);comp=pd.DataFrame(comparisons)
    sig.to_csv(out/'minute_signatures.csv',index=False);cov.to_csv(out/'minute_coverage.csv',index=False)
    comp.to_csv(out/'minute_comparison.csv',index=False)
    pd.concat(audits,ignore_index=True).to_csv(out/'minute_session_audit.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.grid':True,'grid.alpha':.2,'text.color':'#18354A'})
    for base in [1,5]:
        fig,ax=plt.subplots(figsize=(9.6,3.6),layout='constrained')
        for asset,meta in ASSETS.items():
            p=sig[(sig.asset==asset)&(sig.base_minutes==base)]
            ax.plot(p.interval_minutes,p.annualized_vol,'o-',color=COLORS[asset],label=meta['label'],lw=1.8,ms=4)
        dates=cov[cov.base_minutes==base]
        first=min(str(x)[:10] for x in dates.observed_first_bar_utc)
        last=max(str(x)[:10] for x in dates.observed_last_bar_utc)
        ax.set_xscale('log');ticks=[1,2,5,10,30,60,180,360] if base==1 else [5,10,15,30,60,120,180,360]
        ax.set_xticks(ticks,labels=[str(x) for x in ticks]);ax.minorticks_off()
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(title=f'{base}-minute source data | {first}–{last}',xlabel='Sampling interval (minutes; logarithmic scale)',ylabel='Annualized volatility')
        ax.legend(ncols=3)
        for ext in ['png','pdf']:fig.savefig(figdir/f'minute_signature_{base}m.{ext}',dpi=190,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11.2,3.1),layout='constrained')
    for ax,(_,r) in zip(axes,comp.iterrows()):
        ax.bar(['Aggregated 1m','Native 5m'],[r.vol_from_1m,r.vol_native_5m],color=[COLORS[r.asset],'#A4AEB6'],width=.55)
        ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set(title=ASSETS[r.asset]['label'],ylabel='5-minute annualized volatility')
        ax.text(.5,.94,f'{int(r.matched_returns):,} matched blocks',transform=ax.transAxes,ha='center',va='top',fontsize=8)
        ax.set_ylim(0,max(r.vol_from_1m,r.vol_native_5m)*1.22)
    for ext in ['png','pdf']:fig.savefig(figdir/f'minute_comparison.{ext}',dpi=190,bbox_inches='tight')
    plt.close(fig)
    print(cov[['asset','base_minutes','observed_first_bar_utc','observed_last_bar_utc','n_bars','base_sessions']].to_string(index=False))
    return cov,sig,comp


if __name__=='__main__':main()
