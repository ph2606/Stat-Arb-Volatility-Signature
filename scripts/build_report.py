"""Build LaTeX tables and result macros from the computed empirical outputs."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT, REPORT = ROOT/'outputs', ROOT/'report'
TABLES = REPORT/'tables'


def pct(x, digits=2):
    return f'{100*x:.{digits}f}\\%'


def table(name, columns, rows, align=None):
    align = align or 'l'+'r'*(len(columns)-1)
    text = '\\begin{tabular}{'+align+'}\n\\toprule\n'
    text += ' & '.join(columns)+' \\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str,row))+' \\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n'
    (TABLES/(name+'.tex')).write_text(text,encoding='utf-8')


def main():
    TABLES.mkdir(parents=True,exist_ok=True)
    daily=pd.read_csv(OUT/'daily_signatures.csv')
    audit=pd.read_csv(OUT/'data_audit.csv')
    intra=pd.read_csv(OUT/'intraday_signatures.csv')
    metrics=pd.read_csv(OUT/'strategy_metrics.csv')
    costs=pd.read_csv(OUT/'strategy_sensitivity.csv')
    names={'SPY':'SPY','USDJPY':'USD/JPY','EURUSD':'EUR/USD'}
    rows=[]
    for _,r in audit[audit.frequency.eq('daily')].iterrows():
        rows.append([names[r.asset],r.period,r['first'],r['last'],f'{r.n_prices:,}',f'{r.n_rolling_windows:,}'])
    table('daily_coverage',['Asset','Period','First price','Last price','Prices','Windows'],rows,'llllrr')
    rows=[]
    for (asset,period),g in daily.groupby(['asset','period'],sort=False):
        g=g.set_index('step')
        rows.append([names[asset],period,pct(g.loc[1,'vol']),pct(g.loc[5,'vol']),pct(g.loc[30,'vol']),
                     pct(g.loc[30,'phase_min'])+'--'+pct(g.loc[30,'phase_max'])])
    table('daily_endpoints',['Asset','Period','$k=1$','$k=5$','$k=30$','$k=30$ phase range'],rows,'llrrrr')
    rows=[]
    for _,r in audit[audit.frequency.eq('intraday')].iterrows():
        rows.append([names[r.asset],f'{r.n_prices:,}',str(r.retained_sessions),str(r.rejected_sessions),str(r.n_rolling_windows),
                     f'{int(r.min_window_sessions)}--{int(r.max_window_sessions)}'])
    table('intraday_coverage',['Asset','Raw bars','Retained days','Excluded days','Windows','Days/window'],rows,'lrrrrr')
    rows=[]
    for asset,g in intra.groupby('asset',sort=False):
        v=g.set_index('hours').vol
        rows.append([names[asset]]+[pct(v.loc[h]) for h in [1,2,3,6]])
    table('intraday_endpoints',['Asset','60 min','120 min','180 min','360 min'],rows)
    rows=[]
    for _,r in metrics.iterrows():
        rows.append([names[r.asset],r.strategy,pct(r.total_return),pct(r.cagr),pct(r.ann_vol),f'{r.sharpe_zero_rate:.2f}',pct(r.max_drawdown)])
    table('strategy',['Asset','Rule','Total','CAGR','Ann. vol.','Sharpe','Max DD'],rows,'llrrrrr')
    rows=[]
    for asset in names:
        g=costs[(costs.asset==asset)&(costs.slow_days==5)].set_index('cost_bps')
        slope=g.loc[0,'total_return']-g.loc[1,'total_return']
        threshold = g.loc[0,'total_return']/slope
        be=f'{threshold:.2f}' if threshold>0 else 'None'
        rows.append([names[asset]]+[pct(g.loc[c,'total_return'],3) for c in [0,1,5,10]]+[be])
    table('costs',['Asset','0 bps','1 bp','5 bps','10 bps','Break-even bps'],rows)
    rows=[]
    for _,r in metrics[metrics.strategy.eq('LVD')].iterrows():
        rows.append([names[r.asset],f'{r.annual_turnover:.2f}',pct(r.max_abs_exposure),f'{r.cost_quote_units:.2f}'])
    table('exposure',['Asset','Annual turnover / capital','Max net exposure','Total cost (quote units)'],rows)
    sensitivity=pd.read_csv(OUT/'intraday_session_sensitivity.csv')
    rows=[]
    for (asset,span),g in sensitivity.groupby(['asset','observed_hours'],sort=False):
        rows.append([names[asset],str(span),str(g.n_sessions.iloc[0]),pct(g[g.hours.eq(1)].vol.iloc[0]),pct(g[g.hours.eq(6)].vol.iloc[0])])
    table('session_sensitivity',['Asset','Observed hours/day','Sessions','60-min vol.','360-min vol.'],rows)
    macros=[]
    for asset in names:
        r=metrics[(metrics.asset==asset)&(metrics.strategy=='LVD')].iloc[0]
        macros.append('\\newcommand{\\'+asset+'LVD}{'+pct(r.total_return,3)+'}')
    eur=costs[(costs.asset=='EURUSD')&(costs.slow_days==5)].set_index('cost_bps')
    be=eur.loc[0,'total_return']/(eur.loc[0,'total_return']-eur.loc[1,'total_return'])
    macros.append('\\newcommand{\\EURBreakEven}{'+f'{be:.2f}'+'}')
    (REPORT/'results.tex').write_text('\n'.join(macros)+'\n',encoding='utf-8')
    print('Generated 8 result tables and result macros from current outputs.')


if __name__=='__main__':main()
