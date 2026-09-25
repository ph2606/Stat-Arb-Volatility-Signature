# Volatility Signature and Frequency Arbitrage

**Panagiotis Housos · ph2606**
NYU · Statistical Arbitrage · Fall 2026

[Read the LaTeX report](report/Panagiotis_Housos_Assignment1.pdf) · [Open the executed notebook](volatility_signature.ipynb)

This homework estimates volatility signatures for SPY, USD/JPY and EUR/USD,
summarizes their rolling distributions, and implements the two-frequency
inverse-price hedge motivated by *Options delta hedging with no options at all*.
Following the instructor's clarification, it includes genuine one-minute and
five-minute data with explicit sources, date ranges and availability limits.

The baseline strategy's cumulative 2020–2025 net returns at a one-basis-point
one-way cost are **−0.206% for SPY, −0.166% for USD/JPY and +0.080% for EUR/USD**.
The small positive EUR/USD result breaks even at about **2.21 bps**. The
experiment does not establish a dependable arbitrage opportunity.

## Assignment coverage

| Requirement | Implementation |
|---|---|
| Daily signature | Every integer interval from 1 to 30 observed trading days; log-return sample standard deviation, `ddof=1`, annualized with `sqrt(252/k)` |
| Historical comparison | Separate graphs for 2000–2007, 2007–2014 and 2020–2025; 2007 intentionally belongs to both first periods |
| Daily rolling distributions | 252-return windows, with mean, median, 25th and 75th percentiles at every frequency, pooled and by subperiod |
| Minute signatures | 1, 2, 3, 5, 10, 15, 30, 60, 120, 180 and 360 minutes from one-minute bars; 5, 10, 15, 30, 60, 120, 180 and 360 minutes from the longer five-minute history |
| Longer intraday signature | 60, 120, 180 and 360 minutes using October 2024–December 2025 hourly observations |
| Intraday rolling distributions | Exact six-calendar-month windows and all four requested statistics at hourly-supported frequencies; minute-only frequencies are explicitly unavailable over six months with the retrieved inputs |
| Frequency arbitrage | Daily vs five-day inverse-price rebalancing; lagged signal; explicit position/cash-flow ledger; opposite orientation and buy-and-hold controls; costs and frequency sensitivity |

## Data and interpretation

Yahoo Finance supplies SPY daily prices, daily strategy execution marks and all
hourly and minute observations. Main daily FX signatures use complete Federal Reserve H.10
series through FRED: [DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS) and
[DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU). These are New York noon
fixings. FRED supplements the assignment's listed providers because Yahoo's
EUR/USD history starts in 2003 and its older FX data contain isolated large
discrepancies. The attempted Nasdaq Data Link endpoint returned HTTP 403.
No provider series are spliced and no unavailable observations are fabricated.

The minute data were retrieved from **Yahoo Finance via yfinance**, using a
completed-date cutoff of September 23, 2026. Requests for earlier data were
rejected under the tested 30-day one-minute and 60-day five-minute limits.
One-minute requests were split into batches of at most seven days. These are
the longest histories retrieved at each resolution from this source.

| Minute input | Usable return dates | SPY sessions | USD/JPY sessions | EUR/USD sessions |
|---|---|---:|---:|---:|
| One-minute bars | August 26–September 23, 2026 | 20 | 21 | 21 |
| Five-minute bars | July 27–September 23, 2026 | 42 | 43 | 43 |

These session counts refer to the base sampling interval. Exact raw coverage
is shown separately in the report and notebook: FX bars begin August 25 at
21:10 UTC for the one-minute source and July 26 at 23:00 UTC for the five-minute
source, after the selected observation span on those dates. The raw endpoints
are not extra analyzed sessions. Both minute histories are too short to support
genuine six-month rolling windows, so those calculations use the longer hourly
sample. The report states the remaining minute-only availability limit.

SPY intraday returns use a 09:30–15:30 New York observation span. FX uses a common
00:00–18:00 UTC span to retain all weekdays. The hourly analysis requires every
hourly mark in a retained session; the minute analysis requires every base-grid
mark within each nonoverlapping block, rejecting only affected blocks and
preserving subsequent anchors. Minute endpoints use bar opens at their stated
timestamps. There is no overnight bridging or forward fill.
Annualization uses the assignment's `H=6.5` for SPY and `H=24` for FX; extending
the sampled variance rate to a full active day is an explicit assumption.
A complete-24-hour FX sensitivity is included for the hourly sample.

Minute gaps can change the sample composition across frequencies: EUR/USD's
one-minute input supports one-minute returns on 21 days but only 47 complete
six-hour blocks over 16 days. A separate comparison on identical valid
five-minute intervals gives exactly matching stored returns from the one-minute
and native five-minute inputs for all three assets. This validates internal
source consistency, while the full-history curves also reflect different dates.

The strategy uses 100,000 initial quote-currency units and fixed notional of
100,000: USD for SPY and EUR/USD, JPY for USD/JPY. It is not a combined USD
portfolio. Cash interest and FX carry are zero, simultaneous legs are netted,
SPY dividends enter the cash ledger, and final liquidation incurs cost.
Buy and hold has substantially greater exposure and is not risk matched.
The 2020–2025 daily specification is not selected using the later minute sample.

[data_manifest.json](data_manifest.json) records the eight daily/hourly inputs;
[minute_data_manifest.json](minute_data_manifest.json) records six minute inputs,
including request bounds, retention-limit responses, observed timestamps,
cleaning counts and hashes. All 14 files are verified before analysis writes
results.
Raw market data and derived daily trading ledgers are excluded from GitHub.
The supplied course documents are not republished.

## Reproduce

Python 3.12 was used. From the repository root, create an environment and install
the dependencies:

```sh
python -m venv .venv
```

Activate `.venv` using the command appropriate to your shell, then run:

```sh
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/download_minute_data.py
python scripts/run_analysis.py
python -m pytest -q
python scripts/build_notebook.py --execute
python scripts/build_report.py
```

Compile the report from its directory with XeLaTeX (MiKTeX or TeX Live):

```sh
cd report
xelatex -interaction=nonstopmode -halt-on-error -jobname=Panagiotis_Housos_Assignment1 report.tex
xelatex -interaction=nonstopmode -halt-on-error -jobname=Panagiotis_Housos_Assignment1 report.tex
```

The downloaders reuse existing files only if their requests and hashes match
the local manifests. The analysis checks all 14 inputs against both public
manifests before writing results and requires the documented FRED daily FX files;
it cannot silently substitute another provider. Intraday observations are clipped
to their declared UTC request intervals before analysis.
`--refresh` deliberately replaces the relevant cache. Network access
is required for initial downloads; it is not required for analysis of existing
inputs, tests, or report compilation once TeX packages are installed.

Yahoo's intraday retention periods move forward. A future download of these exact
historical samples may fail; the downloaders stop rather than silently replacing
them with different periods. Local original inputs and the saved notebook/PDF
preserve this analysis. Data-provider revisions can also change later downloads.

## Project files

| File | Purpose |
|---|---|
| `volatility_signature.ipynb` | Executed notebook: 13 code cells, full numerical result tables, minute availability summary and 12 figures |
| `src/analysis.py` | Daily/hourly estimators, session construction, rolling windows and strategy accounting |
| `src/minute_analysis.py` | Minute return blocks, signatures and matched-source comparison |
| `scripts/download_data.py` | Daily/hourly acquisition, input validation and source manifest |
| `scripts/download_minute_data.py` | Minute acquisition, retention probes, cleaning and source manifest |
| `scripts/run_analysis.py` | All analysis tables and figures |
| `scripts/analyze_minute_data.py` | Minute input validation, four result tables and three figures |
| `scripts/build_notebook.py` | Rebuild and execute the notebook |
| `scripts/build_report.py` | Generate LaTeX tables and numerical macros from results |
| `report/report.tex` | Authored report source |
| `report/Panagiotis_Housos_Assignment1.pdf` | Standalone submission report |
| `tests/` | 45 checks of formulas, causality, session boundaries, cash flows, minute gaps and input integrity |

The reference's printed profit expression uses an ending price in a holding
applied to the preceding return. This implementation instead credits each
price change to the previously established holding. The report explains this
correction and the distinction between sampling frequencies and lookback lengths.
