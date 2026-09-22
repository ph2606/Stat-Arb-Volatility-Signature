# Volatility Signature and Frequency Arbitrage

**Panagiotis Housos · ph2606**
NYU · Statistical Arbitrage · Fall 2026

[Read the LaTeX report](report/report.pdf) · [Open the executed notebook](volatility_signature.ipynb)

This homework estimates volatility signatures for SPY, USD/JPY and EUR/USD,
summarizes their rolling distributions, and implements the two-frequency
inverse-price hedge motivated by *Options delta hedging with no options at all*.

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
| Intraday signature | 60, 120, 180 and 360 minutes, using October 2024–December 2025 hourly observations |
| Intraday rolling distributions | Exact six-calendar-month windows, with all four requested statistics |
| Frequency arbitrage | Daily vs five-day inverse-price rebalancing; lagged signal; explicit position/cash-flow ledger; opposite orientation and buy-and-hold controls; costs and frequency sensitivity |

## Data and interpretation

Yahoo Finance supplies SPY daily prices, daily strategy execution marks and all
hourly observations. Main daily FX signatures use complete Federal Reserve H.10
series through FRED: [DEXJPUS](https://fred.stlouisfed.org/series/DEXJPUS) and
[DEXUSEU](https://fred.stlouisfed.org/series/DEXUSEU). These are New York noon
fixings. FRED supplements the assignment's listed providers because Yahoo's
EUR/USD history starts in 2003 and its older FX data contain isolated large
discrepancies. The attempted Nasdaq Data Link endpoint returned HTTP 403.
No provider series are spliced and no unavailable observations are fabricated.

SPY intraday returns use 09:30–15:30 New York hourly marks. FX uses a common
00:00–18:00 UTC span to retain all weekdays. Missing required endpoints cause a
day to be excluded. There is no overnight bridging or forward fill.
Annualization uses the assignment's `H=6.5` for SPY and `H=24` for FX; extending
the sampled variance rate to a full active day is an explicit assumption.
A complete-24-hour FX sensitivity is included. Hourly data cannot resolve
sub-hour microstructure effects.

The strategy uses 100,000 initial quote-currency units and fixed notional of
100,000: USD for SPY and EUR/USD, JPY for USD/JPY. It is not a combined USD
portfolio. Cash interest and FX carry are zero, simultaneous legs are netted,
SPY dividends enter the cash ledger, and final liquidation incurs cost.
Buy and hold has substantially greater exposure and is not risk matched.

[data_manifest.json](data_manifest.json) records acquisition timestamps,
coverage, software versions, quality checks and hashes for all eight inputs.
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
python scripts/run_analysis.py
python -m pytest -q
python scripts/build_notebook.py --execute
python scripts/build_report.py
```

Compile the report from its directory with XeLaTeX (MiKTeX or TeX Live):

```sh
cd report
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
```

The downloader reuses existing files only if their requests and hashes match
the local manifest. The analysis also checks all eight inputs against the public
manifest before writing results and requires the documented FRED daily FX files;
it cannot silently substitute another provider. Hourly observations are clipped
to the declared UTC interval before rolling-window eligibility is determined.
`--refresh` deliberately replaces the cache. Network access
is required for initial downloads; it is not required for analysis of existing
inputs, tests, or report compilation once TeX packages are installed.

Yahoo's hourly retention period moves forward. A future download of this exact
historical sample may fail; the downloader stops rather than silently replacing
it with a different period. Local original inputs and the saved notebook/PDF
preserve this analysis. Data-provider revisions can also change later downloads.

## Project files

| File | Purpose |
|---|---|
| `volatility_signature.ipynb` | Executed notebook: 10 code cells, full numerical tables and nine figures |
| `src/analysis.py` | Estimators, session construction, rolling windows and strategy accounting |
| `scripts/download_data.py` | Data acquisition, input validation and source manifest |
| `scripts/run_analysis.py` | All analysis tables and figures |
| `scripts/build_notebook.py` | Rebuild and execute the notebook |
| `scripts/build_report.py` | Generate LaTeX tables and numerical macros from results |
| `report/report.tex` | Authored report source |
| `report/report.pdf` | Standalone submission report |
| `tests/` | 24 checks of formulas, causality, session boundaries, cash flows and input integrity |

The reference's printed profit expression uses an ending price in a holding
applied to the preceding return. This implementation instead credits each
price change to the previously established holding. The report explains this
correction and the distinction between sampling frequencies and lookback lengths.
