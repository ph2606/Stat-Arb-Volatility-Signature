"""Download the assignment's public Yahoo Finance daily and hourly observations.

Run from any directory with ``python scripts/download_data.py``. Raw observations
are local research inputs; the repository retains only aggregate results and the
provenance manifest. Yahoo restricts historical intraday retention, so an exact
future re-download is not guaranteed. No missing historical data are fabricated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from io import StringIO
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "SPY": {
        "symbol": "SPY",
        "description": "SPDR S&P 500 ETF Trust",
        "quote": "USD per ETF share",
        "active_hours_per_day": 6.5,
        "analysis_timezone": "America/New_York",
        "asset_class": "equity_etf",
    },
    "USDJPY": {
        "symbol": "JPY=X",
        "description": "US dollar / Japanese yen",
        "quote": "JPY per USD",
        "active_hours_per_day": 24.0,
        "analysis_timezone": "UTC",
        "asset_class": "foreign_exchange",
    },
    "EURUSD": {
        "symbol": "EURUSD=X",
        "description": "Euro / US dollar",
        "quote": "USD per EUR",
        "active_hours_per_day": 24.0,
        "analysis_timezone": "UTC",
        "asset_class": "foreign_exchange",
    },
    "USDSEK": {
        "symbol": "SEK=X",
        "description": "US dollar / Swedish krona",
        "quote": "SEK per USD",
        "active_hours_per_day": 24.0,
        "analysis_timezone": "UTC",
        "asset_class": "foreign_exchange",
    },
}
PERIODS = {
    "2000-2007": ("2000-01-01", "2007-12-31"),
    "2007-2014": ("2007-01-01", "2014-12-31"),
    "2020-2025": ("2020-01-01", "2025-12-31"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_history(symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Retry transient provider errors without substituting a different sample."""
    error: Exception | None = None
    for attempt in range(3):
        try:
            data = yf.Ticker(symbol).history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=False,
                actions=True,
                prepost=False,
                repair=False,
                keepna=False,
                timeout=30,
            )
            if data.empty:
                raise ValueError(f"Yahoo returned no {interval} data for {symbol}")
            if data.index.tz is None:
                raise ValueError(f"Provider returned timezone-naive data for {symbol}")
            if data.index.has_duplicates:
                raise ValueError(f"Duplicate provider timestamps in {symbol} {interval}")
            return data.sort_index()
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {symbol} {interval}: {error}") from error


def normalize_daily(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Preserve exchange-local trading dates and both raw and adjusted closes."""
    required = ["Open", "Close", "Adj Close"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Daily provider response missing columns: {missing}")
    usable = data[required].notna().all(axis=1)
    usable &= np.isfinite(data[required]).all(axis=1)
    usable &= (data[required] > 0).all(axis=1)
    result = pd.DataFrame({
        "date": data.index.strftime("%Y-%m-%d"),
        "open": data["Open"].to_numpy(),
        "close": data["Close"].to_numpy(),
        "adj_close": data["Adj Close"].to_numpy(),
        "dividends": data.get("Dividends", pd.Series(0.0, index=data.index)).to_numpy(),
        "splits": data.get("Stock Splits", pd.Series(0.0, index=data.index)).to_numpy(),
    }).loc[usable.to_numpy()].reset_index(drop=True)
    if result["date"].duplicated().any():
        raise ValueError("Daily output contains duplicate trading dates")
    diagnostics = {
        "provider_rows": len(data),
        "dropped_nonpositive_or_nonfinite_price_rows": int((~usable).sum()),
        "zero_close_change_rows": int(result["close"].diff().eq(0).sum()),
        "weekend_rows": int((pd.to_datetime(result["date"]).dt.dayofweek >= 5).sum()),
        "max_absolute_adjusted_log_return": float(
            np.log(result["adj_close"]).diff().abs().max()
        ),
    }
    return result, diagnostics


def normalize_hourly(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Preserve provider bar-start timestamps as explicit UTC ISO-8601 strings."""
    required = ["Open", "High", "Low", "Close"]
    usable = data[required].notna().all(axis=1)
    usable &= np.isfinite(data[required]).all(axis=1)
    usable &= (data[required] > 0).all(axis=1)
    result = pd.DataFrame({
        "timestamp": data.index.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "open": data["Open"].to_numpy(),
        "high": data["High"].to_numpy(),
        "low": data["Low"].to_numpy(),
        "close": data["Close"].to_numpy(),
        "volume": data["Volume"].to_numpy(),
    }).loc[usable.to_numpy()].reset_index(drop=True)
    if result["timestamp"].duplicated().any():
        raise ValueError("Hourly output contains duplicate UTC timestamps")
    diagnostics = {
        "provider_rows": len(data),
        "dropped_nonpositive_or_nonfinite_price_rows": int((~usable).sum()),
        "zero_volume_rows": int(result["volume"].eq(0).sum()),
        "zero_close_change_rows": int(result["close"].diff().eq(0).sum()),
    }
    return result, diagnostics


def period_coverage(frame: pd.DataFrame) -> dict:
    coverage = {}
    for name, (start, end) in PERIODS.items():
        sample = frame.loc[frame["date"].between(start, end), "date"]
        first = None if sample.empty else sample.iloc[0]
        last = None if sample.empty else sample.iloc[-1]
        coverage[name] = {
            "requested_start": start,
            "requested_end_inclusive": end,
            "observed_start": first,
            "observed_end": last,
            "rows": len(sample),
            "covers_start_year": bool(first and first[:4] == start[:4]),
            "covers_end_year": bool(last and last[:4] == end[:4]),
        }
    return coverage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-start", default="1999-01-01")
    parser.add_argument("--daily-end", default="2026-01-01", help="Exclusive end date")
    parser.add_argument("--hourly-start", default="2024-10-01")
    parser.add_argument("--hourly-end", default="2026-01-01", help="Exclusive end date")
    parser.add_argument("--assets", nargs="+", default=["SPY", "USDJPY", "EURUSD"], choices=ASSETS)
    parser.add_argument("--refresh", action="store_true", help="Replace previously downloaded files")
    return parser.parse_args()


def download_fred_fx(asset: str, args: argparse.Namespace, previous: dict) -> dict:
    """Keep a complete, standalone primary-source FX daily series.

    FRED DEXUSEU/DEXJPUS are Federal Reserve New York noon buying rates, not
    exchange closing price. ``close`` is only a common downstream column name.
    There is no reported open, so its column is missing rather than invented.
    """
    series = {"EURUSD": "DEXUSEU", "USDJPY": "DEXJPUS"}[asset]
    relative = f"data/raw/{asset}_FRED_daily.csv"
    path = ROOT / relative
    key = f"{asset}_FRED_daily"
    end_inclusive = (pd.Timestamp(args.daily_end) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
        f"&cosd={args.daily_start}&coed={end_inclusive}"
    )
    request = {"series": series, "start": args.daily_start, "end_exclusive": args.daily_end, "url": url}
    old = previous.get("files", {}).get(key)
    if path.exists() and not args.refresh:
        if not old or old.get("request") != request or sha256(path) != old.get("sha256"):
            raise RuntimeError(f"Existing {relative} does not match the manifest/request; use --refresh deliberately")
        print(f"Verified cached {relative}: {old['rows']:,} rows", flush=True)
        if "listed_provider_attempt" in old:
            old["listed_provider_attempt"].pop("checked_utc", None)
            old["listed_provider_attempt"]["checked_date_utc"] = "2026-09-20"
        return old
    print(f"Downloading standalone Federal Reserve {asset} daily series {series}", flush=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text), na_values=["."])
    if not {"observation_date", series}.issubset(raw.columns):
        raise ValueError(f"Unexpected columns in FRED {asset} response")
    values = pd.to_numeric(raw[series], errors="coerce")
    valid = values.notna() & np.isfinite(values) & values.gt(0)
    valid &= raw["observation_date"].between(args.daily_start, end_inclusive)
    result = pd.DataFrame({
        "date": raw.loc[valid, "observation_date"],
        "open": np.nan,
        "close": values.loc[valid],
        "adj_close": values.loc[valid],
        "dividends": 0.0,
        "splits": 0.0,
    }).sort_values("date").reset_index(drop=True)
    if result.empty or result["date"].duplicated().any():
        raise ValueError(f"Empty or duplicate-date FRED {asset} sample")
    result.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
    entry = {
        "path": relative,
        "request": request,
        "source": "Board of Governors of the Federal Reserve System, H.10; retrieved via FRED",
        "source_url": f"https://fred.stlouisfed.org/series/{series}",
        "downloaded_utc": utc_now(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "rows": len(result),
        "columns": result.columns.tolist(),
        "observed_start": result["date"].iloc[0],
        "observed_end": result["date"].iloc[-1],
        "provider_timezone": "America/New_York",
        "measurement": "Noon buying rates in New York City for cable transfers payable in foreign currencies",
        "quote": ASSETS[asset]["quote"],
        "column_semantics": {
            "close": "New York noon buying rate, not a daily exchange closing price",
            "adj_close": "Same noon rate; spot FX has no equity dividend/split adjustment",
            "open": "Missing: source reports one daily fixing, no opening price",
        },
        "diagnostics": {
            "provider_rows": len(raw),
            "dropped_missing_or_invalid_fixings": int((~valid).sum()),
            "zero_close_change_rows": int(result["close"].diff().eq(0).sum()),
            "max_absolute_log_return": float(np.log(result["close"]).diff().abs().max()),
        },
        "subperiod_coverage": period_coverage(result),
    }
    if asset == "EURUSD":
        entry["listed_provider_attempt"] = {
            "url": "https://data.nasdaq.com/api/v3/datasets/FRED/DEXUSEU.csv?start_date=1999-01-01&end_date=2025-12-31",
            "checked_date_utc": "2026-09-20",
            "http_status": 403,
            "resolution": "Accessed the same Federal Reserve series directly from its public FRED distributor",
        }
    print(f"Saved {relative}: {len(result):,} rows; {entry['observed_start']} through {entry['observed_end']}", flush=True)
    return entry


def audit_fx_quality(manifest: dict) -> dict:
    """Compare isolated provider discrepancies and audit the backtest sample.

    Different fixings need not agree. A threshold identifies records for manual
    review; it does not prove errors, and no observations are deleted or patched.
    """
    audit = {
        "method": "Compare standalone provider levels by calendar date; inspect absolute daily log returns.",
        "interpretation": "Yahoo and FRED quote different observation times; discrepancies are review flags, not proven errors.",
        "price_discrepancy_log_threshold": 0.05,
        "recent_return_log_threshold": 0.05,
        "recent_sample_start": "2020-01-01",
        "recent_sample_end": "2025-12-31",
        "assets": {},
    }
    flagged_records = []
    for asset in ["EURUSD", "USDJPY"]:
        yahoo_key, fred_key = f"{asset}_daily", f"{asset}_FRED_daily"
        if yahoo_key not in manifest["files"] or fred_key not in manifest["files"]:
            continue
        yahoo = pd.read_csv(ROOT / manifest["files"][yahoo_key]["path"])
        fred = pd.read_csv(ROOT / manifest["files"][fred_key]["path"])
        yahoo["log_return"] = np.log(yahoo["close"]).diff()
        fred["log_return"] = np.log(fred["close"]).diff()
        joined = yahoo[["date", "close", "log_return"]].merge(
            fred[["date", "close", "log_return"]], on="date", how="left", suffixes=("_yahoo", "_fred")
        )
        joined["log_price_discrepancy"] = np.log(joined["close_yahoo"] / joined["close_fred"])
        recent = joined.loc[joined["date"].between("2020-01-01", "2025-12-31")].copy()
        extreme_levels = joined.loc[joined["log_price_discrepancy"].abs().gt(0.05)]
        maximum_index = recent["log_return_yahoo"].abs().idxmax()
        audit["assets"][asset] = {
            "overlapping_daily_dates": int(joined["close_fred"].notna().sum()),
            "level_discrepancy_flag_count": int(len(extreme_levels)),
            "level_discrepancy_flag_dates": extreme_levels["date"].tolist(),
            "recent_yahoo_return_count": int(recent["log_return_yahoo"].notna().sum()),
            "recent_yahoo_absolute_log_return_above_5pct_count": int(recent["log_return_yahoo"].abs().gt(0.05).sum()),
            "recent_yahoo_max_absolute_log_return": float(recent["log_return_yahoo"].abs().max()),
            "recent_yahoo_max_absolute_log_return_date": recent.loc[maximum_index, "date"],
            "all_sample_fred_max_absolute_log_return": float(fred["log_return"].abs().max()),
        }
        for reason, sample in [
            ("absolute_log_price_discrepancy_above_5pct", extreme_levels),
            ("five_largest_recent_yahoo_absolute_log_returns", recent.loc[recent["log_return_yahoo"].abs().nlargest(5).index]),
        ]:
            for record in sample.to_dict(orient="records"):
                flagged_records.append({"asset": asset, "audit_reason": reason, **record})
    evidence_path = ROOT / "data" / "data_quality_audit.csv"
    if flagged_records:
        pd.DataFrame(flagged_records).to_csv(evidence_path, index=False, float_format="%.12g", lineterminator="\n")
        audit["local_evidence_path"] = "data/data_quality_audit.csv"
        audit["local_evidence_sha256"] = sha256(evidence_path)
    return audit


def main() -> None:
    args = parse_args()
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / ".cache" / "yfinance"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    manifest_path = ROOT / "data" / "manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest = {
        "schema_version": 1,
        "manifest_updated_utc": utc_now(),
        "source": "Yahoo Finance via yfinance; supplementary Federal Reserve daily FX fixings via FRED",
        "source_urls": {
            asset: f"https://finance.yahoo.com/quote/{ASSETS[asset]['symbol']}/history/"
            for asset in args.assets
        },
        "software": {
            "python": platform.python_version(),
            **{pkg: importlib.metadata.version(pkg) for pkg in ["yfinance", "pandas", "numpy", "requests"]},
        },
        "annualization_trading_days": 252,
        "assets": {asset: ASSETS[asset] for asset in args.assets},
        "analysis_daily_file_keys": {asset: (f"{asset}_FRED_daily" if asset in ["EURUSD", "USDJPY"] else f"{asset}_daily") for asset in args.assets},
        "conventions": {
            "date_end": "All download end dates are exclusive.",
            "daily_dates": "Provider-local calendar dates of daily observations.",
            "daily_prices": "Unadjusted open/close and separately reported adjusted close; no manual backfill.",
            "hourly_timestamps": "UTC bar-start times, not bar-end observation times.",
            "hourly_prices": "Unadjusted OHLC; regular-hours-only for SPY.",
            "spy_last_bar": "SPY 15:30 local bar ends at 16:00 and is only 30 minutes long.",
            "fx_volume": "Yahoo spot-FX volume commonly equals zero and is not a traded-volume measure.",
            "missing_data": "No forward filling, interpolation, synthetic data, or alternate-source splicing.",
            "raw_data_retention": "Raw CSV inputs and yfinance caches are excluded from Git.",
            "fx_daily_source": "Use standalone FRED DEXUSEU and DEXJPUS for full-period daily signatures; Yahoo daily remains a separate robustness sample. Never splice providers.",
        },
        "files": {},
        "warnings": [
            "Yahoo may revise adjusted prices; SHA256 fingerprints identify the exact local inputs.",
            "Yahoo's rolling intraday-retention limit prevents guaranteed future re-downloads of this fixed sample.",
        ],
    }
    for asset in args.assets:
        info = ASSETS[asset]
        for frequency, interval, start, end in [
            ("daily", "1d", args.daily_start, args.daily_end),
            ("hourly", "1h", args.hourly_start, args.hourly_end),
        ]:
            relative = f"data/raw/{asset}_{frequency}.csv"
            path = ROOT / relative
            key = f"{asset}_{frequency}"
            old = previous.get("files", {}).get(key)
            request = {"symbol": info["symbol"], "interval": interval, "start": start, "end_exclusive": end}
            if path.exists() and not args.refresh:
                if not old or old.get("request") != request or sha256(path) != old.get("sha256"):
                    raise RuntimeError(f"Existing {relative} does not match the manifest/request; use --refresh deliberately")
                entry = old
                result = pd.read_csv(path)
                print(f"Verified cached {relative}: {len(result):,} rows", flush=True)
            else:
                print(f"Downloading {asset} {interval}: {start} to {end} (exclusive)", flush=True)
                data = fetch_history(info["symbol"], start, end, interval)
                result, diagnostics = normalize_daily(data) if frequency == "daily" else normalize_hourly(data)
                result.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
                time_col = "date" if frequency == "daily" else "timestamp"
                entry = {
                    "path": relative,
                    "request": request,
                    "downloaded_utc": utc_now(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                    "rows": len(result),
                    "columns": result.columns.tolist(),
                    "observed_start": result[time_col].iloc[0],
                    "observed_end": result[time_col].iloc[-1],
                    "provider_timezone": str(data.index.tz),
                    "diagnostics": diagnostics,
                }
                if frequency == "daily":
                    entry["subperiod_coverage"] = period_coverage(result)
                else:
                    timestamps = pd.to_datetime(result["timestamp"], utc=True).dt.tz_convert(info["analysis_timezone"])
                    counts = timestamps.dt.strftime("%Y-%m-%d").value_counts()
                    entry["analysis_calendar_dates"] = int(len(counts))
                    entry["bars_per_calendar_date_min"] = int(counts.min())
                    entry["bars_per_calendar_date_median"] = float(counts.median())
                    entry["bars_per_calendar_date_max"] = int(counts.max())
                print(f"Saved {relative}: {len(result):,} rows; {entry['observed_start']} through {entry['observed_end']}", flush=True)
            manifest["files"][key] = entry
            if frequency == "daily":
                for period, coverage in entry["subperiod_coverage"].items():
                    if not coverage["covers_start_year"] or not coverage["covers_end_year"]:
                        manifest["warnings"].append(
                            f"{asset} {period} has partial coverage: {coverage['observed_start']} through "
                            f"{coverage['observed_end']}; all period labels must disclose this actual sample."
                        )
            # Preserve completed work if a later provider request fails.
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for asset, series in [("EURUSD", "DEXUSEU"), ("USDJPY", "DEXJPUS")]:
        if asset not in args.assets:
            continue
        manifest["files"][f"{asset}_FRED_daily"] = download_fred_fx(asset, args, previous)
        manifest["source_urls"][f"{asset}_FRED"] = f"https://fred.stlouisfed.org/series/{series}"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if "EURUSD" in args.assets or "USDJPY" in args.assets:
        manifest["warnings"].append(
            "FRED supplements the assignment's listed data providers because Nasdaq Data Link returned HTTP403 "
            "and Yahoo EUR/USD begins in 2003. Main daily FX signatures use New York noon fixings; Yahoo hourly FX "
            "uses provider spot quotes. Cross-frequency differences may partly reflect different fixing conventions. "
            "Yahoo daily EUR/USD and USD/JPY both contain suspicious isolated spikes on 2008-12-08, retained untouched "
            "in downloaded data but avoided in the main daily analysis by using standalone FRED series."
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["quality_audit"] = audit_fx_quality(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path.relative_to(ROOT)}", flush=True)
    (ROOT / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("Wrote public data_manifest.json (metadata and hashes only)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Data download failed: {exc}", file=sys.stderr)
        sys.exit(1)
