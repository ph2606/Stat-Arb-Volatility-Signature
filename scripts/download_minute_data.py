"""Retrieve genuine Yahoo Finance one- and five-minute bars for Assignment 1.

The fixed request windows record the September 24, 2026 acquisition. Yahoo's
rolling retention means an exact future re-download is not guaranteed. A normal
run verifies and reuses the recorded local snapshot without contacting Yahoo;
``--refresh`` explicitly authorizes a fresh acquisition. Historical daily/hourly
inputs and their original manifest are never changed by this script.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "minute_data_manifest.json"
ASSETS = {"SPY": "SPY", "USDJPY": "JPY=X", "EURUSD": "EURUSD=X"}
DEFAULT_STARTS = {"1m": "2026-08-25T21:10:00Z", "5m": "2026-07-26T21:10:00Z"}
DEFAULT_END = "2026-09-24T00:00:00Z"
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iso(timestamp: pd.Timestamp) -> str:
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(manifest: dict) -> None:
    manifest["manifest_updated_utc"] = utc_now()
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def history(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return yf.Ticker(symbol).history(
        start=start.to_pydatetime(), end=end.to_pydatetime(), interval=interval,
        auto_adjust=False, actions=False, prepost=False, repair=False,
        keepna=True, timeout=30,
    )


def request_record(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    return {"symbol": symbol, "interval": interval, "start": iso(start),
            "end_exclusive": iso(end), "auto_adjust": False, "prepost": False}


def probe_older(symbol: str, interval: str, start: pd.Timestamp) -> dict:
    """Record one older, short request, avoiding a confounding chunk-length error."""
    probe_start = start - pd.Timedelta(days=2)
    probe_end = start - pd.Timedelta(days=1)
    record = {"purpose": "Test availability beyond the observed rolling retention boundary",
              "request": request_record(symbol, interval, probe_start, probe_end),
              "attempted_utc": utc_now()}
    try:
        data = history(symbol, interval, probe_start, probe_end)
        record.update({"outcome": "returned_rows" if len(data) else "empty", "rows": len(data)})
        if len(data):
            record.update({"observed_start": iso(data.index.min()), "observed_end": iso(data.index.max())})
            raise RuntimeError("Older probe succeeded: extend the requested window before claiming retention coverage")
    except RuntimeError:
        raise
    except Exception as exc:
        record.update({"outcome": "provider_error", "rows": 0,
                       "error_type": type(exc).__name__, "message": str(exc)})
    return record


def normalize(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    if data.empty or data.index.tz is None:
        raise ValueError("Expected nonempty timezone-aware provider bars")
    missing = set(PRICE_COLUMNS + ["Volume"]) - set(data.columns)
    if missing:
        raise ValueError(f"Provider response missing columns: {sorted(missing)}")
    prices = data[PRICE_COLUMNS]
    valid_price = np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1)
    valid_ohlc = (data["High"] >= prices.max(axis=1)) & (data["Low"] <= prices.min(axis=1))
    timestamps = data.index.tz_convert("UTC")
    in_range = (timestamps >= start) & (timestamps < end)
    valid = valid_price & valid_ohlc & in_range
    result = pd.DataFrame({
        "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
        **{name.lower(): data[name].to_numpy() for name in PRICE_COLUMNS},
        "volume": data["Volume"].to_numpy(),
    }).loc[valid.to_numpy()].sort_values("timestamp", kind="stable").reset_index(drop=True)
    duplicate_rows = result.loc[result["timestamp"].duplicated(keep=False)]
    if not duplicate_rows.empty:
        for _, group in duplicate_rows.groupby("timestamp", sort=False):
            if len(group.drop_duplicates()) != 1:
                raise ValueError("Conflicting duplicate timestamps from Yahoo; manual review required")
    duplicate_count = int(result["timestamp"].duplicated().sum())
    result = result.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    if result.empty:
        raise ValueError("No valid bars remained after normalization")
    diagnostics = {
        "provider_rows": int(len(data)),
        "dropped_nonpositive_or_nonfinite_price_rows": int((~valid_price).sum()),
        "dropped_inconsistent_ohlc_rows": int((valid_price & ~valid_ohlc).sum()),
        "dropped_outside_requested_range_rows": int((valid_price & valid_ohlc & ~in_range).sum()),
        "dropped_identical_duplicate_rows": duplicate_count,
        "conflicting_duplicate_rows": 0,
        "total_dropped_rows": int(len(data) - len(result)),
        "zero_volume_rows": int(result["volume"].eq(0).sum()),
        "missing_volume_rows": int(result["volume"].isna().sum()),
        "zero_close_change_rows": int(result["close"].diff().eq(0).sum()),
    }
    if diagnostics["total_dropped_rows"] != sum(diagnostics[name] for name in [
        "dropped_nonpositive_or_nonfinite_price_rows", "dropped_inconsistent_ohlc_rows",
        "dropped_outside_requested_range_rows", "dropped_identical_duplicate_rows",
    ]):
        raise AssertionError("Cleaning diagnostics do not reconcile")
    return result, diagnostics


def acquire(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict]]:
    frames, attempts = [], []
    cursor = start
    # One-minute requests are split into nonoverlapping <=7-day UTC intervals.
    # Five-minute bars use one request covering the available retention window.
    chunk_span = pd.Timedelta(days=7 if interval == "1m" else 60)
    while cursor < end:
        chunk_end = min(cursor + chunk_span, end)
        error = None
        for attempt in range(1, 4):
            record = {"request": request_record(symbol, interval, cursor, chunk_end),
                      "attempt": attempt, "attempted_utc": utc_now()}
            try:
                frame = history(symbol, interval, cursor, chunk_end)
                if frame.empty:
                    raise ValueError("Yahoo returned an empty requested chunk")
                record.update({"outcome": "returned_rows", "rows": len(frame),
                               "observed_start": iso(frame.index.min()), "observed_end": iso(frame.index.max())})
                attempts.append(record)
                frames.append(frame)
                error = None
                break
            except Exception as exc:
                error = exc
                record.update({"outcome": "provider_error", "error_type": type(exc).__name__, "message": str(exc)})
                attempts.append(record)
                if attempt < 3:
                    time.sleep(2 ** (attempt - 1))
        if error is not None:
            raise RuntimeError(f"Failed {symbol} {interval} {iso(cursor)}: {error}") from error
        cursor = chunk_end
    return pd.concat(frames).sort_index(kind="stable"), attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-1m", default=DEFAULT_STARTS["1m"])
    parser.add_argument("--start-5m", default=DEFAULT_STARTS["5m"])
    parser.add_argument("--end", default=DEFAULT_END, help="Exclusive UTC end; defaults to completed UTC dates")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    starts = {"1m": pd.Timestamp(args.start_1m), "5m": pd.Timestamp(args.start_5m)}
    end = pd.Timestamp(args.end)
    if end.tzinfo is None or any(value.tzinfo is None or value >= end for value in starts.values()):
        raise ValueError("Supply timezone-aware starts strictly before the end")
    if end > pd.Timestamp.now(tz="UTC").normalize():
        raise ValueError("End must exclude the current incomplete UTC calendar day")
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    expected_keys = {f"{asset}_{interval}" for asset in ASSETS for interval in starts}
    # Fail closed on altered/missing provenance rather than silently accepting different data.
    cached = {}
    if not args.refresh:
        for asset, symbol in ASSETS.items():
            for interval, start in starts.items():
                key = f"{asset}_{interval}"
                path = ROOT / "data" / "raw" / f"{key}.csv"
                if path.exists():
                    entry = previous.get("files", {}).get(key)
                    if not entry or entry.get("request") != request_record(symbol, interval, start, end) or sha256(path) != entry.get("sha256"):
                        raise RuntimeError(f"Existing {key} does not match its manifest/request; review and use --refresh deliberately")
                    cached[key] = entry
                    print(f"Verified cached {key}: {entry['rows']:,} rows", flush=True)
        if set(cached) == expected_keys:
            print("Verified all six snapshots; no network requests or files changed.", flush=True)
            return
    (ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    cache = ROOT / ".cache" / "yfinance"
    cache.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache))
    yf.config.debug.hide_exceptions = False
    manifest = previous if previous and not args.refresh else {
        "schema_version": 1,
        "source": "Yahoo Finance chart API via yfinance",
        "source_urls": {asset: f"https://finance.yahoo.com/quote/{symbol}/history/" for asset, symbol in ASSETS.items()},
        "api_endpoint_template": "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
        "software": {"python": platform.python_version(), **{pkg: importlib.metadata.version(pkg) for pkg in ["yfinance", "pandas", "numpy"]}},
        "conventions": {
            "timestamp": "UTC ISO-8601 bar-start timestamp; OHLC refers to the following interval.",
            "end": "Exclusive cutoff at 2026-09-24 00:00 UTC; all samples exclude the unfinished retrieval day.",
            "prices": "Unadjusted genuine provider OHLC; SPY regular market hours only; FX quotes are not centralized trade prices.",
            "one_minute_chunks": "Nonoverlapping UTC request intervals of at most seven days, merged in chronological order.",
            "missing_data": "No interpolation, forward filling, synthetic observations, or provider splicing.",
            "duplicates": "Identical duplicate bars are counted and removed; conflicting duplicates stop the download.",
            "fx_volume": "Yahoo FX volume is commonly zero; it is not a measure of traded market volume.",
            "raw_retention": "Raw CSV files and yfinance caches remain local and are excluded from Git.",
            "oldest_boundary": "Starts lie just inside the rolling 30-day (1m) and 60-day (5m) windows; older short requests are recorded below.",
        },
        "scope": "Longest windows sought from the accessible public Yahoo endpoint at acquisition, subject to observed rolling retention; no claim about subscription-provider maximum history.",
        "warnings": ["Yahoo's rolling retention and revisions prevent guaranteed later retrieval of the identical fixed snapshot.",
                     "Minute samples are a recent extension and are not observations from the assignment's historical 2000–2007, 2007–2014, or 2020–2025 windows."],
        "availability_probes": [], "files": {},
    }
    for asset, symbol in ASSETS.items():
        for interval, start in starts.items():
            key = f"{asset}_{interval}"
            if key in cached:
                continue
            print(f"Probing older {asset} {interval} availability", flush=True)
            manifest["availability_probes"].append(probe_older(symbol, interval, start))
            write_manifest(manifest)
            print(f"Downloading {asset} {interval}: {iso(start)} to {iso(end)} (exclusive)", flush=True)
            data, attempts = acquire(symbol, interval, start, end)
            result, diagnostics = normalize(data, start, end)
            path = ROOT / "data" / "raw" / f"{key}.csv"
            result.to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
            analysis_timezone = "America/New_York" if asset == "SPY" else "UTC"
            counts = pd.to_datetime(result["timestamp"], utc=True).dt.tz_convert(analysis_timezone).dt.strftime("%Y-%m-%d").value_counts()
            manifest["files"][key] = {
                "path": path.relative_to(ROOT).as_posix(), "request": request_record(symbol, interval, start, end),
                "downloaded_utc": utc_now(), "source": "Yahoo Finance via yfinance",
                "source_url": manifest["source_urls"][asset], "sha256": sha256(path), "bytes": path.stat().st_size,
                "rows": len(result), "columns": result.columns.tolist(),
                "observed_start": result["timestamp"].iloc[0], "observed_end": result["timestamp"].iloc[-1],
                "provider_timezone": str(data.index.tz), "analysis_timezone": analysis_timezone,
                "analysis_calendar_dates": int(len(counts)), "bars_per_calendar_date_min": int(counts.min()),
                "bars_per_calendar_date_median": float(counts.median()), "bars_per_calendar_date_max": int(counts.max()),
                "diagnostics": diagnostics, "requests": attempts,
            }
            write_manifest(manifest)
            print(f"Saved {key}: {len(result):,} rows, {result['timestamp'].iloc[0]} through {result['timestamp'].iloc[-1]}", flush=True)
    print("Wrote minute_data_manifest.json; original historical inputs are unchanged.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Minute-data download failed: {exc}", file=sys.stderr)
        sys.exit(1)
