"""Regression checks for source integrity and the declared analysis interval."""

from functools import partial
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_analysis
from src.analysis import intraday_analysis


@pytest.fixture
def input_manifest(tmp_path):
    """Eight separately fingerprinted inputs, without downloading market data."""
    keys = [
        "SPY_daily", "SPY_hourly", "USDJPY_daily", "USDJPY_hourly",
        "EURUSD_daily", "EURUSD_hourly", "USDJPY_FRED_daily", "EURUSD_FRED_daily",
    ]
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    manifest = {"files": {}}
    for key in keys:
        path = raw / f"{key}.csv"
        path.write_text(f"source,value\n{key},1\n", encoding="utf-8")
        manifest["files"][key] = {
            "path": f"data/raw/{key}.csv",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest


def test_manifest_accepts_all_eight_fingerprinted_inputs(input_manifest):
    root, expected = input_manifest
    actual = run_analysis.verify_input_manifest(root)
    assert len(actual["files"]) == 8
    assert actual == expected


def test_modified_input_stops_before_existing_results_are_overwritten(input_manifest, monkeypatch):
    root, _ = input_manifest
    (root / "data" / "raw" / "USDJPY_FRED_daily.csv").write_text(
        "source,value\nchanged,999\n", encoding="utf-8"
    )
    out = root / "outputs"
    out.mkdir()
    saved = out / "daily_signatures.csv"
    saved.write_bytes(b"previous verified results\n")
    verifier = run_analysis.verify_input_manifest
    monkeypatch.setattr(run_analysis, "verify_input_manifest", partial(verifier, root))
    monkeypatch.setattr(run_analysis, "OUT", out)

    def unexpected_analysis():
        pytest.fail("Analysis began before validating the input fingerprints.")

    monkeypatch.setattr(run_analysis, "style", unexpected_analysis)
    with pytest.raises(ValueError, match="Input hash mismatch"):
        run_analysis.main()
    assert saved.read_bytes() == b"previous verified results\n"


@pytest.mark.parametrize("missing", ["entry", "file"])
def test_missing_required_source_is_rejected(input_manifest, missing):
    root, manifest = input_manifest
    if missing == "entry":
        del manifest["files"]["EURUSD_FRED_daily"]
        (root / "data_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        error, message = ValueError, "Missing or inconsistent manifest entry"
    else:
        (root / "data" / "raw" / "EURUSD_FRED_daily.csv").unlink()
        error, message = FileNotFoundError, "Required input missing"
    with pytest.raises(error, match=message):
        run_analysis.verify_input_manifest(root)


@pytest.mark.parametrize("asset", ["EURUSD", "USDJPY"])
def test_daily_fx_never_substitutes_yahoo_when_fred_is_missing(tmp_path, asset):
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    # A valid Yahoo file is present, so a fallback would silently succeed.
    (raw / f"{asset}_daily.csv").write_text(
        "date,adj_close\n2000-01-03,1.0\n2000-01-04,1.1\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError):
        run_analysis.daily_data(asset, root=tmp_path)


def write_hourly_fixture(root, frame):
    raw = root / "data" / "raw"
    raw.mkdir(parents=True)
    frame.to_csv(raw / "EURUSD_hourly.csv", index=False)
    return {"files": {"EURUSD_hourly": {"request": {
        "start": "2024-10-01", "end_exclusive": "2026-01-01",
    }}}}


def test_hourly_interval_includes_start_and_excludes_end_and_provider_spillover(tmp_path):
    source = pd.DataFrame({
        "timestamp": ["2024-09-30T23:00:00Z", "2024-10-01T00:00:00Z",
                      "2025-12-31T23:00:00Z", "2026-01-01T00:00:00Z"],
        "open": [1.0, 1.1, 1.2, 1.3], "close": [1.01, 1.11, 1.21, 1.31],
    })
    manifest = write_hourly_fixture(tmp_path, source)
    actual = run_analysis.load_hourly("EURUSD", manifest, root=tmp_path)
    assert actual["timestamp"].tolist() == ["2024-10-01T00:00:00Z", "2025-12-31T23:00:00Z"]
    assert actual["open"].tolist() == [1.1, 1.2]


def test_partial_previous_date_does_not_start_six_month_windows_early(tmp_path):
    frames = [pd.DataFrame({"timestamp": ["2024-09-30T23:00:00Z"],
                            "open": [1.0], "close": [1.001]})]
    for day in pd.bdate_range("2024-10-01", "2025-04-02"):
        timestamps = pd.date_range(day, periods=19, freq="h", tz="UTC")
        frames.append(pd.DataFrame({
            "timestamp": timestamps.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "open": np.exp(np.arange(19) * 0.001),
            "close": np.exp(np.arange(1, 20) * 0.001),
        }))
    manifest = write_hourly_fixture(tmp_path, pd.concat(frames, ignore_index=True))
    hourly = run_analysis.load_hourly("EURUSD", manifest, root=tmp_path)
    _, rolling, audit = intraday_analysis(hourly, run_analysis.ASSETS["EURUSD"])
    assert audit["date"].min() == "2024-10-01"
    assert rolling.index.tolist() == [pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-02")]
