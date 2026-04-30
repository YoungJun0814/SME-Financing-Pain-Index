from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import RequestsDependencyWarning


warnings.filterwarnings("ignore", category=RequestsDependencyWarning)


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ECB_API = "https://data-api.ecb.europa.eu/service/data"
EUROSTAT_API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/sts_rb_q"
COUNTRIES = ["AT", "BE", "DE", "ES", "FI", "FR", "GR", "IE", "IT", "NL", "PT", "SK"]
COUNTRY_KEY = "+".join(COUNTRIES)


ECB_FORECAST_SERIES = [
    {
        "dataset": "BLS",
        "key": f"Q.{COUNTRY_KEY}.ALL.SME.E.Z.B3.ST.S.FNET",
        "file": "forecast_bls_credit_standards_sme.csv",
        "variable": "bls_credit_standards_sme",
        "label": "BLS SME credit standards, past three months",
        "frequency": "quarterly",
    },
    {
        "dataset": "BLS",
        "key": f"Q.{COUNTRY_KEY}.ALL.SME.E.Z.F3.ST.S.FNET",
        "file": "forecast_bls_credit_standards_sme_fwd.csv",
        "variable": "bls_credit_standards_sme_fwd",
        "label": "BLS SME credit standards, next three months",
        "frequency": "quarterly",
    },
    {
        "dataset": "BLS",
        "key": f"Q.{COUNTRY_KEY}.ALL.O.E.SME.B3.TC.S.FNET",
        "file": "forecast_bls_terms_conditions_sme.csv",
        "variable": "bls_terms_conditions_sme",
        "label": "BLS SME loan terms and conditions, past three months",
        "frequency": "quarterly",
    },
    {
        "dataset": "BLS",
        "key": f"Q.{COUNTRY_KEY}.ALL.SME.E.Z.B3.ZZ.D.FNET",
        "file": "forecast_bls_loan_demand_sme.csv",
        "variable": "bls_loan_demand_sme",
        "label": "BLS SME loan demand, past three months",
        "frequency": "quarterly",
    },
    {
        "dataset": "BLS",
        "key": f"Q.{COUNTRY_KEY}.ALL.O.E.Z.B3.RA.D.FNET",
        "file": "forecast_bls_rejection_share_enterprise.csv",
        "variable": "bls_rejection_share_enterprise",
        "label": "BLS rejected enterprise loan applications, past three months",
        "frequency": "quarterly",
    },
    {
        "dataset": "MIR",
        "key": f"M.{COUNTRY_KEY}.B.A2A.A.R.2.2240.EUR.N",
        "file": "forecast_mir_small_loan_rate.csv",
        "variable": "mir_small_loan_rate",
        "label": "MIR new business rate, corporate loans up to EUR 0.25M",
        "frequency": "monthly",
    },
    {
        "dataset": "MIR",
        "key": f"M.{COUNTRY_KEY}.B.A2A.A.R.1.2240.EUR.N",
        "file": "forecast_mir_large_loan_rate.csv",
        "variable": "mir_large_loan_rate",
        "label": "MIR new business rate, corporate loans over EUR 1M",
        "frequency": "monthly",
    },
    {
        "dataset": "MIR",
        "key": f"M.{COUNTRY_KEY}.B.A2A.A.B.2.2240.EUR.N",
        "file": "forecast_mir_small_loan_volume.csv",
        "variable": "mir_small_loan_volume",
        "label": "MIR new business volume, corporate loans up to EUR 0.25M",
        "frequency": "monthly",
    },
]

EUROSTAT_FORECAST_SERIES = [
    {
        "indicator": "BKRT",
        "file": "forecast_eurostat_bankruptcies_index.csv",
        "variable": "eurostat_bankruptcies_index",
        "label": "Eurostat business bankruptcy declarations index, 2015=100",
        "frequency": "quarterly",
    },
    {
        "indicator": "REG",
        "file": "forecast_eurostat_registrations_index.csv",
        "variable": "eurostat_registrations_index",
        "label": "Eurostat business registrations index, 2015=100",
        "frequency": "quarterly",
    },
]


def download(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            dest.write_bytes(response.content)
            print(f"saved {dest.name} ({len(response.content):,} bytes)")
            return
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 2 * attempt
            print(f"download failed ({exc}); retrying in {wait}s")
            time.sleep(wait)


def download_ecb_series() -> pd.DataFrame:
    records = []
    for series in ECB_FORECAST_SERIES:
        url = (
            f"{ECB_API}/{series['dataset']}/{series['key']}"
            "?format=csvdata&startPeriod=2009-01"
        )
        dest = RAW_DIR / series["file"]
        try:
            download(url, dest)
            status = "downloaded"
        except Exception as exc:
            status = f"failed: {exc}"
            print(f"could not download {series['variable']}: {exc}")
        records.append({**series, "status": status, "url": url})
    meta = pd.DataFrame(records)
    meta.to_csv(RAW_DIR / "forecast_external_series_metadata.csv", index=False)
    return meta


def parse_eurostat_json(payload: dict, country: str, variable: str, indicator: str) -> pd.DataFrame:
    time_index = payload.get("dimension", {}).get("time", {}).get("category", {}).get("index", {})
    values = payload.get("value", {})
    if not time_index or not values:
        return pd.DataFrame(columns=["REF_AREA", "TIME_PERIOD", "OBS_VALUE", "indic_bt"])
    position_to_period = {int(position): period for period, position in time_index.items()}
    rows = []
    for position, period in sorted(position_to_period.items()):
        value = values.get(str(position))
        if value is None:
            continue
        rows.append(
            {
                "REF_AREA": country,
                "TIME_PERIOD": period,
                "OBS_VALUE": value,
                "indic_bt": indicator,
                "variable": variable,
            }
        )
    return pd.DataFrame(rows)


def download_eurostat_series() -> pd.DataFrame:
    records = []
    for series in EUROSTAT_FORECAST_SERIES:
        frames = []
        for country in COUNTRIES:
            url = (
                f"{EUROSTAT_API}?format=JSON&lang=en&freq=Q"
                f"&indic_bt={series['indicator']}"
                "&nace_r2=B-S_X_O_S94&s_adj=NSA&unit=I15"
                f"&geo={country}"
            )
            try:
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                frames.append(parse_eurostat_json(response.json(), country, series["variable"], series["indicator"]))
                status = "downloaded"
            except Exception as exc:
                status = f"failed for {country}: {exc}"
                print(f"could not download {series['variable']} for {country}: {exc}")
            time.sleep(0.15)
        frames = [frame for frame in frames if not frame.empty]
        data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        dest = RAW_DIR / series["file"]
        data.to_csv(dest, index=False)
        print(f"saved {dest.name} ({len(data):,} rows)")
        records.append(
            {
                "dataset": "Eurostat sts_rb_q",
                "key": series["indicator"],
                "file": series["file"],
                "variable": series["variable"],
                "label": series["label"],
                "frequency": series["frequency"],
                "status": "downloaded" if not data.empty else status,
                "url": EUROSTAT_API,
            }
        )
    meta = pd.DataFrame(records)
    meta.to_csv(RAW_DIR / "forecast_eurostat_series_metadata.csv", index=False)
    return meta


def main() -> None:
    ecb_meta = download_ecb_series()
    eurostat_meta = download_eurostat_series()
    meta = pd.concat([ecb_meta, eurostat_meta], ignore_index=True, sort=False)
    meta.to_csv(RAW_DIR / "forecast_external_series_metadata.csv", index=False)
    print(meta[["variable", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
