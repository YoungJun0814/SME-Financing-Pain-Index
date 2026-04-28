from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ECB_API = "https://data-api.ecb.europa.eu/service/data"


COMPONENTS = [
    {
        "component": "access_finance_main_problem",
        "label": "Access to finance is the most pressing problem",
        "question": "Q0",
        "item": "ZZZZ",
        "answer": "P3",
        "direction": "higher_more_pain",
    },
    {
        "component": "bank_loan_rejected",
        "label": "Bank loan application rejected",
        "question": "Q7B",
        "item": "FBLN",
        "answer": "S4",
        "direction": "higher_more_pain",
    },
    {
        "component": "bank_loan_cost_too_high",
        "label": "Bank loan offer refused because cost was too high",
        "question": "Q7B",
        "item": "FBLN",
        "answer": "S3",
        "direction": "higher_more_pain",
    },
    {
        "component": "bank_loan_limited_amount",
        "label": "Bank loan only granted for a limited amount",
        "question": "Q7B",
        "item": "FBLN",
        "answer": "S6",
        "direction": "higher_more_pain",
    },
    {
        "component": "bank_willingness_deteriorated",
        "label": "Banks' willingness to provide credit deteriorated",
        "question": "Q11",
        "item": "11F",
        "answer": "N3",
        "direction": "higher_more_pain",
    },
    {
        "component": "interest_rates_increased",
        "label": "Interest rates increased",
        "question": "Q10",
        "item": "10A",
        "answer": "N1",
        "direction": "higher_more_pain",
    },
]


def download(url: str, dest: Path, retries: int = 3) -> None:
    """Download a URL to disk with small retry protection."""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            dest.write_bytes(response.content)
            print(f"saved {dest} ({len(response.content):,} bytes)")
            return
        except Exception as exc:
            if attempt == retries:
                raise
            wait = 2 * attempt
            print(f"download failed ({exc}); retrying in {wait}s")
            time.sleep(wait)


def safe_component_url(question: str, item: str, answer: str) -> str:
    # SAFE key dimensions:
    # FREQ.REF_AREA.FIRM_SIZE.FIRM_SECTOR.FIRM_TURNOVER.FIRM_AGE.FIRM_OWNERSHIP.
    # SAFE_QUESTION.SAFE_ITEM.SAFE_ANSWER.SAFE_FILTER.SAFE_DENOM
    #
    # This query intentionally keeps firm size and sector dimensions, so the raw
    # data retain a multidimensional structure for the Big Data visualization task.
    firm_sizes = "SME+MIC+SML+MED+LAR"
    sectors = "A+C+I+S+T"
    key = f"H..{firm_sizes}.{sectors}.0.0.0.{question}.{item}.{answer}.AL.WP"
    return f"{ECB_API}/SAFE/{key}?format=csvdata&startPeriod=2009-S1"


def main() -> None:
    metadata = pd.DataFrame(COMPONENTS)
    metadata.to_csv(RAW_DIR / "safe_component_metadata.csv", index=False)

    series_key_url = (
        f"{ECB_API}/SAFE?detail=serieskeysonly&format=csvdata"
    )
    download(series_key_url, RAW_DIR / "ecb_safe_serieskeys.csv")

    for comp in COMPONENTS:
        url = safe_component_url(comp["question"], comp["item"], comp["answer"])
        dest = RAW_DIR / f"safe_{comp['component']}.csv"
        download(url, dest)

    # ECB New CISS is the official market-side/systemic stress benchmark.
    ciss_key = "D.U2.Z0Z.4F.EC.SS_CIN.IDX"
    ciss_url = f"{ECB_API}/CISS/{ciss_key}?format=csvdata&startPeriod=2009-01-01"
    download(ciss_url, RAW_DIR / "ecb_ciss_new_daily.csv")

    print("Data download complete.")


if __name__ == "__main__":
    main()
