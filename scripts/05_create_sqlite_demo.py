from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SQL_DIR = BASE_DIR / "reports" / "sql_outputs"
SQL_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROCESSED_DIR / "sme_fpi.sqlite"


QUERIES = {
    "latest_hidden_stress_ranking": """
        WITH latest AS (
            SELECT MAX(period_sort) AS max_period
            FROM panel
            WHERE Hidden_SME_Stress IS NOT NULL
        )
        SELECT
            p.country_name,
            p.REF_AREA,
            p.TIME_PERIOD,
            ROUND(p.SME_FPI_equal_z, 3) AS SME_FPI_z,
            ROUND(p.CISS_z, 3) AS CISS_z,
            ROUND(p.Hidden_SME_Stress, 3) AS Hidden_SME_Stress,
            RANK() OVER (ORDER BY p.Hidden_SME_Stress DESC) AS stress_gap_rank
        FROM panel p
        JOIN latest l ON p.period_sort = l.max_period
        ORDER BY p.Hidden_SME_Stress DESC;
    """,
    "average_index_by_country": """
        SELECT
            country_name,
            REF_AREA,
            ROUND(AVG(SME_FPI_equal_z), 3) AS avg_SME_FPI_z,
            ROUND(AVG(Hidden_SME_Stress), 3) AS avg_hidden_gap,
            COUNT(*) AS periods
        FROM panel
        GROUP BY country_name, REF_AREA
        ORDER BY avg_SME_FPI_z DESC;
    """,
    "cluster_counts": """
        SELECT
            cluster_name,
            COUNT(*) AS country_periods,
            ROUND(AVG(SME_FPI_equal_z), 3) AS avg_index
        FROM panel
        WHERE cluster_name IS NOT NULL
        GROUP BY cluster_name
        ORDER BY avg_index DESC;
    """,
    "firm_size_component_means": """
        SELECT
            component,
            FIRM_SIZE,
            ROUND(AVG(OBS_VALUE), 3) AS mean_weighted_percentage,
            COUNT(*) AS observations
        FROM safe_long
        WHERE FIRM_SECTOR = 'A'
          AND REF_AREA NOT IN ('U2', 'T2')
        GROUP BY component, FIRM_SIZE
        ORDER BY component, mean_weighted_percentage DESC;
    """,
    "big_cube_problem_severity": """
        SELECT
            problem,
            firm_size_label,
            ROUND(AVG(severity_score_1_10), 3) AS avg_severity,
            COUNT(*) AS cells
        FROM problem_severity_cube
        WHERE FIRM_SECTOR = 'A'
          AND FIRM_AGE = 0
          AND REF_AREA NOT IN ('U2', 'T2')
        GROUP BY problem, firm_size_label
        ORDER BY problem, avg_severity DESC;
    """,
}


def main() -> None:
    panel = pd.read_csv(PROCESSED_DIR / "sme_fpi_panel.csv")
    safe_long = pd.read_csv(PROCESSED_DIR / "safe_component_long.csv")
    loadings = pd.read_csv(PROCESSED_DIR / "pca_loadings.csv")
    profiles = pd.read_csv(PROCESSED_DIR / "sme_fpi_cluster_profiles.csv")
    big_cube_path = PROCESSED_DIR / "safe_problem_severity_cube.csv"
    big_cube = pd.read_csv(big_cube_path) if big_cube_path.exists() else None

    with sqlite3.connect(DB_PATH) as con:
        panel.to_sql("panel", con, if_exists="replace", index=False)
        safe_long.to_sql("safe_long", con, if_exists="replace", index=False)
        loadings.to_sql("pca_loadings", con, if_exists="replace", index=False)
        profiles.to_sql("cluster_profiles", con, if_exists="replace", index=False)
        if big_cube is not None:
            big_cube.to_sql("problem_severity_cube", con, if_exists="replace", index=False)

        for name, query in QUERIES.items():
            out = pd.read_sql_query(query, con)
            out.to_csv(SQL_DIR / f"{name}.csv", index=False)
            print(f"saved {SQL_DIR / f'{name}.csv'}")

    sql_file = BASE_DIR / "reports" / "sql_queries.sql"
    sql_file.write_text(
        "\n\n".join(f"-- {name}\n{query.strip()}" for name, query in QUERIES.items()),
        encoding="utf-8",
    )
    print(f"saved {DB_PATH}")
    print(f"saved {sql_file}")


if __name__ == "__main__":
    main()
