import importlib.util
import unittest
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ForecastingLayerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layer = import_module_from_path(
            "forecasting_layer",
            BASE_DIR / "scripts" / "11_build_forecasting_layer.py",
        )
        cls.app = import_module_from_path("dashboard_app", BASE_DIR / "dashboard" / "app.py")

    def test_model_suite_and_validation_outputs_exist(self):
        evaluation = pd.read_csv(DATA_DIR / "forecasting_model_evaluation.csv")
        required_models = {
            "naive",
            "country_mean",
            "country_ar1",
            "momentum",
            "pooled_lag_ols",
            "elastic_net",
            "ridge",
            "random_forest",
            "gradient_boosting",
        }
        self.assertTrue(required_models.issubset(set(evaluation["model_key"])))

        expected_outputs = [
            "forecast_model_predictions_all_origins.csv",
            "decision_board_all_origins.csv",
            "forecasting_country_error.csv",
            "risk_tier_validation.csv",
        ]
        for filename in expected_outputs:
            path = DATA_DIR / filename
            self.assertTrue(path.exists(), f"Missing {filename}")
            self.assertGreater(len(pd.read_csv(path)), 0, f"{filename} is empty")

    def test_forecast_feature_columns_do_not_leak_future_targets(self):
        frame = pd.read_csv(DATA_DIR / "forecasting_feature_panel.csv")
        cols = self.layer.feature_columns(frame, "SME_FPI_equal_z")
        leaked = [
            col
            for col in cols
            if col == "SME_FPI_equal_z" or col.startswith("future_") or col.startswith("target_")
        ]
        self.assertEqual(leaked, [])

    def test_source_catalog_explains_external_predictors(self):
        catalog = pd.read_csv(DATA_DIR / "dashboard_source_catalog.csv")
        datasets = set(catalog["dataset"].astype(str))
        self.assertIn("forecast_eurostat_bankruptcies_index.csv", datasets)
        self.assertIn("forecast_eurostat_registrations_index.csv", datasets)
        self.assertTrue(catalog["dashboard_use"].astype(str).str.contains("not part of SME-FPI Core").any())

    def test_dashboard_new_figures_render(self):
        period_range = [0, len(self.app.PERIODS) - 1]
        countries = self.app.ALL_COUNTRY_CODES
        board = self.app.build_decision_board(period_range, countries, "SME_FPI_equal_z")

        self.assertGreater(len(board), 0)
        self.assertGreater(len(self.app.make_risk_history_figure(period_range, countries).data), 0)
        self.assertGreater(len(self.app.make_tier_validation_figure().data), 0)
        self.assertGreater(len(self.app.make_model_rank_heatmap().data), 0)
        self.assertEqual(type(self.app.decision_narrative(board)).__name__, "Div")
        self.assertEqual(type(self.app.diagnosis_cards(board)).__name__, "Div")


if __name__ == "__main__":
    unittest.main()
