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
            "country_arima",
            "country_arimax_ciss",
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

    def test_forecast_outputs_include_reader_facing_diagnostics(self):
        feature_panel = pd.read_csv(DATA_DIR / "forecasting_feature_panel.csv")
        for col in ["availability_pressure_z", "cost_pressure_z", "salience_pressure_z"]:
            self.assertIn(col, feature_panel.columns)

        decision = pd.read_csv(DATA_DIR / "forecast_decision_board.csv")
        for col in ["signal_type", "agreement_quality", "primary_drivers"]:
            self.assertIn(col, decision.columns)
        self.assertTrue(decision["signal_type"].astype(str).str.len().gt(0).all())

        summary = pd.read_csv(DATA_DIR / "forecasting_layer_summary.csv")
        for col in [
            "rolling_origin_count",
            "ml_beats_strongest_baseline_count",
            "median_ml_edge_vs_strongest_baseline",
            "most_frequent_best_model",
        ]:
            self.assertIn(col, summary.columns)

    def test_dashboard_new_figures_render(self):
        period_range = [0, len(self.app.PERIODS) - 1]
        countries = self.app.ALL_COUNTRY_CODES
        board = self.app.build_decision_board(period_range, countries, "SME_FPI_equal_z")

        self.assertGreater(len(board), 0)
        self.assertIn("signal_type", board.columns)
        self.assertIn("agreement_quality", board.columns)
        self.assertGreater(len(self.app.make_risk_history_figure(period_range, countries).data), 0)
        self.assertGreater(len(self.app.make_tier_validation_figure().data), 0)
        self.assertGreater(len(self.app.make_forecast_loss_figure(countries).data), 0)
        self.assertGreater(len(self.app.make_model_rank_heatmap().data), 0)
        self.assertEqual(type(self.app.decision_narrative(board)).__name__, "Div")
        self.assertEqual(type(self.app.standardization_caveat("SME_FPI_equal_z", period_range)).__name__, "Div")
        self.assertEqual(type(self.app.risk_rule_guide()).__name__, "Div")
        self.assertEqual(type(self.app.no_overclaim_panel()).__name__, "Div")
        self.assertEqual(type(self.app.diagnosis_cards(board)).__name__, "Div")
        self.assertEqual(type(self.app.professor_submission_panel()).__name__, "Div")
        self.assertEqual(type(self.app.professor_glossary_panel()).__name__, "Div")
        self.assertEqual(type(self.app.professor_reading_path()).__name__, "Div")
        self.assertEqual(type(self.app.professor_start_cards()).__name__, "Div")
        self.assertEqual(type(self.app.hero_monitor_snapshot()).__name__, "Div")
        self.assertEqual(type(self.app.command_center_panel(board, "SME_FPI_equal_z", period_range)).__name__, "Div")
        self.assertEqual(type(self.app.visual_bullets(["a", "b"])).__name__, "Ul")
        self.assertEqual(type(self.app.compact_copy_bullets([("Read", "Compact supporting copy")])).__name__, "Ul")
        self.assertEqual(type(self.app.compact_summary_panel("Kicker", "Title", ["one", "two"])).__name__, "Div")
        checkpoint = self.app.reader_checkpoint("Question", "Answer", "Read", "Caution")
        self.assertEqual(getattr(checkpoint, "className", ""), "rc-grid")
        self.assertEqual(len(checkpoint.children), 4)
        self.assertTrue(all("rc-card-" in getattr(card, "className", "") for card in checkpoint.children))
        self.assertEqual(type(self.app.defense_question_panel()).__name__, "Div")
        self.assertEqual(type(self.app.forecast_defense_panel()).__name__, "Div")
        self.assertEqual(type(self.app.forecast_audit_card()).__name__, "Div")
        self.assertEqual(type(self.app.forecast_view_selector()).__name__, "Div")
        self.assertEqual(type(self.app.data_role_taxonomy_panel()).__name__, "Div")
        self.assertEqual(type(self.app.data_lineage_map()).__name__, "Div")
        self.assertEqual(type(self.app.claim_evidence_matrix()).__name__, "Div")
        self.assertEqual(type(self.app.research_value_panel()).__name__, "Div")
        self.assertEqual(type(self.app.pca_answer_panel()).__name__, "Div")
        self.assertEqual(type(self.app.firm_survey_pressure_panel()).__name__, "Div")
        self.assertEqual(type(self.app.reader_lens_panel("first")).__name__, "Div")
        self.assertEqual(type(self.app.reader_lens_panel("defense")).__name__, "Div")
        self.assertEqual(type(self.app.interaction_hub()).__name__, "Section")
        self.assertEqual(type(self.app.reading_path_detail("forecast")).__name__, "list")
        self.assertIn("reading-path-node--active", self.app.reading_path_node_class("forecast", "forecast"))
        self.assertEqual(len(self.app.reading_path_node_classes("forecast")), 7)
        self.assertEqual(self.app.forecast_view_styles("benchmark")[2], {"display": "block"})
        self.assertEqual(len(self.app.READING_PATH_STEPS), 7)
        self.assertTrue(all("outcome" in step for step in self.app.READING_PATH_STEPS))
        self.assertGreaterEqual(len(self.app.source_role_options()), 1)
        self.assertEqual(self.app.target_tab_from_button("jump-forecast"), "forecast")
        self.assertEqual(self.app.target_tab_from_button("lens-primary-action", "forecast"), "forecast")
        self.assertEqual(self.app.target_tab_from_button("read-path-methodology"), "methodology")
        self.assertEqual(self.app.target_tab_from_button("unknown"), "start")

    def test_layout_starts_with_start_here_and_reader_friendly_labels(self):
        main = self.app.app.layout.children[1]
        tabs = next(child for child in main.children if getattr(child, "id", None) == "main-tabs")
        self.assertEqual(tabs.value, "start")
        start = next(child for child in tabs.children if getattr(child, "value", None) == "start")
        self.assertEqual(start.label, "Start Here")
        current = next(child for child in tabs.children if getattr(child, "value", None) == "decision")
        self.assertEqual(current.label, "Current Board")
        findings = next(child for child in tabs.children if getattr(child, "value", None) == "findings")
        self.assertEqual(findings.label, "Defense & Findings")
        hidden = next(child for child in tabs.children if getattr(child, "value", None) == "hidden-stress")
        self.assertEqual(hidden.label, "Borrower-Market Gap")
        explorer = next(child for child in tabs.children if getattr(child, "value", None) == "explorer")
        self.assertEqual(explorer.label, "Trend Explorer")
        regimes = next(child for child in tabs.children if getattr(child, "value", None) == "regimes")
        self.assertEqual(regimes.label, "PCA Analysis")
        survey = next(child for child in tabs.children if getattr(child, "value", None) == "big-data")
        self.assertEqual(survey.label, "Firm Survey Detail")


if __name__ == "__main__":
    unittest.main()
