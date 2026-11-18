"""Regression check for training-only hyperparameter search in a two-stack run."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import experiment as lab
from tuning import evaluate_configured, model_roles


class NestedSearchTest(unittest.TestCase):
    def test_evaluation_rows_are_excluded_from_tuning_and_upstream_fits(self):
        frame = pd.DataFrame({"stroke": [0, 1] * 120, "age": [0, 1, 2, 3] * 60})
        train, held = frame.iloc[:200], frame.iloc[200:]
        roles, upstream = model_roles(
            ["Decision tree", "KNN"], "Logistic regression",
            ["AdaBoost", "Gaussian Naive Bayes"], "Logistic regression", "Logistic regression",
        )
        configuration = {
            "enabled": True, "method": "Grid search", "metric": "Average precision",
            "folds": 2, "seed": 42, "budget": 2, "profiles": {
                "Stack 1 base · Decision tree": {"fixed": {}, "search": {"max_depth": [2, 3]}},
                "Final combiner · Logistic regression": {"fixed": {}, "search": {"C": [0.1, 1.0]}},
            },
        }
        trials = []

        def check_base(training, evaluation, features, seed, models, options):
            self.assertTrue(set(training.index).isdisjoint(evaluation.index))
            self.assertTrue(set(training.index).isdisjoint(held.index))
            return {models[0]: np.asarray(evaluation.age) / 4}, {}

        def check_features(training, features, method, count):
            self.assertTrue(set(training.index).isdisjoint(held.index))
            return features

        with (patch.object(lab, "fit_base", side_effect=check_base),
              patch.object(lab, "select_features", side_effect=check_features),
              patch.object(lab, "combine_scores", side_effect=lambda name, x, y, evaluation, options: evaluation.mean(axis=1))):
            scores, _, _ = evaluate_configured(
                train, held, ["age"], "As selected", 1, roles, upstream,
                configuration, trials, 1, "Selected predictors", lambda *args: None,
            )

        self.assertEqual(len(scores["Final stack"]), len(held))
        audit = pd.DataFrame(trials)
        self.assertEqual(set(audit.Role), {"Stack 1 base", "Final combiner"})
        self.assertTrue((audit.groupby("Search fit").Selected.sum() == 1).all())
        self.assertTrue((audit["Training rows"] <= len(train)).all())


if __name__ == "__main__":
    unittest.main()

