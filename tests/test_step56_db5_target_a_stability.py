from __future__ import annotations

import unittest

import pandas as pd

import step56_db5_target_a_stability as step56


def fixture(*, alternative_b56: float = 1.0, alternative_gain: float = 0.05) -> pd.DataFrame:
    rows = []
    parent_cluster = step56.CLUSTER_IDS[0]
    alternative = step56.candidate_id(step56.CLUSTER_IDS[1], "median")
    for cluster_id in step56.CLUSTER_IDS:
        for method in step56.NAMING_METHODS:
            identifier = step56.candidate_id(cluster_id, method)
            for index in range(12):
                value = 0.4 + (alternative_gain if identifier == alternative else 0.0)
                rows.append(
                    {
                        "candidate_id": identifier,
                        "cluster_candidate_id": cluster_id,
                        "naming_method": method,
                        "proxy": f"A_{index}",
                        "trial_balanced_accuracy": value,
                        "window_balanced_accuracy": value,
                    }
                )
            rows.append(
                {
                    "candidate_id": identifier,
                    "cluster_candidate_id": cluster_id,
                    "naming_method": method,
                    "proxy": "B56",
                    "trial_balanced_accuracy": (
                        alternative_b56 if identifier == alternative else 1.0
                    ),
                    "window_balanced_accuracy": 1.0,
                }
            )
    return pd.DataFrame(rows)


class Step56Tests(unittest.TestCase):
    def test_candidate_family_has_27_members(self) -> None:
        self.assertEqual(len(step56.CLUSTER_IDS) * len(step56.NAMING_METHODS), 27)

    def test_stable_candidate_is_selected(self) -> None:
        selected, _ = step56.select_candidate(fixture(), step56.CLUSTER_IDS[0])
        self.assertEqual(
            selected, step56.candidate_id(step56.CLUSTER_IDS[1], "median")
        )

    def test_b56_guard_rejects_candidate(self) -> None:
        selected, _ = step56.select_candidate(
            fixture(alternative_b56=0.7), step56.CLUSTER_IDS[0]
        )
        self.assertEqual(selected, step56.candidate_id(step56.CLUSTER_IDS[0], "mean"))

    def test_small_stability_gain_rejects_candidate(self) -> None:
        selected, _ = step56.select_candidate(
            fixture(alternative_gain=0.01), step56.CLUSTER_IDS[0]
        )
        self.assertEqual(selected, step56.candidate_id(step56.CLUSTER_IDS[0], "mean"))


if __name__ == "__main__":
    unittest.main()
