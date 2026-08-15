from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import step42c_balanced_transductive as step42c


class Step42CBalancedTransductiveTests(unittest.TestCase):
    def test_proxy_panels_are_fixed_and_balanced(self) -> None:
        specs = step42c.proxy_specs()
        a_specs = [item for item in specs if item.held_exercise == "A"]
        self.assertEqual(len(specs), 13)
        self.assertEqual(len(a_specs), 12)
        self.assertTrue(all(len(item.held_classes) == 5 for item in a_specs))
        counts = {
            class_id: sum(class_id in item.held_classes for item in a_specs)
            for class_id in range(1, 13)
        }
        self.assertEqual(set(counts.values()), {5})

    def test_candidate_grid_is_locked(self) -> None:
        candidates = step42c.candidate_specs()
        self.assertEqual(len(candidates), 11)
        self.assertEqual(candidates[0].candidate_id, "independent_pose")
        self.assertEqual(candidates[1].candidate_id, "balanced_pose")
        self.assertEqual(sum(item.algorithm == "bcn" for item in candidates), 9)

    def test_proxy_masks_hold_whole_classes(self) -> None:
        rows = []
        for exercise, classes in (("A", range(1, 13)), ("B", (5, 6))):
            for class_id in classes:
                for repetition in step42c.REPETITIONS:
                    rows.append(
                        {
                            "subject": "s2",
                            "exercise_code": exercise,
                            "class": class_id,
                            "repetition": repetition,
                        }
                    )
        manifest = pd.DataFrame(rows)
        for proxy in step42c.proxy_specs():
            fit, query = step42c.proxy_masks(manifest, proxy)
            self.assertFalse(np.any(fit & query))
            self.assertEqual(
                set(manifest.loc[query, "class"]), set(proxy.held_classes)
            )
            self.assertEqual(query.sum(), len(proxy.held_classes) * 6)

    def test_exact_capacity_assignment_obeys_prior(self) -> None:
        scores = np.asarray(
            [
                [3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 2.0],
            ]
        )
        assignments = step42c.exact_capacity_assignment(scores, capacity=2)
        np.testing.assert_array_equal(assignments, [0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(np.bincount(assignments), [2, 2, 2])

    def test_balanced_kmeans_is_deterministic_and_exact(self) -> None:
        values = np.asarray(
            [
                [-3.1, 0.0],
                [-2.9, 0.1],
                [0.0, 3.1],
                [0.1, 2.9],
                [3.0, 0.0],
                [3.1, -0.1],
            ]
        )
        first = step42c.balanced_kmeans(values, k=3, capacity=2, seed=42, starts=8)
        second = step42c.balanced_kmeans(values, k=3, capacity=2, seed=42, starts=8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertAlmostEqual(first[2], second[2])
        self.assertEqual(set(np.bincount(first[0], minlength=3).tolist()), {2})

    def test_cluster_oracle_recovers_permuted_cluster_names(self) -> None:
        assignments = np.asarray([2, 2, 0, 0, 1, 1])
        truth = np.asarray([10, 10, 20, 20, 30, 30])
        self.assertEqual(step42c.cluster_oracle_accuracy(assignments, truth), 1.0)

    def test_candidate_inference_does_not_require_truth(self) -> None:
        trial_ids = np.asarray([f"q{index:03d}" for index in range(6)])
        pose = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
            ]
        )
        episode = step42c.UnlabeledEpisode(
            subject="s2",
            proxy_id="synthetic",
            catalog=np.asarray([1, 2, 3]),
            trial_ids=trial_ids,
            window_trial_index=np.arange(6),
            window_row_keys=np.arange(6),
            predicted_pose_trial=pose,
            predicted_pose_window=pose,
            source_prototypes=np.eye(3),
            emg_trial_views={name: pose for name in step42c.FEATURE_SETS},
            n_fit_windows=12,
            ridge_alpha=1.0,
            source_subjects=("s3",),
        )
        balanced = step42c.predict_candidate(
            episode, step42c.get_candidate("balanced_pose")
        )
        np.testing.assert_array_equal(balanced.predictions, [1, 1, 2, 2, 3, 3])
        bcn = step42c.predict_candidate(
            episode, step42c.get_candidate("bcn__mav16__posew_0")
        )
        np.testing.assert_array_equal(bcn.predictions, [1, 1, 2, 2, 3, 3])

    def test_loso_source_scope_never_contains_target(self) -> None:
        self.assertEqual(step42c.source_subjects_for("s1"), step42c.PRIMARY_SUBJECTS)
        for subject in step42c.PRIMARY_SUBJECTS:
            sources = step42c.source_subjects_for(subject)
            self.assertNotIn(subject, sources)
            self.assertEqual(len(sources), 8)

    def test_outer_selection_excludes_outer_subject(self) -> None:
        rows = []
        for subject in step42c.SUBJECTS:
            for candidate in step42c.candidate_specs():
                value = 0.8 if candidate.candidate_id == "balanced_pose" else 0.2
                rows.append(
                    {
                        "subject": subject,
                        "candidate_id": candidate.candidate_id,
                        "algorithm": candidate.algorithm,
                        "feature_set": candidate.feature_set,
                        "pose_view_weight": candidate.pose_view_weight,
                        "a_trial_balanced_accuracy": value,
                        "a_window_balanced_accuracy": value,
                        "a_mean_true_pose_margin": value,
                        "a_cluster_oracle_accuracy": np.nan,
                        "b56_trial_balanced_accuracy": value,
                        "b56_window_balanced_accuracy": value,
                        "b56_cluster_oracle_accuracy": np.nan,
                    }
                )
        selections = step42c.outer_subject_select(pd.DataFrame(rows))
        self.assertTrue(selections["selected_candidate_id"].eq("balanced_pose").all())
        for row in selections.itertuples(index=False):
            if row.outer_subject != "s1":
                self.assertNotIn(row.outer_subject, row.development_subjects.split(","))


if __name__ == "__main__":
    unittest.main()
