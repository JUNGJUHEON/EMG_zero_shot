from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

import step42b0_graded_primitive as common
import step42c_balanced_transductive as transductive
import step42c_primary_cache as primary_cache
from project_paths import CAPSTONE_ROOT as CAPSTONE, CODE_DIR


STEP_ROOT = (
    CAPSTONE
    / "11. Result"
    / "NinaproEMBridge"
    / "DB5_cross_subject"
    / "step42c_balanced_transductive"
)
PHASE1 = STEP_ROOT / "phase1_frozen_predictions" / "primary_s1_s10_20260722_v1"
QUERY_SEAL = STEP_ROOT / "query_cache_seal" / "primary_s1_s10_20260722_v1"
OUTPUT = STEP_ROOT / "phase2_evaluation" / "primary_s1_s10_20260722_v1"
PROTOCOL = CODE_DIR / "STEP42C_PRIMARY_EVALUATION_PROTOCOL.md"
CLASSES = np.asarray([1, 2, 3, 4, 7], dtype=int)


def sha256_file(path: Path) -> str:
    return primary_cache.sha256_file(path)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def metrics(truth: np.ndarray, prediction: np.ndarray, prefix: str) -> dict[str, object]:
    recalls = recall_score(truth, prediction, labels=CLASSES, average=None, zero_division=0)
    result: dict[str, object] = {
        f"{prefix}_balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        f"{prefix}_macro_f1": float(
            f1_score(truth, prediction, labels=CLASSES, average="macro", zero_division=0)
        ),
        f"{prefix}_accuracy": float(accuracy_score(truth, prediction)),
        f"{prefix}_prediction_coverage": int(len(set(prediction.tolist()))),
    }
    for class_id, value in zip(CLASSES, recalls):
        result[f"{prefix}_recall_B{class_id}"] = float(value)
    return result


def score_frozen_predictions(
    predictions: pd.DataFrame,
    window_manifest: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trial_truth = (
        truth.groupby(["subject", "opaque_trial_id"], as_index=False)
        .agg(
            truth=("class", "first"),
            repetition=("repetition", "first"),
            n_windows=("opaque_row_id", "size"),
            truth_class_count=("class", "nunique"),
            repetition_count=("repetition", "nunique"),
        )
    )
    if not trial_truth["truth_class_count"].eq(1).all() or not trial_truth[
        "repetition_count"
    ].eq(1).all():
        raise AssertionError("Step42C sealed trial maps to multiple targets")
    trial_truth = trial_truth.drop(columns=["truth_class_count", "repetition_count"])
    scored_trials = predictions.merge(
        trial_truth,
        on=["subject", "opaque_trial_id"],
        how="left",
        validate="many_to_one",
    )
    if scored_trials["truth"].isna().any() or len(scored_trials) != len(predictions):
        raise AssertionError("Step42C frozen predictions do not join to trial targets")

    window_truth = truth[["subject", "opaque_row_id", "opaque_trial_id", "class"]].rename(
        columns={"class": "truth"}
    )
    scored_window_base = window_manifest.merge(
        window_truth,
        on=["subject", "opaque_row_id", "opaque_trial_id"],
        how="left",
        validate="one_to_one",
    )
    if scored_window_base["truth"].isna().any():
        raise AssertionError("Step42C frozen window manifest does not join to targets")

    metric_rows: list[dict[str, object]] = []
    scored_windows: list[pd.DataFrame] = []
    grouping = ["subject", "candidate_id", "control_type", "control_index"]
    for key, trial_group in scored_trials.groupby(grouping, sort=False):
        subject, candidate_id, control_type, control_index = key
        trial_group = trial_group.copy()
        window_predictions = scored_window_base.merge(
            trial_group[["subject", "opaque_trial_id", "prediction"]],
            on=["subject", "opaque_trial_id"],
            how="inner",
            validate="many_to_one",
        )
        if len(window_predictions) != len(scored_window_base[scored_window_base["subject"].eq(subject)]):
            raise AssertionError("Step42C did not propagate every trial decision to its windows")
        row: dict[str, object] = {
            "subject": subject,
            "candidate_id": candidate_id,
            "control_type": control_type,
            "control_index": int(control_index),
            "n_trials": len(trial_group),
            "n_windows": len(window_predictions),
        }
        row.update(
            metrics(
                trial_group["truth"].to_numpy(dtype=int),
                trial_group["prediction"].to_numpy(dtype=int),
                "trial",
            )
        )
        row.update(
            metrics(
                window_predictions["truth"].to_numpy(dtype=int),
                window_predictions["prediction"].to_numpy(dtype=int),
                "window",
            )
        )
        metric_rows.append(row)
        window_predictions.insert(1, "candidate_id", candidate_id)
        window_predictions.insert(2, "control_type", control_type)
        window_predictions.insert(3, "control_index", int(control_index))
        scored_windows.append(window_predictions)
    return pd.DataFrame(metric_rows), scored_trials, pd.concat(scored_windows, ignore_index=True)


def primary_gate(
    evaluation_metrics: pd.DataFrame,
    selected_details: pd.DataFrame,
    trial_truth: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    primary_subjects = set(transductive.PRIMARY_SUBJECTS)
    primary = evaluation_metrics[evaluation_metrics["subject"].isin(primary_subjects)]
    selected = primary[primary["control_type"].eq("selected")].set_index("subject")
    independent = primary[primary["control_type"].eq("independent_pose")].set_index("subject")
    balanced = primary[primary["control_type"].eq("balanced_pose")].set_index("subject")
    semantic = (
        primary[primary["control_type"].eq("semantic_permutation")]
        .groupby("subject")["trial_balanced_accuracy"]
        .mean()
    )
    shuffle = (
        primary[primary["control_type"].eq("cluster_feature_permutation")]
        .groupby("subject")["trial_balanced_accuracy"]
        .mean()
    )
    order = list(transductive.PRIMARY_SUBJECTS)
    selected_score = selected.loc[order, "trial_balanced_accuracy"]
    independent_score = independent.loc[order, "trial_balanced_accuracy"]
    balanced_score = balanced.loc[order, "trial_balanced_accuracy"]
    semantic_score = semantic.loc[order]
    shuffle_score = shuffle.loc[order]

    detail = selected_details[selected_details["subject"].isin(primary_subjects)].merge(
        trial_truth[["subject", "opaque_trial_id", "truth"]],
        on=["subject", "opaque_trial_id"],
        how="left",
        validate="one_to_one",
    )
    oracle_by_subject = detail.groupby("subject").apply(
        lambda frame: transductive.cluster_oracle_accuracy(
            frame["selected_cluster"].to_numpy(dtype=int),
            frame["truth"].to_numpy(dtype=int),
        ),
        include_groups=False,
    )
    independent_gain = (selected_score - independent_score).to_numpy(dtype=float)
    balanced_gain = (selected_score - balanced_score).to_numpy(dtype=float)
    semantic_gain = (selected_score - semantic_score).to_numpy(dtype=float)
    shuffle_gain = (selected_score - shuffle_score).to_numpy(dtype=float)
    independent_ci = transductive.bootstrap_interval(independent_gain, seed=42_210)
    balanced_ci = transductive.bootstrap_interval(balanced_gain, seed=42_211)
    semantic_ci = transductive.bootstrap_interval(semantic_gain, seed=42_212)

    selected_trial_rows = primary[
        primary["control_type"].eq("selected")
    ]
    scored_selected = detail.groupby("subject")["selected_prediction"].value_counts()
    exact_prior = all(
        len(scored_selected.loc[subject]) == 5
        and set(scored_selected.loc[subject].to_numpy(dtype=int).tolist()) == {6}
        for subject in order
    )
    finite_columns = ["trial_balanced_accuracy", "trial_macro_f1", "window_balanced_accuracy"]
    checks = {
        "mean_selected_trial_at_least_0p50": bool(selected_score.mean() >= 0.50),
        "mean_gain_over_independent_at_least_0p05": bool(independent_gain.mean() >= 0.05),
        "independent_gain_bootstrap_low_above_zero": bool(independent_ci[0] > 0.0),
        "independent_gain_sign_flip_p_below_0p05": bool(
            common.exact_one_sided_sign_flip_p(independent_gain) < 0.05
        ),
        "at_least_6_of_9_beat_independent": bool(np.sum(independent_gain > 0) >= 6),
        "mean_gain_over_balanced_pose_at_least_0p03": bool(balanced_gain.mean() >= 0.03),
        "balanced_pose_gain_bootstrap_low_above_zero": bool(balanced_ci[0] > 0.0),
        "balanced_pose_gain_sign_flip_p_below_0p05": bool(
            common.exact_one_sided_sign_flip_p(balanced_gain) < 0.05
        ),
        "at_least_6_of_9_beat_balanced_pose": bool(np.sum(balanced_gain > 0) >= 6),
        "mean_gain_over_semantic_permutation_at_least_0p15": bool(
            semantic_gain.mean() >= 0.15
        ),
        "semantic_gain_bootstrap_low_above_zero": bool(semantic_ci[0] > 0.0),
        "semantic_gain_sign_flip_p_below_0p05": bool(
            common.exact_one_sided_sign_flip_p(semantic_gain) < 0.05
        ),
        "at_least_6_of_9_beat_semantic_permutation": bool(
            np.sum(semantic_gain > 0) >= 6
        ),
        "mean_cluster_oracle_at_least_0p70": bool(oracle_by_subject.mean() >= 0.70),
        "mean_gain_over_cluster_shuffle_at_least_0p03": bool(shuffle_gain.mean() >= 0.03),
        "all_selected_predictions_exact_six_per_class": bool(exact_prior),
        "all_metrics_finite": bool(
            np.isfinite(primary[finite_columns].to_numpy(dtype=float)).all()
        ),
    }
    gate = {
        "promotion_pass": bool(all(checks.values())),
        "checks": checks,
        "chance": 0.20,
        "mean_selected_trial_balanced_accuracy": float(selected_score.mean()),
        "mean_selected_window_balanced_accuracy": float(
            selected.loc[order, "window_balanced_accuracy"].mean()
        ),
        "mean_selected_trial_macro_f1": float(
            selected.loc[order, "trial_macro_f1"].mean()
        ),
        "mean_independent_trial_balanced_accuracy": float(independent_score.mean()),
        "mean_gain_over_independent": float(independent_gain.mean()),
        "gain_over_independent_bootstrap_95pct": list(independent_ci),
        "gain_over_independent_sign_flip_p": common.exact_one_sided_sign_flip_p(
            independent_gain
        ),
        "subjects_beating_independent": int(np.sum(independent_gain > 0)),
        "mean_balanced_pose_trial_balanced_accuracy": float(balanced_score.mean()),
        "mean_gain_over_balanced_pose": float(balanced_gain.mean()),
        "gain_over_balanced_pose_bootstrap_95pct": list(balanced_ci),
        "gain_over_balanced_pose_sign_flip_p": common.exact_one_sided_sign_flip_p(
            balanced_gain
        ),
        "subjects_beating_balanced_pose": int(np.sum(balanced_gain > 0)),
        "mean_semantic_permutation_trial_balanced_accuracy": float(semantic_score.mean()),
        "mean_gain_over_semantic_permutation": float(semantic_gain.mean()),
        "gain_over_semantic_permutation_bootstrap_95pct": list(semantic_ci),
        "gain_over_semantic_permutation_sign_flip_p": common.exact_one_sided_sign_flip_p(
            semantic_gain
        ),
        "subjects_beating_semantic_permutation": int(np.sum(semantic_gain > 0)),
        "mean_cluster_shuffle_trial_balanced_accuracy": float(shuffle_score.mean()),
        "mean_gain_over_cluster_shuffle": float(shuffle_gain.mean()),
        "mean_cluster_oracle_accuracy": float(oracle_by_subject.mean()),
    }
    subject_summary = pd.DataFrame(
        {
            "subject": order,
            "selected_trial_balanced_accuracy": selected_score.to_numpy(),
            "independent_trial_balanced_accuracy": independent_score.to_numpy(),
            "balanced_pose_trial_balanced_accuracy": balanced_score.to_numpy(),
            "semantic_permutation_trial_balanced_accuracy": semantic_score.to_numpy(),
            "cluster_shuffle_trial_balanced_accuracy": shuffle_score.to_numpy(),
            "cluster_oracle_accuracy": oracle_by_subject.loc[order].to_numpy(),
            "selected_minus_independent": independent_gain,
            "selected_minus_balanced_pose": balanced_gain,
            "selected_minus_semantic_permutation": semantic_gain,
            "selected_minus_cluster_shuffle": shuffle_gain,
        }
    )
    return gate, subject_summary


def make_plots(
    evaluation_metrics: pd.DataFrame,
    scored_trials: pd.DataFrame,
    selected_details: pd.DataFrame,
    trial_truth: pd.DataFrame,
    subject_summary: pd.DataFrame,
    gate: dict[str, object],
) -> None:
    primary_subjects = set(transductive.PRIMARY_SUBJECTS)
    primary = evaluation_metrics[evaluation_metrics["subject"].isin(primary_subjects)]
    selected = primary[primary["control_type"].eq("selected")]
    values = [
        selected["trial_balanced_accuracy"].mean(),
        primary[primary["control_type"].eq("independent_pose")][
            "trial_balanced_accuracy"
        ].mean(),
        primary[primary["control_type"].eq("balanced_pose")][
            "trial_balanced_accuracy"
        ].mean(),
        primary[primary["control_type"].eq("semantic_permutation")]
        .groupby("subject")["trial_balanced_accuracy"]
        .mean()
        .mean(),
        primary[primary["control_type"].eq("cluster_feature_permutation")]
        .groupby("subject")["trial_balanced_accuracy"]
        .mean()
        .mean(),
        gate["mean_cluster_oracle_accuracy"],
    ]
    labels = ["Selected BCN", "Independent", "Balanced pose", "Semantic perm.", "Cluster shuffle", "Cluster oracle"]
    colors = ["#2878B5", "#777777", "#4C72B0", "#C44E52", "#DD8452", "#55A868"]
    selected_trials = scored_trials[
        scored_trials["subject"].isin(primary_subjects)
        & scored_trials["control_type"].eq("selected")
    ]
    matrix = confusion_matrix(
        selected_trials["truth"], selected_trials["prediction"], labels=CLASSES
    ).astype(float)
    matrix /= np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes[0, 0].bar(labels, values, color=colors)
    axes[0, 0].axhline(0.2, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel("Trial balanced accuracy")
    axes[0, 0].set_title("Frozen endpoint and controls")

    gains = subject_summary.set_index("subject")
    x = np.arange(len(gains))
    axes[0, 1].bar(x - 0.18, gains["selected_minus_independent"], width=0.36, label="vs independent", color="#55A868")
    axes[0, 1].bar(x + 0.18, gains["selected_minus_balanced_pose"], width=0.36, label="vs balanced pose", color="#2878B5")
    axes[0, 1].axhline(0, color="black", linewidth=1)
    axes[0, 1].set_xticks(x, gains.index)
    axes[0, 1].set_ylabel("Balanced-accuracy gain")
    axes[0, 1].set_title("Per-subject frozen gains")
    axes[0, 1].legend()

    image = axes[1, 0].imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axes[1, 0].set_xticks(range(len(CLASSES)), [f"B{x}" for x in CLASSES])
    axes[1, 0].set_yticks(range(len(CLASSES)), [f"B{x}" for x in CLASSES])
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_title("Selected BCN row-normalized confusion")
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            axes[1, 0].text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white" if matrix[row, column] > 0.5 else "black")
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046)

    axes[1, 1].axis("off")
    status = "PASS" if gate["promotion_pass"] else "FAIL"
    lines = [f"Step42C primary gate: {status}", ""]
    lines.extend(
        f"{'PASS' if passed else 'FAIL'}  {name}"
        for name, passed in gate["checks"].items()
    )
    axes[1, 1].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)
    fig.suptitle(
        "Step42C frozen balanced-transductive endpoint\n"
        "Target A1-12+B5/B6 fit; unlabeled B1/B2/B3/B4/B7 query batch",
        fontsize=15,
    )
    fig.savefig(OUTPUT / "primary_endpoint.png", dpi=180)
    plt.close(fig)

    class_recalls = selected[["subject", *[f"trial_recall_B{x}" for x in CLASSES]]].melt(
        id_vars="subject", var_name="class", value_name="recall"
    )
    class_recalls["class"] = class_recalls["class"].str.replace("trial_recall_", "", regex=False)
    class_summary = class_recalls.groupby("class", as_index=False)["recall"].mean()
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.bar(class_summary["class"], class_summary["recall"], color=["#2878B5", "#55A868", "#C44E52", "#8172B2", "#CCB974"])
    ax.axhline(0.2, color="black", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean trial recall")
    ax.set_title("Step42C primary class recall across s2-s10")
    fig.savefig(OUTPUT / "primary_class_recall.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite Step42C Phase 2: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    phase1_verification = json.loads(
        (PHASE1 / "verification" / "verified_phase1_predictions.json").read_text(
            encoding="utf-8"
        )
    )
    prediction_seal = json.loads(
        (PHASE1 / "prediction_seal.json").read_text(encoding="utf-8")
    )
    if not phase1_verification["verified"] or not prediction_seal["frozen"]:
        raise AssertionError("Step42C Phase 2 lacks a verified frozen prediction seal")
    if sha256_file(PHASE1 / "frozen_trial_predictions.csv") != prediction_seal[
        "frozen_trial_predictions_sha256"
    ]:
        raise AssertionError("Step42C frozen predictions changed before evaluation")
    query_config = json.loads((QUERY_SEAL / "config.json").read_text(encoding="utf-8"))
    truth_path = Path(query_config["truth_seal"])
    if sha256_file(truth_path) != query_config["truth_seal_sha256"]:
        raise AssertionError("Step42C sealed targets changed before evaluation")

    predictions = pd.read_csv(PHASE1 / "frozen_trial_predictions.csv")
    window_manifest = pd.read_csv(PHASE1 / "frozen_window_manifest.csv")
    selected_details = pd.read_csv(PHASE1 / "selected_trial_details.csv")
    truth = pd.read_csv(truth_path)
    evaluation_metrics, scored_trials, scored_windows = score_frozen_predictions(
        predictions, window_manifest, truth
    )
    trial_truth = (
        truth.groupby(["subject", "opaque_trial_id"], as_index=False)["class"]
        .first()
        .rename(columns={"class": "truth"})
    )
    gate, subject_summary = primary_gate(
        evaluation_metrics, selected_details, trial_truth
    )

    evaluation_metrics.to_csv(OUTPUT / "evaluation_metrics.csv", index=False)
    scored_trials.to_csv(OUTPUT / "scored_trial_predictions.csv", index=False)
    scored_windows.to_csv(OUTPUT / "scored_window_predictions.csv", index=False)
    subject_summary.to_csv(OUTPUT / "subject_summary.csv", index=False)
    write_json(OUTPUT / "primary_gate.json", gate)
    make_plots(
        evaluation_metrics,
        scored_trials,
        selected_details,
        trial_truth,
        subject_summary,
        gate,
    )

    config = {
        "created": datetime.now().astimezone().isoformat(),
        "evaluation_only": True,
        "model_fit_performed": False,
        "normalization_performed": False,
        "selection_performed": False,
        "clustering_performed": False,
        "predictions_modified": False,
        "phase1": str(PHASE1),
        "phase1_prediction_seal_sha256": sha256_file(PHASE1 / "prediction_seal.json"),
        "phase1_verification_sha256": sha256_file(
            PHASE1 / "verification" / "verified_phase1_predictions.json"
        ),
        "frozen_predictions_sha256": sha256_file(
            PHASE1 / "frozen_trial_predictions.csv"
        ),
        "query_seal": str(QUERY_SEAL),
        "sealed_targets_sha256": sha256_file(truth_path),
        "primary_protocol": str(PROTOCOL),
        "primary_protocol_sha256": sha256_file(PROTOCOL),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "primary_subjects": list(transductive.PRIMARY_SUBJECTS),
        "diagnostic_subject": "s1",
        "classes": CLASSES.tolist(),
        "b8_included": False,
    }
    write_json(OUTPUT / "config.json", config)
    status = "PASS" if gate["promotion_pass"] else "FAIL"
    summary = "\n".join(
        [
            "# Step42C Frozen Balanced-Transductive Primary Evaluation",
            "",
            f"Status: **{status}**",
            "",
            "The evaluator joined a verified frozen prediction seal to the sealed",
            "B1/B2/B3/B4/B7 targets. It performed no fit, normalization, selection,",
            "clustering, or prediction update.",
            "",
            "```json",
            json.dumps(gate, indent=2, sort_keys=True, default=json_default),
            "```",
            "",
            "This endpoint is balanced transductive ZSL with a known five-class",
            "inventory and six-trial class prior. It is not inductive ZSL.",
            "",
        ]
    )
    (OUTPUT / "summary.md").write_text(summary, encoding="utf-8")
    files = []
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            files.append(
                {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    write_json(OUTPUT / "artifact_manifest.json", {"files": files})
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    print(f"Step42C Phase 2 output: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
