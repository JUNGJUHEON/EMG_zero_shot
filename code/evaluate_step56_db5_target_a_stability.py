from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate_step42c_primary_phase2 as evaluator
import step52_db5_margin_gate as step52
import step56_db5_target_a_stability as step56


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = values[
        rng.integers(0, len(values), size=(draws, len(values)))
    ].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def sign_flip(values: np.ndarray) -> float:
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def subject_plot(frame: pd.DataFrame, output: Path) -> None:
    order = [f"s{index}" for index in range(1, 11)]
    table = frame.set_index("subject").loc[order]
    positions = np.arange(len(order))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.bar(
        positions - width / 2,
        table["parent_ba"],
        width,
        color="#7A8793",
        label="Frozen margin-gated parent",
    )
    bars = ax.bar(
        positions + width / 2,
        table["selected_ba"],
        width,
        color="#2F7D67",
        label="Target-A stability selection",
    )
    bars[3].set_edgecolor("#B23A48")
    bars[3].set_linewidth(2.5)
    for index, value in enumerate(table["delta"]):
        if abs(value) > 1e-12:
            y = max(table.iloc[index]["parent_ba"], table.iloc[index]["selected_ba"])
            ax.text(index, y + 0.025, f"{value:+.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.2, color="#222222", linestyle="--", linewidth=1, label="5-class chance")
    ax.set_xticks(positions, order)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Trial balanced accuracy")
    ax.set_title("DB5 target-A stability selection versus frozen parent")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def proxy_transfer_plot(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    colors = [
        "#B23A48"
        if subject == "s4"
        else "#2F7D67"
        if delta > 1e-12
        else "#7A8793"
        for subject, delta in zip(frame["subject"], frame["delta"])
    ]
    ax.scatter(frame["stability_gain"], frame["delta"], s=75, c=colors, alpha=0.9)
    offsets = {
        "s1": (5, 8),
        "s3": (5, 5),
        "s4": (5, 5),
        "s5": (-20, -16),
        "s6": (5, 8),
        "s7": (5, -16),
    }
    for row in frame.itertuples(index=False):
        if row.stability_gain <= 1e-12:
            continue
        ax.annotate(
            row.subject,
            (row.stability_gain, row.delta),
            xytext=offsets.get(row.subject, (5, 4)),
            textcoords="offset points",
        )
    ax.axhline(0, color="#222222", linewidth=1)
    ax.axvline(0, color="#222222", linewidth=1)
    ax.set_xlabel("Legal A-proxy stability gain")
    ax.set_ylabel("Evaluated B trial-BA gain")
    ax.set_title("Do legal proxy gains transfer to evaluated gestures?")
    ax.grid(color="#D9DEE3", linewidth=0.7)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-dir",
        default=str(
            step56.RESULT_ROOT
            / "phase2_primary_prediction"
            / "db5_target_a_stability_20260812_v1"
        ),
    )
    parser.add_argument("--run-name", default="db5_target_a_stability_20260812_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_dir = Path(args.prediction_dir)
    lock_path = prediction_dir / "prediction_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    predictions_path = prediction_dir / "trial_predictions.csv"
    windows_path = prediction_dir / "window_manifest.csv"
    if lock["status"] != "predictions_locked_before_truth_join" or lock["truth_opened"]:
        raise AssertionError("Step56 prediction lock failed")
    if step56.sha256_file(predictions_path) != lock["trial_predictions_sha256"]:
        raise AssertionError("Step56 predictions changed")
    if step56.sha256_file(windows_path) != lock["window_manifest_sha256"]:
        raise AssertionError("Step56 window manifest changed")
    output = step56.RESULT_ROOT / "phase3_evaluation" / args.run_name
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    truth_path = step52.QUERY_SEAL / "truth_seal.csv"
    predictions = pd.read_csv(predictions_path)
    windows = pd.read_csv(windows_path)
    truth = pd.read_csv(truth_path)
    evaluation, scored_trials, scored_windows = evaluator.score_frozen_predictions(
        predictions, windows, truth
    )
    evaluation.to_csv(output / "evaluation_metrics.csv", index=False)
    scored_trials.to_csv(output / "scored_trial_predictions.csv", index=False)
    scored_windows.to_csv(output / "scored_window_predictions.csv", index=False)
    order = [f"s{index}" for index in range(1, 11)]
    selected = evaluation[evaluation["control_type"].eq("selected")].set_index("subject")
    parent = evaluation[evaluation["control_type"].eq("step52_parent")].set_index("subject")
    selected_ba = selected.loc[order, "trial_balanced_accuracy"].to_numpy(float)
    parent_ba = parent.loc[order, "trial_balanced_accuracy"].to_numpy(float)
    delta = selected_ba - parent_ba
    low, high = bootstrap(delta, seed=85600)
    selections_path = (
        step56.RESULT_ROOT
        / "phase1_legal_proxy"
        / args.run_name
        / "frozen_subject_selections.csv"
    )
    selections = pd.read_csv(selections_path).set_index("subject").loc[order]
    subject_frame = pd.DataFrame(
        {
            "subject": order,
            "selected_candidate_id": selections["selected_candidate_id"].to_numpy(),
            "parent_candidate_id": selections["parent_candidate_id"].to_numpy(),
            "stability_gain": selections["stability_gain"].to_numpy(float),
            "selected_ba": selected_ba,
            "parent_ba": parent_ba,
            "delta": delta,
        }
    )
    subject_frame.to_csv(output / "subject_comparison.csv", index=False)
    max_subject_loss = float(np.min(delta))
    promotion_checks = {
        "mean_gain_at_least_0p01": bool(float(delta.mean()) >= 0.01 - 1e-15),
        "bootstrap_lower_above_zero": bool(low > 0),
        "no_subject_loses_more_than_0p10": bool(max_subject_loss >= -0.10 - 1e-15),
        "improved_at_least_worsened": bool(np.sum(delta > 1e-15) >= np.sum(delta < -1e-15)),
    }
    promoted = bool(all(promotion_checks.values()))
    summary = {
        "created": datetime.now().astimezone().isoformat(),
        "status": "post_hoc_db5_only_internal_candidate",
        "selected_mean_trial_balanced_accuracy": float(selected_ba.mean()),
        "parent_mean_trial_balanced_accuracy": float(parent_ba.mean()),
        "mean_gain": float(delta.mean()),
        "bootstrap_95_interval": [low, high],
        "exact_two_sided_sign_flip_p": sign_flip(delta),
        "n_improved": int(np.sum(delta > 1e-15)),
        "n_equal": int(np.sum(np.abs(delta) <= 1e-15)),
        "n_worse": int(np.sum(delta < -1e-15)),
        "largest_subject_loss": max_subject_loss,
        "s4_selected_ba": float(subject_frame.loc[subject_frame["subject"].eq("s4"), "selected_ba"].iloc[0]),
        "s4_parent_ba": float(subject_frame.loc[subject_frame["subject"].eq("s4"), "parent_ba"].iloc[0]),
        "s4_gain": float(subject_frame.loc[subject_frame["subject"].eq("s4"), "delta"].iloc[0]),
        "promotion_checks": promotion_checks,
        "promoted": promoted,
        "prediction_lock_sha256": step56.sha256_file(lock_path),
        "truth_seal_sha256": step56.sha256_file(truth_path),
    }
    step56.write_json(output / "headline_metrics.json", summary)
    subject_plot(subject_frame, output / "subject_comparison.png")
    proxy_transfer_plot(subject_frame, output / "proxy_to_endpoint_transfer.png")
    lines = [
        "# Step56 DB5 Target-A Stability Result",
        "",
        f"- selected mean trial BA: `{selected_ba.mean():.4f}`",
        f"- frozen Step52 parent: `{parent_ba.mean():.4f}`",
        f"- paired mean gain: `{delta.mean():+.4f}`",
        f"- subject-bootstrap 95% interval: `[{low:+.4f}, {high:+.4f}]`",
        f"- exact two-sided sign-flip p: `{sign_flip(delta):.4f}`",
        f"- improved/equal/worse: `{np.sum(delta > 1e-15)}/{np.sum(np.abs(delta) <= 1e-15)}/{np.sum(delta < -1e-15)}`",
        f"- s4 BA: `{summary['s4_parent_ba']:.4f} -> {summary['s4_selected_ba']:.4f}` (`{summary['s4_gain']:+.4f}`)",
        f"- promoted: `{promoted}`",
        "",
        "This is DB5-only post hoc internal evidence. Selection used legal target A/B5/B6 proxies; evaluated B labels were joined only after prediction lock.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(subject_frame.to_string(index=False), flush=True)
    print(f"Step56 evaluation output: {output}", flush=True)


if __name__ == "__main__":
    main()
