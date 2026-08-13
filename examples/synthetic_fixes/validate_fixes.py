#!/usr/bin/env python3
"""Instance-level, checkpoint-free validation of fixes I-001 and I-004.

The example deliberately uses an analytic posterior instead of fitting TabPFN.  This
isolates the two integration contracts from surrogate fitting quality:

* I-001: TabPFN-v2 returns query rows only, so slicing by ``n_train`` again shifts
  acquisition values away from their candidates.
* I-004: target standardization requires predictive variance to be multiplied by
  ``y_std**2`` when returning to physical units.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class I001Result:
    n_train: int
    n_queries: int
    legacy_queries_kept: int
    fixed_selected_index: int
    fixed_selected_x: float
    fixed_objective: float
    legacy_score_source_index: int
    legacy_selected_index: int
    legacy_selected_x: float
    legacy_objective: float
    index_shift: int
    objective_loss: float


@dataclass(frozen=True)
class I004Result:
    y_std: float
    standardized_variance_at_global_peak: float
    legacy_physical_variance_at_global_peak: float
    fixed_physical_variance_at_global_peak: float
    legacy_selected_index: int
    legacy_selected_x: float
    legacy_objective: float
    fixed_selected_index: int
    fixed_selected_x: float
    fixed_objective: float
    objective_gain: float


def synthetic_objective(x: np.ndarray) -> np.ndarray:
    """A local peak at x=0.25 and a higher, narrower peak at x=0.80."""

    local_peak = 90.0 * np.exp(-((x - 0.25) / 0.12) ** 2)
    global_peak = 100.0 * np.exp(-((x - 0.80) / 0.08) ** 2)
    return local_peak + global_peak


def synthetic_posterior(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return physical mean and standardized variance for the I-004 example.

    The mean represents a surrogate trained near the visible local peak.  Its
    uncertainty is deliberately largest around the unobserved global peak.
    """

    mean_physical = 60.0 + 30.0 * np.exp(-((x - 0.25) / 0.15) ** 2)
    variance_standardized = 0.0025 + 0.2475 * np.exp(-((x - 0.80) / 0.08) ** 2)
    return mean_physical, variance_standardized


def sampling_ucb_scores(
    mean: np.ndarray,
    variance: np.ndarray,
    standard_normal_draws: np.ndarray,
) -> np.ndarray:
    """Mirror GIT-BO SamplingUCB: maximum of posterior-normal draws."""

    samples = mean[:, None] + np.sqrt(variance)[:, None] * standard_normal_draws
    return samples.max(axis=1)


def validate_i001(candidates: np.ndarray, n_train: int = 20) -> I001Result:
    """Compare query alignment before and after I-001 on an oracle posterior."""

    query_scores = synthetic_objective(candidates)

    # Fixed: query score i remains attached to candidate i.
    fixed_index = int(np.argmax(query_scores))

    # Legacy: TabPFN-v2 already returned query-only rows, but production sliced
    # them by n_train again.  The argmax position was then applied to the
    # unsliced candidate array, shifting the selected point by n_train rows.
    legacy_scores = query_scores[n_train:]
    legacy_selected_index = int(np.argmax(legacy_scores))
    legacy_score_source_index = legacy_selected_index + n_train

    truth = synthetic_objective(candidates)
    result = I001Result(
        n_train=n_train,
        n_queries=len(candidates),
        legacy_queries_kept=len(legacy_scores),
        fixed_selected_index=fixed_index,
        fixed_selected_x=float(candidates[fixed_index]),
        fixed_objective=float(truth[fixed_index]),
        legacy_score_source_index=legacy_score_source_index,
        legacy_selected_index=legacy_selected_index,
        legacy_selected_x=float(candidates[legacy_selected_index]),
        legacy_objective=float(truth[legacy_selected_index]),
        index_shift=legacy_score_source_index - legacy_selected_index,
        objective_loss=float(truth[fixed_index] - truth[legacy_selected_index]),
    )

    assert result.legacy_score_source_index == result.fixed_selected_index
    assert result.index_shift == n_train
    assert result.fixed_objective > result.legacy_objective
    return result


def validate_i004(
    candidates: np.ndarray,
    *,
    y_std: float = 25.0,
    sample_count: int = 512,
    seed: int = 2026,
) -> tuple[I004Result, np.ndarray, np.ndarray, np.ndarray]:
    """Compare SamplingUCB with legacy and corrected physical variance."""

    mean, variance_standardized = synthetic_posterior(candidates)
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((len(candidates), sample_count))

    legacy_variance = variance_standardized
    fixed_variance = variance_standardized * y_std**2
    legacy_scores = sampling_ucb_scores(mean, legacy_variance, draws)
    fixed_scores = sampling_ucb_scores(mean, fixed_variance, draws)

    legacy_index = int(np.argmax(legacy_scores))
    fixed_index = int(np.argmax(fixed_scores))
    global_peak_index = int(np.argmax(synthetic_objective(candidates)))
    truth = synthetic_objective(candidates)

    result = I004Result(
        y_std=y_std,
        standardized_variance_at_global_peak=float(
            variance_standardized[global_peak_index]
        ),
        legacy_physical_variance_at_global_peak=float(
            legacy_variance[global_peak_index]
        ),
        fixed_physical_variance_at_global_peak=float(fixed_variance[global_peak_index]),
        legacy_selected_index=legacy_index,
        legacy_selected_x=float(candidates[legacy_index]),
        legacy_objective=float(truth[legacy_index]),
        fixed_selected_index=fixed_index,
        fixed_selected_x=float(candidates[fixed_index]),
        fixed_objective=float(truth[fixed_index]),
        objective_gain=float(truth[fixed_index] - truth[legacy_index]),
    )

    np.testing.assert_allclose(
        result.fixed_physical_variance_at_global_peak,
        result.standardized_variance_at_global_peak * y_std**2,
    )
    assert abs(result.legacy_selected_x - 0.25) <= 0.02
    assert abs(result.fixed_selected_x - 0.80) <= 0.02
    assert result.fixed_objective > result.legacy_objective
    return result, mean, legacy_scores, fixed_scores


def save_plot(
    path: Path,
    candidates: np.ndarray,
    i001: I001Result,
    i004: I004Result,
    posterior_mean: np.ndarray,
    legacy_scores: np.ndarray,
    fixed_scores: np.ndarray,
) -> None:
    """Render the two before/after comparisons."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    truth = synthetic_objective(candidates)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)

    ax = axes[0]
    ax.plot(candidates, truth, color="black", linewidth=2, label="synthetic objective")
    ax.scatter(
        [i001.legacy_selected_x],
        [i001.legacy_objective],
        color="#d95f02",
        s=65,
        zorder=3,
        label="legacy selection",
    )
    ax.scatter(
        [i001.fixed_selected_x],
        [i001.fixed_objective],
        color="#1b9e77",
        s=65,
        zorder=3,
        label="fixed selection",
    )
    ax.set(title="I-001: query/candidate alignment", xlabel="x", ylabel="objective")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(candidates, posterior_mean, "--", color="0.35", label="posterior mean")
    ax.plot(candidates, legacy_scores, color="#d95f02", label="legacy SamplingUCB")
    ax.plot(candidates, fixed_scores, color="#1b9e77", label="fixed SamplingUCB")
    ax.axvline(i004.legacy_selected_x, color="#d95f02", alpha=0.45, linestyle=":")
    ax.axvline(i004.fixed_selected_x, color="#1b9e77", alpha=0.45, linestyle=":")
    ax.set(title="I-004: variance in physical units", xlabel="x", ylabel="acquisition")
    ax.legend(frameon=False)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    candidates = np.linspace(0.0, 1.0, 101)
    i001 = validate_i001(candidates)
    i004, mean, legacy_scores, fixed_scores = validate_i004(candidates)

    payload = {
        "description": "checkpoint-free semantic validation of GIT-BO fixes",
        "candidate_count": len(candidates),
        "i001": asdict(i001),
        "i004": asdict(i004),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "synthetic_fix_validation.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    plot_path = args.output_dir / "synthetic_fix_validation.png"
    if not args.no_plot:
        save_plot(
            plot_path,
            candidates,
            i001,
            i004,
            mean,
            legacy_scores,
            fixed_scores,
        )

    print("I-001 query alignment")
    print(
        f"  legacy: score for candidate[{i001.legacy_score_source_index}] "
        f"was applied to candidate[{i001.legacy_selected_index}], "
        f"x={i001.legacy_selected_x:.2f}, f={i001.legacy_objective:.6f}"
    )
    print(
        f"  fixed:  candidate[{i001.fixed_selected_index}], "
        f"x={i001.fixed_selected_x:.2f}, f={i001.fixed_objective:.6f}"
    )
    print(f"  recovered objective: {i001.objective_loss:.6f}")
    print("I-004 variance units")
    print(
        f"  variance at x=0.80: legacy={i004.legacy_physical_variance_at_global_peak:.6f}, "
        f"fixed={i004.fixed_physical_variance_at_global_peak:.6f}"
    )
    print(
        f"  legacy: x={i004.legacy_selected_x:.2f}, f={i004.legacy_objective:.6f}"
    )
    print(f"  fixed:  x={i004.fixed_selected_x:.2f}, f={i004.fixed_objective:.6f}")
    print(f"  recovered objective: {i004.objective_gain:.6f}")
    print(f"JSON: {json_path}")
    if not args.no_plot:
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
