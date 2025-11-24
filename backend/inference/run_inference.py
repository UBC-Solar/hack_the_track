from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from inference.insights import get_insights, ControlModification, make_predictor


def plot_trajectories_and_gates(
    lat_true: pd.Series,
    lon_true: pd.Series,
    pred_results: Dict[str, Tuple[pd.Series, pd.Series]],
    gates_with_intersections: Dict[Tuple[Tuple[float, float], Tuple[float, float]], list],
):
    plt.figure(figsize=(8, 8))

    # True trajectory
    plt.plot(lon_true, lat_true, label="True", linewidth=2, color="black")
    plt.scatter(
        lon_true[0],
        lat_true[0],
        c="green",
        marker="o",
        s=60,
        label="Start (true)",
    )
    plt.scatter(
        lon_true[-1],
        lat_true[-1],
        c="red",
        marker="x",
        s=60,
        label="End (true)",
    )

    # Predicted variants
    for name, (lat_pred, lon_pred) in pred_results.items():
        plt.plot(lon_pred, lat_pred, label=name, linestyle="--", linewidth=1.5)

    # Gates that got intersections
    gate_label_added = False
    for gate_key_, _hit_variants in gates_with_intersections.items():
        (x1, y1), (x2, y2) = gate_key_
        label = "Gate (intersected)" if not gate_label_added else None
        plt.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.5,
            alpha=0.7,
            color="red",
            label=label,
        )
        gate_label_added = True

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Predicted vs True Trajectory (Control Variants)")
    plt.legend()
    plt.gca().set_aspect("equal", "box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main(
    vehicle_id: str = "GR86-040-3",
    duration_s: float = 5.0,
    modifications: List[ControlModification] | None = None,
):
    (
        lat_true,
        lon_true,
        pred_results,
        gates_with_intersections,
        best_controls,
        best_improvement,
    ) = get_insights(
        vehicle_id,
        duration_s,
        modifications,
        predictor=make_predictor()
    )

    # Print best control(s) and time improvement
    if best_improvement is not None and best_controls:
        if len(best_controls) == 1:
            print(
                f"\nBest control modification: {best_controls[0]} "
                f"(Δt = {best_improvement:.3f} s vs baseline)"
            )
        else:
            combo_str = " + ".join(best_controls)
            print(
                f"\nBest combined controls: {combo_str} "
                f"(Δt = {best_improvement:.3f} s vs baseline)"
            )
    else:
        print("\nNo beneficial control modifications found.")

    # Plotting
    plot_trajectories_and_gates(
        lat_true,
        lon_true,
        pred_results,
        gates_with_intersections,
    )


if __name__ == "__main__":
    main()
