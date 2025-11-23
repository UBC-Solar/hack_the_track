from matplotlib import pyplot as plt


# -------------------------------------------------------------------
# Global constants / config
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# High-level: create standardized seq datasets + loaders from a dataframe
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------



def plot_true_vs_predicted_latlon(
    predictor,
    horizon=500,
    start_idx=2000,
    title="Predicted trajectory vs true (lat/lon)",
):
    true_lat, true_lon, pred_lat, pred_lon = predictor.predict_free_run(
        horizon=horizon,
        start_idx=start_idx,
    )

    plt.figure(figsize=(8, 8))

    # True path
    plt.plot(true_lon, true_lat, label="True", linewidth=2)
    # Predicted path
    plt.plot(pred_lon, pred_lat, label="Predicted", linestyle="--", linewidth=2)

    # Mark start & end
    plt.scatter(true_lon[0], true_lat[0], c="green", marker="o", s=60, label="Start (true)")
    plt.scatter(true_lon[-1], true_lat[-1], c="red", marker="x", s=60, label="End (true)")

    plt.scatter(pred_lon[0], pred_lat[0], c="green", marker="o", s=30, alpha=0.6)
    plt.scatter(pred_lon[-1], pred_lat[-1], c="red", marker="x", s=30, alpha=0.6)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.legend()
    plt.gca().set_aspect("equal", "box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
