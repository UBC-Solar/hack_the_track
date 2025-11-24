from matplotlib import pyplot as plt
import numpy as np
import torch


def plot_rolling_one_step(
    model,
    test_dataset,
    scaler,
    state_cols,
    control_cols,
    scale,
    horizon=100,
):
    """
    Rolling one-step evaluation (delta-state model) and plots per-state timeseries
    in original units.
    """
    model.eval()
    device = next(model.parameters()).device

    state_dim = len(state_cols)
    cols_to_scale = state_cols + control_cols

    horizon = min(horizon, len(test_dataset))

    all_true = []
    all_pred = []

    with torch.no_grad():
        for idx in range(horizon):
            x_seq, y_true, _ = test_dataset[idx]

            x_seq_dev = x_seq.unsqueeze(0).to(device)
            last_state = x_seq_dev[:, -1, :state_dim]

            scaled_delta_pred = model(x_seq_dev)   # [1, state_dim]
            delta = scaled_delta_pred / scale
            y_hat = last_state + delta            # [1, state_dim]

            all_true.append(y_true)
            all_pred.append(y_hat.squeeze(0).cpu())

    true_seq = torch.stack(all_true).cpu().numpy()  # [horizon, state_dim]
    pred_seq = torch.stack(all_pred).cpu().numpy()  # [horizon, state_dim]

    num_state = len(state_cols)
    num_all   = len(cols_to_scale)

    true_full = np.zeros((horizon, num_all))
    pred_full = np.zeros((horizon, num_all))

    true_full[:, :num_state] = true_seq
    pred_full[:, :num_state] = pred_seq

    true_full_unscaled = scaler.inverse_transform(true_full)
    pred_full_unscaled = scaler.inverse_transform(pred_full)

    true_seq_plot = true_full_unscaled[:, :num_state]
    pred_seq_plot = pred_full_unscaled[:, :num_state]

    num_states = num_state
    plt.figure(figsize=(15, 10))
    for i, name in enumerate(state_cols):
        plt.subplot(num_states, 1, i + 1)
        plt.plot(true_seq_plot[:, i], label="True")
        plt.plot(pred_seq_plot[:, i], label="Predicted", linestyle="--")
        plt.title(f"{name} (rolling one-step, Δ-state model)")
        plt.xlabel("Step")
        plt.ylabel(name)
        if i == 0:
            plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_free_running_rollout(
    model,
    test_dataset,
    scaler,
    state_cols,
    control_cols,
    scale,
    horizon=500,
    start_idx=2000,
    title_suffix="Δ-state model",
):
    """
    Free-running (closed-loop) rollout from test_dataset starting at start_idx,
    plus trajectory plot in latitude/longitude.
    """
    model.eval()
    device = next(model.parameters()).device

    seq_len   = test_dataset.seq_len
    state_dim = len(state_cols)
    cols_to_scale = state_cols + control_cols

    states   = test_dataset.states    # [N, state_dim], standardized
    controls = test_dataset.controls  # [N, ctrl_dim], standardized
    N = states.size(0)

    # Don't run past the end of the data
    max_horizon = N - (start_idx + seq_len)
    horizon = max(0, min(horizon, max_horizon))
    if horizon <= 0:
        raise ValueError("Horizon/start_idx combination runs past the dataset.")

    # initial window: teacher-forced true data
    current_states   = states[start_idx : start_idx + seq_len].clone().cpu()
    current_controls = controls[start_idx : start_idx + seq_len].clone().cpu()

    pred_states_list = []

    with torch.no_grad():
        for step in range(horizon):
            x_seq   = torch.cat([current_states, current_controls], dim=1)  # [seq_len, in_features]
            x_input = x_seq.unsqueeze(0).to(device)

            last_state = x_input[:, -1, :state_dim]
            scaled_delta = model(x_input)
            delta = scaled_delta / scale
            y_hat = last_state + delta
            y_hat_cpu = y_hat.squeeze(0).cpu()

            pred_states_list.append(y_hat_cpu)

            # roll the window: drop oldest, append predicted
            current_states = torch.cat(
                [current_states[1:], y_hat_cpu.unsqueeze(0)],
                dim=0,
            )

            # controls are known truth: pick next from dataset
            next_ctrl_idx = start_idx + seq_len + step
            next_ctrl = controls[next_ctrl_idx].cpu().unsqueeze(0)
            current_controls = torch.cat(
                [current_controls[1:], next_ctrl],
                dim=0,
            )

    pred_seq_std = torch.stack(pred_states_list, dim=0).numpy()  # [horizon, state_dim]
    true_seq_std = states[start_idx + seq_len : start_idx + seq_len + horizon].cpu().numpy()

    num_state = len(state_cols)
    num_all   = len(cols_to_scale)

    true_full = np.zeros((horizon, num_all))
    pred_full = np.zeros((horizon, num_all))

    true_full[:, :num_state] = true_seq_std
    pred_full[:, :num_state] = pred_seq_std

    true_full_unscaled = scaler.inverse_transform(true_full)
    pred_full_unscaled = scaler.inverse_transform(pred_full)

    true_seq_plot = true_full_unscaled[:, :num_state]
    pred_seq_plot = pred_full_unscaled[:, :num_state]

    # --- Timeseries plots for each state ---
    plt.figure(figsize=(15, 10))
    for i, name in enumerate(state_cols):
        plt.subplot(len(state_cols), 1, i + 1)
        plt.plot(true_seq_plot[:, i], label="True")
        plt.plot(pred_seq_plot[:, i], label="Predicted", linestyle="--")
        plt.title(f"{name} (free-running rollout, {title_suffix})")
        plt.xlabel("Step")
        plt.ylabel(name)
        if i == 0:
            plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # --- Trajectory in lat/lon from x/y ---
    x_idx = state_cols.index("longitude")
    y_idx = state_cols.index("latitude")

    true_lat = true_seq_plot[:, y_idx]
    true_lon = true_seq_plot[:, x_idx]

    pred_lat = pred_seq_plot[:, y_idx]
    pred_lon = pred_seq_plot[:, x_idx]

    plt.figure(figsize=(8, 8))
    plt.plot(true_lon, true_lat, label="True", linewidth=2)
    plt.plot(pred_lon, pred_lat, label="Predicted", linestyle="--", linewidth=2)

    plt.scatter(true_lon[0], true_lat[0], c="green", marker="o", s=60, label="Start (true)")
    plt.scatter(true_lon[-1], true_lat[-1], c="red", marker="x", s=60, label="End (true)")

    plt.scatter(pred_lon[0], pred_lat[0], c="green", marker="o", s=30, alpha=0.6)
    plt.scatter(pred_lon[-1], pred_lat[-1], c="red", marker="x", s=30, alpha=0.6)

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Predicted trajectory over rollout ({title_suffix})")
    plt.legend()
    plt.gca().set_aspect("equal", "box")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
