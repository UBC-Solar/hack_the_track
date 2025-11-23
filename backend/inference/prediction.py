from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


# -------------------------------------------------------------------
# Global constants / config
# -------------------------------------------------------------------
R_EARTH = 6371000.0  # meters
telemetry_names = [
    "accx", "accy", "speed", "gear", "aps",
    "nmot", "pbrake_f", "pbrake_r", "latitude", "longitude",
]

DB_URL = "postgresql+psycopg2://racer:changeme@100.120.36.75:5432/racing"

# -------------------------------------------------------------------
# Low-level helpers: telemetry -> common index -> combined df -> x/y
# -------------------------------------------------------------------
def index(list_dfs):
    """
    Standardize timestamp column, rename 'telemetry_value' -> 'value',
    and compute a global common index at 50ms.
    """
    for i, df in enumerate(list_dfs):
        list_dfs[i] = df.copy()
        list_dfs[i]["timestamp"] = pd.to_datetime(
            list_dfs[i]["timestamp"], unit="ns"
        )
        if "telemetry_value" in list_dfs[i].columns:
            list_dfs[i].rename(columns={"telemetry_value": "value"}, inplace=True)

    start_time = min(df["timestamp"].min() for df in list_dfs)
    end_time   = max(df["timestamp"].max() for df in list_dfs)
    common_index = pd.date_range(start=start_time, end=end_time, freq="50ms")
    return common_index, list_dfs


def resample(df, common_index):
    """
    Resample one telemetry DataFrame onto the common index and interpolate in time.
    """
    df_resampled = df.copy()
    df = df[~df["timestamp"].duplicated()]
    df_new = df.set_index("timestamp", inplace=False)
    df_resampled["value"] = pd.to_numeric(df_resampled["value"], errors="coerce")

    df_resampled = df_new.reindex(common_index).interpolate(method="time")
    df_resampled["value"] = df_resampled["value"].ffill().bfill()
    df_resampled.drop(columns=["name"], inplace=True, errors="ignore")

    return df_resampled


def combine_dfs_car(telemetry_names, common_index, all_dfs):
    """
    Resample each telemetry DataFrame and combine them into a single wide DataFrame.
    """
    combined_df = pd.DataFrame(index=common_index)

    for name, df in zip(telemetry_names, all_dfs):
        df_interp = resample(df, common_index)
        combined_df[name] = pd.to_numeric(df_interp["value"], errors="coerce").values

    return combined_df


# -------------------------------------------------------------------
# High-level: build the per-car dataframe
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# Dataset + model definitions
# -------------------------------------------------------------------
class CarSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, df, state_cols, control_cols, seq_len):
        self.seq_len = seq_len
        self.states = torch.tensor(df[state_cols].values, dtype=torch.float32)
        self.controls = torch.tensor(df[control_cols].values, dtype=torch.float32)

    def __len__(self):
        # last usable window starts at len - seq_len - 1 (we need seq_len steps + one target)
        return self.states.size(0) - self.seq_len

    def __getitem__(self, idx):
        x_seq_states = self.states[idx : idx + self.seq_len]
        x_seq_ctrls  = self.controls[idx : idx + self.seq_len]
        x_seq = torch.cat([x_seq_states, x_seq_ctrls], dim=1)  # [seq_len, input_size]

        y_next = self.states[idx + self.seq_len]               # [output_size]
        return x_seq, y_next, idx


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        batch_size = x.size(0)
        device = x.device

        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

        out, _ = self.lstm(x, (h0, c0))   # [batch, seq_len, hidden_size]
        last_hidden = out[:, -1, :]       # [batch, hidden_size]
        out = self.fc(last_hidden)        # [batch, output_size]
        return out

# -------------------------------------------------------------------
# High-level: create standardized seq datasets + loaders from a dataframe
# -------------------------------------------------------------------
def make_sequence_datasets(
    df_xy,
    state_cols,
    control_cols,
    seq_len,
    train_frac=0.8,
    batch_size=64,
):
    """
    Given a dataframe with states+controls (+whatever else),
    build standardized CarSequenceDataset train/test + DataLoaders + scaler.
    """
    cols_to_scale = state_cols + control_cols

    n_total = len(df_xy)
    train_len = int(train_frac * n_total)

    df_train_raw = df_xy.iloc[:train_len].reset_index(drop=True)
    df_test_raw  = df_xy.iloc[train_len:].reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(df_train_raw[cols_to_scale])

    df_train = df_train_raw.copy()
    df_test  = df_test_raw.copy()
    df_train[cols_to_scale] = scaler.transform(df_train_raw[cols_to_scale])
    df_test[cols_to_scale]  = scaler.transform(df_test_raw[cols_to_scale])

    train_dataset = CarSequenceDataset(df_train, state_cols, control_cols, seq_len)
    test_dataset  = CarSequenceDataset(df_test,  state_cols, control_cols, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader, scaler

# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------
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
    origin_rad,
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


def prepare_tickdb_dataframe_for_model(df_tick, state_cols, control_cols):
    """
    Convert a dataframe pulled from tickdb into the unified dataframe format
    used to train the RNN model.

    - Renames tickdb signal names → model names
    - Adds x/y from latitude/longitude
    - Returns (df_ready, origin_rad)

    Assumes:
      - df_tick has a proper datetime index
      - already at correct sampling rate
      - no resampling required
    """

    # --- rename tickdb fields → model fields ---
    rename_map = {
        "accx_can": "accx",
        "accy_can": "accy",
        "speed": "speed",
        "gear": "gear",
        "aps": "aps",
        "nmot": "nmot",
        "pbrake_f": "pbrake_f",
        "pbrake_r": "pbrake_r",
        "VBOX_Lat_Min": "latitude",
        "VBOX_Long_Minutes": "longitude",
    }

    df = df_tick.copy()
    df.rename(columns=rename_map, inplace=True)

    df = df_tick.copy()
    df.rename(columns=rename_map, inplace=True)

    # basic forward-fill/backward-fill on the raw columns we care about
    fill_cols = list(rename_map.values())
    df[fill_cols] = df[fill_cols].ffill().bfill()

    # --- required *raw* fields (exclude x,y because we create them) ---
    base_state_cols = [c for c in state_cols if c not in ("x", "y")]
    required = set(base_state_cols + control_cols + ["latitude", "longitude"])

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Tick DB dataframe missing fields: {missing}")

    return df


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


class CarTrajectoryPredictor:
    """
    Given a car dataframe (with state+control columns, x/y, latitude/longitude),
    this class:
      - loads a saved scaler and model
      - standardizes the data
      - can run a free-running rollout
      - returns true & predicted lat/lon
    """

    def __init__(
        self,
        df_xy,
        state_cols,
        control_cols,
        model_path="car13_multistep_model.pt",
        scaler_path="car13_scaler.pkl",
        seq_len=10,
        scale=50.0,
        hidden_size=256,
        num_layers=2,
        dropout=0.0,
        device=None,
    ):
        self.df_xy = df_xy.reset_index(drop=True)
        self.state_cols = list(state_cols)
        self.control_cols = list(control_cols)
        self.seq_len = seq_len
        self.scale = scale

        # device
        if device is None:
            device = (
                torch.accelerator.current_accelerator().type
                if hasattr(torch, "accelerator") and torch.accelerator.is_available()
                else "cpu"
            )
        self.device = torch.device(device)

        # ---- load scaler ----
        self.scaler = joblib.load(scaler_path)

        # ---- standardize full dataframe ----
        cols_to_scale = self.state_cols + self.control_cols
        df_std = self.df_xy.copy()
        df_std[cols_to_scale] = self.scaler.transform(self.df_xy[cols_to_scale])

        # tensors for states/controls (standardized)
        self.states = torch.tensor(
            df_std[self.state_cols].values, dtype=torch.float32
        )
        self.controls = torch.tensor(
            df_std[self.control_cols].values, dtype=torch.float32
        )

        # ---- load model ----
        input_size = len(self.state_cols) + len(self.control_cols)
        output_size = len(self.state_cols)

        self.model = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, horizon=None, start_idx=0):
        """
        Free-running rollout starting at `start_idx` for `horizon` steps.

        Returns:
            true_lat, true_lon, pred_lat, pred_lon  (each shape [T])
        """
        seq_len   = self.seq_len
        state_dim = len(self.state_cols)

        states   = self.states
        controls = self.controls
        N = states.size(0)

        # clip horizon so we don't run past the data
        max_horizon = N - (start_idx + seq_len)
        if horizon is None:
            horizon = max_horizon

        horizon = max(0, min(horizon, max_horizon))
        if horizon <= 0:
            raise ValueError("Horizon/start_idx combination runs past the dataset.")

        # initial window: teacher-forced true data
        current_states   = states[start_idx : start_idx + seq_len].clone().cpu()
        current_controls = controls[start_idx : start_idx + seq_len].clone().cpu()

        pred_states_list = []

        with torch.no_grad():
            for step in range(horizon):
                # build input window: [seq_len, state_dim + ctrl_dim]
                x_seq = torch.cat([current_states, current_controls], dim=1)
                x_input = x_seq.unsqueeze(0).to(self.device)

                last_state = x_input[:, -1, :state_dim]
                scaled_delta = self.model(x_input)
                delta = scaled_delta / self.scale
                y_hat = last_state + delta
                y_hat_cpu = y_hat.squeeze(0).cpu()  # [state_dim]

                pred_states_list.append(y_hat_cpu)

                # roll state window: drop oldest, append predicted
                current_states = torch.cat(
                    [current_states[1:], y_hat_cpu.unsqueeze(0)],
                    dim=0,
                )

                # controls: take the next true control from the dataset
                next_ctrl_idx = start_idx + seq_len + step
                next_ctrl = controls[next_ctrl_idx].cpu().unsqueeze(0)
                current_controls = torch.cat(
                    [current_controls[1:], next_ctrl],
                    dim=0,
                )

        pred_seq_std = torch.stack(pred_states_list, dim=0).numpy()  # [T, state_dim]
        true_seq_std = states[start_idx + seq_len : start_idx + seq_len + horizon].cpu().numpy()

        # ---- inverse transform back to physical units ----
        cols_to_scale = self.state_cols + self.control_cols
        num_state = len(self.state_cols)
        num_all   = len(cols_to_scale)

        true_full = np.zeros((horizon, num_all))
        pred_full = np.zeros((horizon, num_all))

        true_full[:, :num_state] = true_seq_std
        pred_full[:, :num_state] = pred_seq_std

        true_full_unscaled = self.scaler.inverse_transform(true_full)
        pred_full_unscaled = self.scaler.inverse_transform(pred_full)

        true_seq_phys = true_full_unscaled[:, :num_state]
        pred_seq_phys = pred_full_unscaled[:, :num_state]

        # indices of x, y in the state vector
        x_idx = self.state_cols.index("longitude")
        y_idx = self.state_cols.index("latitude")

        true_lon = true_seq_phys[:, x_idx]
        true_lat = true_seq_phys[:, y_idx]

        pred_lon = pred_seq_phys[:, x_idx]
        pred_lat = pred_seq_phys[:, y_idx]

        return true_lat, true_lon, pred_lat, pred_lon