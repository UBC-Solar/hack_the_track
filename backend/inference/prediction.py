from backend.inference.models import RNN, MODEL_PATH, CarSequenceDataset
from backend.inference.constants import STATE_COLS, CONTROL_COLS, EARTH_RADIUS, SEQ_LEN, SCALE

import numpy as np
import joblib
import torch
import pandas as pd


TRAIN_DT = "50ms"   # what you used during training
TICK_DT  = "100ms"  # tickdb rate


class PathPredictor:
    def __init__(self, model_name: str = "multicar_model.pt", scaler_name: str = "multicar_scaler.pkl"):
        input_size = len(STATE_COLS) + len(CONTROL_COLS)
        output_size = len(STATE_COLS)

        self._device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

        self._model = RNN(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            output_size=output_size,
            dropout=0.0,
        ).to(self._device)

        self._model.load_state_dict(torch.load(str(MODEL_PATH / model_name), map_location=self._device))
        self._model.eval()

        self._scaler = joblib.load(str(MODEL_PATH / scaler_name))

    @staticmethod
    def inject_xy(car_df):
        origin_deg = (car_df['latitude'].iloc[0], car_df['longitude'].iloc[0])
        origin_rad = np.deg2rad(origin_deg)

        df = car_df.copy()
        lat_rad = np.deg2rad(df['latitude'].values)
        lon_rad = np.deg2rad(df['longitude'].values)

        dlat = lat_rad - origin_rad[0]
        dlon = lon_rad - origin_rad[1]

        x = EARTH_RADIUS * dlon * np.cos(origin_rad[0])  # east
        y = EARTH_RADIUS * dlat  # north

        df['x'] = x
        df['y'] = y

        return df, origin_rad

    @staticmethod
    def resample_to_training_rate(df_tick: pd.DataFrame) -> pd.DataFrame:
        """
        Tickdb is at 10 Hz (100 ms). Model was trained at 20 Hz (50 ms).
        Upsample the tick data to 50 ms using time-based interpolation.

        Assumes df_tick has either:
          - a DatetimeIndex already, or
          - a 'ts' column that can be converted to datetime.
        """
        df = df_tick.copy()

        # --- 1. Ensure DatetimeIndex ---
        if not isinstance(df.index, pd.DatetimeIndex):
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"])
                df = df.set_index("ts")
            else:
                raise ValueError("Need a DatetimeIndex or a 'ts' column to resample.")

        df = df.sort_index()

        # --- 2. Separate numeric and non-numeric columns ---
        numeric_cols = df.select_dtypes(include="number").columns
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

        df_num = df[numeric_cols]

        # --- 3. Resample numeric columns to 50 ms and interpolate in time ---
        df_num_resampled = (
            df_num
            .resample(TRAIN_DT)
            .interpolate(method="time")
        )

        # --- 4. Bring back non-numeric columns with forward-fill ---
        df_resampled = df_num_resampled.copy()
        for c in non_numeric_cols:
            df_resampled[c] = (
                df[c]
                .resample(TRAIN_DT)
                .ffill()
            )

        return df_resampled

    @staticmethod
    def _rename_columns(car_df):
        # map raw names → short names that you want in the model
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

        df_renamed = car_df.rename(columns=rename_map)

        return df_renamed

    @staticmethod
    def build_inference_dataset(
            df_xy,
            scaler,
            state_cols,
            control_cols,
            seq_len,
    ):
        """
        Build a CarSequenceDataset for an inference-only car,
        using an already-fitted scaler from the training car.
        """
        cols_to_scale = state_cols + control_cols

        # df_raw = df_xy.reset_index(drop=True).copy()
        df_raw = df_xy.copy()
        df_scaled = df_raw.copy()
        df_scaled[cols_to_scale] = scaler.transform(df_raw[cols_to_scale])

        infer_dataset = CarSequenceDataset(df_scaled, state_cols, control_cols, seq_len)
        return infer_dataset

    @staticmethod
    def free_running_rollout(
            model,
            dataset,
            scaler,
            state_cols,
            control_cols,
            scale,
            start_idx=0,
            horizon=500,
            device="cpu",
    ):
        """
        Closed-loop rollout on a single dataset, starting from start_idx.

        - Uses dataset.states / dataset.controls (already standardised).
        - Uses the same scaler that was fit on the training car.
        - Returns (true_seq_plot, pred_seq_plot) in ORIGINAL UNITS.
        """
        model.eval()

        state_dim = len(state_cols)
        cols_to_scale = state_cols + control_cols

        states = dataset.states  # [N, state_dim]
        controls = dataset.controls  # [N, ctrl_dim]
        N = states.size(0)
        seq_len = dataset.seq_len

        # We need seq_len history + 'horizon' predicted steps
        # Make sure we don't run off the end of the data starting from start_idx.
        max_horizon = N - (start_idx + seq_len)
        horizon = min(horizon, max_horizon)
        if horizon <= 0:
            raise ValueError(
                f"start_idx={start_idx} too close to end of dataset (N={N}, seq_len={seq_len})."
            )

        # Initial window: TRUE data from [start_idx, start_idx + seq_len)
        current_states = states[start_idx: start_idx + seq_len].clone().cpu()
        current_controls = controls[start_idx: start_idx + seq_len].clone().cpu()

        pred_states_list = []

        with torch.no_grad():
            for step in range(horizon):
                # Build input window [seq_len, state_dim + ctrl_dim]
                x_seq = torch.cat([current_states, current_controls], dim=1)
                x_input = x_seq.unsqueeze(0).to(device)  # [1, seq_len, input_dim]

                last_state = x_input[:, -1, :state_dim]

                scaled_delta = model(x_input)
                delta = scaled_delta / scale
                y_hat = last_state + delta  # [1, state_dim]

                y_hat_cpu = y_hat.squeeze(0).cpu()  # [state_dim]
                pred_states_list.append(y_hat_cpu)

                # Roll window: drop oldest, append predicted state
                current_states = torch.cat(
                    [current_states[1:], y_hat_cpu.unsqueeze(0)],
                    dim=0,
                )

                # Controls for the *correct* future index:
                # at step 0 we want index start_idx + seq_len
                next_ctrl_idx = start_idx + seq_len + step
                next_ctrl = controls[next_ctrl_idx].cpu().unsqueeze(0)
                current_controls = torch.cat(
                    [current_controls[1:], next_ctrl],
                    dim=0,
                )

        # --- convert to numpy (still standardized) ---
        pred_seq_std = torch.stack(pred_states_list, dim=0).numpy()  # [horizon, state_dim]
        # True states should be aligned with prediction window:
        # states from [start_idx + seq_len, start_idx + seq_len + horizon)
        true_seq_std = states[
                       start_idx + seq_len: start_idx + seq_len + horizon
                       ].cpu().numpy()  # [horizon, state_dim]

        # --- inverse transform back to original units ---
        num_state = len(state_cols)
        num_all = len(cols_to_scale)

        true_full = np.zeros((horizon, num_all))
        pred_full = np.zeros((horizon, num_all))

        true_full[:, :num_state] = true_seq_std
        pred_full[:, :num_state] = pred_seq_std

        true_full_unscaled = scaler.inverse_transform(true_full)
        pred_full_unscaled = scaler.inverse_transform(pred_full)

        true_seq_plot = true_full_unscaled[:, :num_state]
        pred_seq_plot = pred_full_unscaled[:, :num_state]

        return true_seq_plot, pred_seq_plot

    @staticmethod
    def xy_to_latlon(x, y, origin_rad):
        lat_rad = origin_rad[0] + y / EARTH_RADIUS
        lon_rad = origin_rad[1] + x / (EARTH_RADIUS * np.cos(origin_rad[0]))
        return np.rad2deg(lat_rad), np.rad2deg(lon_rad)

    @staticmethod
    def get_lat_lon(predictions, origin):
        x_idx = STATE_COLS.index("x")
        y_idx = STATE_COLS.index("y")

        x = predictions[:, x_idx]
        y = predictions[:, y_idx]

        return PathPredictor.xy_to_latlon(x, y, origin)

    def predict(self, car_df, num_indices=None):
        # car_df_20hz = PathPredictor.resample_to_training_rate(car_df)
        car_df_renamed = self._rename_columns(car_df)

        car_df_xy, origin_rad = PathPredictor.inject_xy(car_df_renamed)

        infer_dataset = self.build_inference_dataset(car_df_xy, self._scaler, STATE_COLS, CONTROL_COLS, SEQ_LEN)

        if num_indices is None:
            num_indices = len(infer_dataset) - 1

        true_path, predicted_path = PathPredictor.free_running_rollout(
            self._model, infer_dataset, self._scaler,
            STATE_COLS, CONTROL_COLS,
            scale=SCALE, start_idx=0, horizon=num_indices, device=self._device
        )

        print("STATE_COLS =", STATE_COLS)
        print("predicted_path.shape =", predicted_path.shape)
        print("first few unscaled states:")
        print(predicted_path[:5])

        predicted_latitude, predicted_longitude = PathPredictor.get_lat_lon(predicted_path, origin_rad)
        true_latitude, true_longitude = PathPredictor.get_lat_lon(true_path, origin_rad)

        return (predicted_latitude, predicted_longitude), (true_latitude, true_longitude)
