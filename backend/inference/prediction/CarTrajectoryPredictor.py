import numpy as np
import torch
import joblib
from inference.models import RNN


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
        state_cols,
        control_cols,
        model_path="new_multicar_multistep_model.pt",
        scaler_path="new_multicar_multistep_model.pkl",
        seq_len=10,
        scale=100.0,
        hidden_size=128,
        num_layers=2,
        dropout=0.0,
        device=None,
    ):
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

    def predict(self, df, horizon=None, start_idx=0):
        """
        Free-running rollout starting at `start_idx` for `horizon` steps.

        Returns:
            true_lat, true_lon, pred_lat, pred_lon  (each shape [T])
        """
        df = df.reset_index(drop=True)
        df_std = df.copy()

        cols_to_scale = self.state_cols + self.control_cols
        df_std[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        # tensors for states/controls (standardized)
        states = torch.tensor(
            df_std[self.state_cols].values, dtype=torch.float32
        )
        controls = torch.tensor(
            df_std[self.control_cols].values, dtype=torch.float32
        )

        seq_len   = self.seq_len
        state_dim = len(self.state_cols)

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