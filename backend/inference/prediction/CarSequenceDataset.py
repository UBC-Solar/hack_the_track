import torch


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
