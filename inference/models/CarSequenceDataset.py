import torch
from torch.utils.data import Dataset


class CarSequenceDataset(Dataset):
    def __init__(self, df, state_cols, control_cols, seq_len):
        self.seq_len = seq_len
        self.state_cols = state_cols
        self.control_cols = control_cols

        self.states = torch.tensor(df[state_cols].values, dtype=torch.float32)
        self.controls = torch.tensor(df[control_cols].values, dtype=torch.float32)

    def __len__(self):
        return self.states.size(0) - self.seq_len

    def __getitem__(self, idx):
        x_seq_states = self.states[idx : idx + self.seq_len]
        x_seq_ctrls  = self.controls[idx : idx + self.seq_len]
        x_seq = torch.cat([x_seq_states, x_seq_ctrls], dim=1)

        y_next = self.states[idx + self.seq_len]
        return x_seq, y_next, idx  # include idx for multi-step training
