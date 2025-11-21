import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
#model class to declare RNN and defining a forward pass of the model

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, seq_length, output_size):
        #inherits from nn.Module
        super(RNN, self).__init__()
        self.hidden_size = hidden_size  #dim of memory inside lstm
        self.num_layers = num_layers  #stacked lstm layers
        #lstm: long short term memory - looks at lng term dependencies in sequential data

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  #correspond to input data shape
        self.seq_length = seq_length  #no of timestamps to look at to predict the next control output

        #num classes is the no of outputs predicted by the model

        #to convert memory vector to outputs (shaping constraints)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        #inital hidden, cell states - these are internal memory vectors
        #hidden = short term memory, current output of LSTM at a given time
        hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #cell state = long term memory, stores trends (remmebers info over many time steps)

        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #forward propagate lstm
        out, _ = self.lstm(x)
        #out, _ = self.lstm(x, (hidden_state,
        #                        cell_states))  #out; tensor of shape(batch_soze, seq_length, hidden_size) - at the final time step
        #decode the hidden state of t
        out = self.fc(out[:, -1, :])
        return out


