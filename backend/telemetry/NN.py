import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from telemetry.raw.TelemetryDB import TelemetryDB



#custom dataset from telemetry data
db = TelemetryDB("postgresql+psycopg2://racer:changeme@100.120.36.75:5432/racing")


# Speed
# nmot
#figure out how weather data affects?
#clean up data, do th egraphs to check for hte laps of all the cars, then try to remove those that might not be necessary. we then also look at making databases for the neural network. think about how th einputs/outputs map and how we are predicting the next state ?
# ath
# accx_can
# accy_can
# VBOX_Long_Minutes
# VBOX_Lat_Min
# Laptrigger_lapdist_dls


from torch.utils.data import Dataset


class telemetryDataset(Dataset):
    def __init__(self, state, control):
        self.state = state
        self.control = control


    def __get__(self, index):


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]



