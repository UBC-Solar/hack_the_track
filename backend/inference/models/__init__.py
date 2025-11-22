from pathlib import Path
from .RNN import RNN
from .CarSequenceDataset import CarSequenceDataset

MODEL_PATH = Path(__file__).parent.absolute()

__all__ = [
    "RNN",
    "MODEL_PATH",
    "CarSequenceDataset"
]
