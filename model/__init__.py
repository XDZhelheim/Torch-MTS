from .LSTM import LSTM
from .AGCRN import AGCRN
from .STGCN import STGCN

# __all__ = ["LSTM", "AGCRN", "STGCN"]


def model_select(name):
    name = name.upper()

    if name == "LSTM":
        return LSTM
    elif name == "AGCRN":
        return AGCRN
    elif name == "STGCN":
        return STGCN

    else:
        raise NotImplementedError
