from .LSTM import LSTM
from .AGCRN import AGCRN
from .STGCN import STGCN
from .GraphWaveNet import GWNET
from .MTGNN import MTGNN

# __all__ = ["LSTM", "AGCRN", "STGCN"]


def model_select(name):
    name = name.upper()

    if name == "LSTM":
        return LSTM
    elif name == "AGCRN":
        return AGCRN
    elif name == "STGCN":
        return STGCN
    elif name in ("GWNET", "GRAPHWAVENET", "GWN"):
        return GWNET
    elif name == "MTGNN":
        return MTGNN

    else:
        raise NotImplementedError
