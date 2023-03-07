from .LSTM import LSTM
from .AGCRN import AGCRN
from .STGCN import STGCN
from .GraphWaveNet import GWNET
from .MTGNN import MTGNN
from .STMetaLSTM import STMetaLSTM
from .SMetaLSTM import SMetaLSTM
from .TMetaLSTM import TMetaLSTM

# __all__ = ["LSTM", "AGCRN", "STGCN"]


def model_select(name):
    name = name.upper()

    if name == "LSTM":
        return LSTM
    elif name == "AGCRN":
        return AGCRN
    elif name == "STGCN":
        raise NotImplementedError
        return STGCN
    elif name in ("GWNET", "GRAPHWAVENET", "GWN"):
        return GWNET
    elif name == "MTGNN":
        return MTGNN
    elif name == "STMETALSTM":
        return STMetaLSTM
    elif name == "SMETALSTM":
        return SMetaLSTM
    elif name == "TMETALSTM":
        return TMetaLSTM

    else:
        raise NotImplementedError
