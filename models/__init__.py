from .LSTM import LSTM
from .AGCRN import AGCRN
from .GraphWaveNet import GWNET
from .MTGNN import MTGNN
from .STMetaLSTM import STMetaLSTM
from .SMetaLSTM import SMetaLSTM
from .TMetaLSTM import TMetaLSTM
from .GCLSTM import GCLSTM


def model_select(name):
    name = name.upper()

    if name == "LSTM":
        return LSTM
    elif name == "AGCRN":
        return AGCRN
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
    elif name == "GCLSTM":
        return GCLSTM

    else:
        raise NotImplementedError
