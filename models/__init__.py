from .LSTM import LSTM
from .GRU import GRU
from .Attention import Attention
from .AGCRN import AGCRN
from .GraphWaveNet import GWNET
from .MTGNN import MTGNN
from .STMetaLSTM import STMetaLSTM
from .SMetaLSTM import SMetaLSTM
from .TMetaLSTM import TMetaLSTM
from .GCLSTM import GCLSTM
from .STWA import STWA


def model_select(name):
    name = name.upper()

    if name == "LSTM":
        return LSTM
    elif name == "GRU":
        return GRU
    elif name in ("ATTENRTION", "ATTN", "TRANSFORMER"):
        return Attention
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
    elif name == "STWA":
        return STWA

    else:
        raise NotImplementedError
