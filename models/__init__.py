from .baseline.HistoricalInertia import HistoricalInertia
from .baseline.LSTM import LSTM
from .baseline.GRU import GRU
from .baseline.Attention import Attention
from .baseline.WaveNet import WaveNet
from .baseline.GCLSTM import GCLSTM
from .baseline.GCGRU import GCGRU

from .meta_param.STMetaLSTM import STMetaLSTM
from .meta_param.STMetaGRU import STMetaGRU
from .meta_param.STMetaAttention import STMetaAttention
from .meta_param.STMetaGCGRU import STMetaGCGRU

from .AGCRN import AGCRN
from .GraphWaveNet import GWNET
from .MTGNN import MTGNN
from .STWA import STWA
from .STID import STID


def model_select(name):
    name = name.upper()

    if name in ("HI", "HISTORICALINERTIA", "COPYLASTSTEPS"):
        return HistoricalInertia
    elif name == "LSTM":
        return LSTM
    elif name == "GRU":
        return GRU
    elif name in ("ATTENTION", "ATTN", "TRANSFORMER"):
        return Attention
    elif name in ("TCN", "WAVENET"):
        return WaveNet

    elif name == "GCLSTM":
        return GCLSTM
    elif name == "GCGRU":
        return GCGRU

    elif name == "STMETALSTM":
        return STMetaLSTM
    elif name == "STMETAGRU":
        return STMetaGRU
    elif name in ("STMETAATTN", "STMETAATTENTION", "STMETATRANSFORMER"):
        return STMetaAttention
    elif name == "STMETAGCGRU":
        return STMetaGCGRU

    elif name == "AGCRN":
        return AGCRN
    elif name in ("GWNET", "GRAPHWAVENET", "GWN"):
        return GWNET
    elif name == "MTGNN":
        return MTGNN
    elif name == "STWA":
        return STWA
    elif name == "STID":
        return STID

    else:
        raise NotImplementedError
