from .baseline.HistoricalInertia import HistoricalInertia
from .baseline.MLP import MLP
from .baseline.LSTM import LSTM
from .baseline.GRU import GRU
from .baseline.Transformer import Transformer
from .baseline.WaveNet import WaveNet
from .baseline.Mamba import Mamba
from .baseline.GCLSTM import GCLSTM
from .baseline.GCGRU import GCGRU
from .baseline.GCRN import GCRN

from .AGCRN import AGCRN
from .GraphWaveNet import GWNET
from .MTGNN import MTGNN
from .STWA import STWA
from .STID import STID
from .STNorm import STNorm
from .StemGNN import StemGNN
from .MegaCRN import MegaCRN
from .GMAN import GMAN
from .STGCN import STGCN
from .DCRNN import DCRNN
from .STAEformer import STAEformer
from .GTS import GTS

from .DLinear import DLinear
from .PatchTST import PatchTST


def model_select(name):
    name = name.upper()

    if name in ("HI", "HISTORICALINERTIA", "COPYLASTSTEPS"):
        return HistoricalInertia
    elif name == "MLP":
        return MLP
    elif name == "LSTM":
        return LSTM
    elif name == "GRU":
        return GRU
    elif name in ("ATTENTION", "ATTN", "TRANSFORMER", "TF"):
        return Transformer
    elif name in ("TCN", "WAVENET"):
        return WaveNet
    elif name == "MAMBA":
        return Mamba

    elif name == "GCRN":
        return GCRN
    elif name == "GCLSTM":
        return GCLSTM
    elif name == "GCGRU":
        return GCGRU

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
    elif name == "STNORM":
        return STNorm
    elif name == "STEMGNN":
        return StemGNN
    elif name == "MEGACRN":
        return MegaCRN
    elif name == "GMAN":
        return GMAN
    elif name == "STGCN":
        return STGCN
    elif name == "DCRNN":
        return DCRNN
    elif name in ("STAE", "STAEFORMER"):
        return STAEformer
    elif name == "GTS":
        return GTS
    
    elif name == "DLINEAR":
        return DLinear
    elif name == "PATCHTST":
        return PatchTST

    else:
        raise NotImplementedError
