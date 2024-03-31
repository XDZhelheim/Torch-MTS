from .STFRunner import STFRunner
from .MegaCRNRunner import MegaCRNRunner
from .GCRNRunner import GCRNRunner
from .GMANRunner import GMANRunner
from .DCRNNRunner import DCRNNRunner
from .GTSRunner import GTSRunner

from .LTSFRunner import LTSFRunner


def runner_select(name):
    name = name.upper()

    if name in ("STF", "BASIC", "DEFAULT"):
        return STFRunner
    elif name in ("MEGACRNRUNNER", "MEGACRN"):
        return MegaCRNRunner
    elif name in ("GCRNRUNNER", "GCRN"):
        return GCRNRunner
    elif name in ("GMANRUNNER", "GMAN"):
        return GMANRunner
    elif name in ("DCRNNRUNNER", "DCRNN"):
        return DCRNNRunner
    elif name in ("GTSRUNNER", "GTS"):
        return GTSRunner
    
    elif name in ("LTSF", "LONG", "LONGTERM"):
        return LTSFRunner

    else:
        raise NotImplementedError
