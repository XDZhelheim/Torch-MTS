from .BasicRunner import BasicRunner
from .MegaCRNRunner import MegaCRNRunner
from .GCRNRunner import GCRNRunner
from .GMANRunner import GMANRunner
from .DCRNNRunner import DCRNNRunner


def runner_select(name):
    name = name.upper()

    if name == "BASIC":
        return BasicRunner
    elif name in ("MEGACRNRUNNER", "MEGACRN"):
        return MegaCRNRunner
    elif name in ("GCRNRUNNER", "GCRN"):
        return GCRNRunner
    elif name in ("GMANRUNNER", "GMAN"):
        return GMANRunner
    elif name in ("DCRNNRUNNER", "DCRNN"):
        return DCRNNRunner

    else:
        raise NotImplementedError
