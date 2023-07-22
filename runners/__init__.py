from .BasicRunner import BasicRunner
from .MegaCRNRunner import MegaCRNRunner
from .GCRNRunner import GCRNRunner


def runner_select(name):
    name = name.upper()

    if name == "BASIC":
        return BasicRunner
    elif name in ("MEGACRNRUNNER", "MEGACRN"):
        return MegaCRNRunner
    elif name in ("GCRNRUNNER", "GCRN"):
        return GCRNRunner

    else:
        raise NotImplementedError
