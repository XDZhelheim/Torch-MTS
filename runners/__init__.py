from .BasicRunner import BasicRunner
from .MegaCRNRunner import MegaCRNRunner


def runner_select(name):
    name = name.upper()

    if name == "BASIC":
        return BasicRunner
    elif name in ("MEGACRNRUNNER", "MEGACRN"):
        return MegaCRNRunner

    else:
        raise NotImplementedError
