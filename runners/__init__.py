from .BasicRunner import BasicRunner


def runner_select(name):
    name = name.upper()

    if name == "BASIC":
        return BasicRunner

    else:
        raise NotImplementedError
