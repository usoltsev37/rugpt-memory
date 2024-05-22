import math


def format_log(loss: float, split: str) -> str:
    log_str = "| {0} loss {1:5.2f} | {0} ppl {2:9.3f} ".format(split, loss, math.exp(loss))
    return log_str
