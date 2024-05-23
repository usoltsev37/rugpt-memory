import math
import numpy as np
import scipy.stats as stats

def format_log(ci_loss: float, ci_ppl, split: str) -> str:
    log_str_loss = (f"{split} loss: Mean = {ci_loss['mean']:5.6f}, "
                    f"CI = ({ci_loss['mean']-ci_loss['border']:5.6f}, "
                    f"{ci_loss['mean']+ci_loss['border']:5.6f})")
    
    log_str_ppl = (f"{split} ppl: Mean = {ci_ppl['mean']:9.6f}, "
                   f"CI = ({ci_ppl['mean']-ci_ppl['border']:9.6f}, "
                   f"{ci_ppl['mean']+ci_ppl['border']:9.6f})")

    log_str = "\n" + log_str_loss + "\n" + log_str_ppl
    return log_str

def calculate_confidence_interval(array):
    mean = np.mean(array)
    std = np.std(array, ddof=1)
    sample_size = len(array)
    t_critical = stats.t.ppf(0.975, df=sample_size-1)
    border = t_critical * std / np.sqrt(sample_size)
    return {"mean": mean, "border": border}
