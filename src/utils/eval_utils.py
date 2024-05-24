import math
import numpy as np
import scipy.stats as stats
import torch

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

def calculate_accuracy_on_batch(lm_logits, input_ids, k=5):
    ks = [5, 10, 20, 50, 100]
    num_correct = torch.zeros(5)
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    non_pad_mask = shift_labels != 0
    
    for i, k in enumerate(ks):
        top_k_predictions = torch.topk(shift_logits, k, dim=-1).indices
        true_in_top_k = (top_k_predictions == shift_labels.unsqueeze(-1)).any(dim=-1)
        num_correct[i] = (true_in_top_k & non_pad_mask).sum().item()
    
    num_samples = non_pad_mask.sum().item()
    return num_correct, num_samples
