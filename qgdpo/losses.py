import torch
import torch.nn.functional as F
from .quantization import QuantizedReferenceCache, QuantizeSTE

def qgdpo_loss(policy_chosen, policy_rejected, ref_chosen_q, ref_rejected_q, beta=0.1):
    ref_chosen = QuantizedReferenceCache.decompress(*ref_chosen_q)
    ref_rejected = QuantizedReferenceCache.decompress(*ref_rejected_q)

    chosen_ratios = policy_chosen - ref_chosen
    rejected_ratios = policy_rejected - ref_rejected

    q_chosen = QuantizeSTE.apply(chosen_ratios)
    q_rejected = QuantizeSTE.apply(rejected_ratios)

    logits = beta * (q_chosen - q_rejected)
    return -F.logsigmoid(logits).mean()
