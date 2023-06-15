import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import undeco


def compute_metric(metric):
    def wrapper(*args, **kwargs):
        score, value_idx = metric(*args, **kwargs)
        result = (score * value_idx).sum(axis=-1)
        return result
    return wrapper


@compute_metric
def cg_fn(relevance: torch.Tensor) -> torch.Tensor:
    value_idx = torch.where(relevance == -1, 0, 1).to(relevance.device)
    return relevance, value_idx


@compute_metric
def dcg_fn(relevance: torch.Tensor, exp: bool=False) -> torch.Tensor:
    if exp:
        relevance = 2**relevance - 1
    weights = torch.log(torch.tensor(range(1, relevance.shape[-1] + 1)) + 1).to(relevance.device)
    dcg = relevance / weights
    value_idx = torch.where(relevance == -1, 0, 1).to(relevance.device)

    return dcg, value_idx


@compute_metric
def idcg_fn(relevance: torch.Tensor, exp: bool=False) -> torch.Tensor:
    device = relevance.device
    relevance, _ = torch.sort(relevance.to("cpu"), dim=1, descending=True)
    relevance = relevance.to(device)
    idcg, value_idx = undeco(dcg_fn)(relevance, exp)

    return idcg, value_idx


@compute_metric
def ndcg_fn(relevance: torch.Tensor, exp: bool=False) -> torch.Tensor:
    dcg, value_idx = undeco(dcg_fn)(relevance, exp)
    idcg = idcg_fn(relevance, exp)
    return dcg / idcg.unsqueeze(-1), value_idx
