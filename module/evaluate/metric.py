import torch


def compute_cg(relevance: torch.Tensor) -> torch.Tensor:
    batch_size = relevance.shape[0]
    device = relevance.device
    value_idx = torch.where(relevance == -1, 0, 1).to(device)
    cg = (relevance * value_idx).sum() / batch_size

    return cg


def compute_dcg(relevance: torch.Tensor, exp=False):
    batch_size = relevance.shape[0]
    device = relevance.device
    value_idx = torch.where(relevance == -1, 0, 1).to(device)
    if exp:
        relevance = 2**relevance - 1
    denom = torch.log(torch.tensor(range(1, relevance.shape[1] + 1)) + 1).to(device)
    dcg = (relevance / denom * value_idx).sum() / batch_size

    return dcg


def compute_idcg(relevance: torch.Tensor, exp=False):
    batch_size = relevance.shape[0]
    device = relevance.device
    value_idx = torch.where(relevance == -1, 0, 1).to(device)
    relevance, _ = torch.sort(relevance.to("cpu"), dim=1, descending=True)
    relevance = relevance.to(device)
    if exp:
        relevance = 2**relevance - 1
    denom = torch.log(torch.tensor(range(1, relevance.shape[1] + 1)) + 1).to(device)
    idcg = (relevance / denom * value_idx).sum() / batch_size

    return idcg


def compute_ndcg(relevance: torch.Tensor, exp=False):
    dcg = compute_dcg(relevance, exp)
    idcg = compute_idcg(relevance, exp)
    ndcg = dcg / idcg

    return ndcg


if __name__ == "__main__":
    relevance = torch.tensor([[0,2,3,1,1,2], [1,3,2,2,1,0]])
    compute_cg(relevance)
    compute_dcg(relevance)
    compute_idcg(relevance)
    compute_ndcg(relevance)
