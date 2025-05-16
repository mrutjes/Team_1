import torch
import torch.nn as nn

def compute_loss(p_borylation, borylation_mask, predicted_yield, true_yield,
                 alpha=1.0, beta=1.0, gamma=0.1, pos_weight_val=15.0) -> tuple:
    """
    Compute the loss of the """

    pos_weight = torch.tensor([pos_weight_val], device=p_borylation.device)
    loss_site = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(p_borylation, borylation_mask)

    loss_yield = nn.MSELoss()(predicted_yield, true_yield)

    total_loss = alpha * loss_site + beta + gamma * loss_yield
    return total_loss, loss_site, loss_yield
