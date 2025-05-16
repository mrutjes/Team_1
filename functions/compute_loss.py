import torch
import torch.nn as nn

def compute_loss(p_borylation, borylation_mask, predicted_yield, true_yield,
                 alpha=0.5, pos_weight_val=15.0) -> tuple:
    """
    Compute the loss of the """

    pos_weight = torch.tensor([pos_weight_val], device=p_borylation.device)
    loss_site = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(p_borylation, borylation_mask)

    loss_yield = nn.MSELoss()(predicted_yield, true_yield)

    total_loss = (1-alpha) * loss_site + (alpha) * loss_yield
    return total_loss, loss_site, loss_yield
