import torch
import torch.nn as nn

def compute_loss(p_borylation, borylation_mask, reactivity_score, reactivity_target, predicted_yield, true_yield,
                 alpha=1.0, beta=1.0, gamma=0.1, pos_weight_val=15.0):
    # --- Borylation: Binary classification met correctie voor class imbalance ---
    pos_weight = torch.tensor([pos_weight_val], device=p_borylation.device)
    loss_site = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(p_borylation, borylation_mask.float())

    # --- Reactivity: Regressie ---
    loss_react = nn.MSELoss()(reactivity_score, reactivity_target)

    # --- Yield: Regressie ---
    loss_yield = nn.MSELoss()(predicted_yield, true_yield)

    total_loss = alpha * loss_site + beta * loss_react + gamma * loss_yield
    return total_loss, loss_site, loss_react, loss_yield
