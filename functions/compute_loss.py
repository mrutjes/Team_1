import torch.nn as nn

def compute_loss(p_borylation, borylation_mask, reactivity_score, reactivity_target, predicted_yield, true_yield,
                 alpha=1.0, beta=1.0, gamma=0.1):
    # Borylation: Binary classification (sigmoid niet in model maar in BCEWithLogits)
    loss_site = nn.BCEWithLogitsLoss()(p_borylation, borylation_mask)

    # Reactivity: Regressie per node
    loss_react = nn.MSELoss()(reactivity_score, reactivity_target)

    # Yield: Regressie per graaf
    loss_yield = nn.MSELoss()(predicted_yield, true_yield)

    total_loss = alpha * loss_site + beta * loss_react + gamma * loss_yield
    return total_loss, loss_site, loss_react, loss_yield

