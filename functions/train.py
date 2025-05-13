import torch
from functions.compute_loss import compute_loss


def train_MPNN_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()
        p_borylation, reactivity_score, predicted_yield = model.forward(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        loss, l_site, l_react, l_yield = compute_loss(
            p_borylation, batch.borylation_mask,
            reactivity_score, batch.reactivity,
            predicted_yield, batch.y
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

