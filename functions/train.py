import torch
from functions.compute_loss import compute_loss

def train_MPNN_model(model, dataloader, optimizer, device, pos_weight_val=15.0):
    model.train()
    total_loss = 0
    total_site_loss = 0
    total_yield_loss = 0

    for batch in dataloader:
        batch = batch.to(device)

        optimizer.zero_grad()
        p_borylation, predicted_yield = model.forward(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch
        )

        loss, l_site, l_yield = compute_loss(
            p_borylation, batch.borylation_mask.float(),
            predicted_yield, batch.y,
            pos_weight_val=pos_weight_val  # <- dit is optioneel
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_site_loss += l_site.item()
        total_yield_loss += l_yield.item()

    num_batches = len(dataloader)
    return {
        "total": total_loss / num_batches,
        "site": total_site_loss / num_batches,
        "yield": total_yield_loss / num_batches
    }
