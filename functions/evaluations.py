import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import spearmanr, pearsonr

def evaluate_yield(y_true, y_pred):
    """
    Evaluates the yield predictions (y_pred) against the ground truth (y_true).
    Returns a dictionary with MSE, MAE, and R2 scores as a dict.
    """

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"yield_MSE": mse, "yield_MAE": mae, "yield_R2": r2}


def evaluate_borylation_site(pred_logits, true_mask):
    """
    Evaluates binary classification performance for predicted borylation sites at the node level.
    """
    pred_probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    true_mask = true_mask.detach().cpu().numpy()
    
    pred_binary = (pred_probs >= 0.5).astype(int)

    return {
        "site_Accuracy": accuracy_score(true_mask, pred_binary),
        "site_Precision": precision_score(true_mask, pred_binary, zero_division=0),
        "site_AUC": roc_auc_score(true_mask, pred_probs) if len(set(true_mask)) > 1 else float("nan")
    }

def topk_accuracy(p_borylation, borylation_mask, batch, k=3):
    """
    Computes the top-k accuracy for borylation site predictions.
    """

    correct = 0
    total = batch.max().item() + 1

    for graph_id in range(total):
        node_mask = (batch == graph_id)
        preds = p_borylation[node_mask]
        target = borylation_mask[node_mask]

        topk = preds.topk(k).indices
        true_index = target.argmax()

        if true_index in topk:
            correct += 1

    return correct / total

def evaluate_model(model, dataloader, device):
    """
    Evaluates the model on a dataset for both graph-level yield prediction and 
    node-level borylation site prediction.
    """

    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    all_site_logits = []
    all_site_masks = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            p_borylation, predicted_yield = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

            # Yield
            all_y_true.append(batch.y.cpu())
            all_y_pred.append(predicted_yield.cpu())

            # Borylation mask
            all_site_logits.append(p_borylation)
            all_site_masks.append(batch.borylation_mask)

    # concat
    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()
    
    site_logits = torch.cat(all_site_logits)
    site_masks = torch.cat(all_site_masks)

    # evaluate
    metrics = {}
    metrics.update(evaluate_yield(y_true, y_pred))
    metrics.update(evaluate_borylation_site(site_logits, site_masks))

    return metrics, y_true, y_pred
