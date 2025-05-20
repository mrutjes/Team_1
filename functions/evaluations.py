import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import spearmanr, pearsonr
import numpy as np
from torch_geometric.utils import softmax


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
    pred_probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    true_mask = true_mask.detach().cpu().numpy()

    pred_binary = (pred_probs >= 0.5).astype(int)

    if true_mask.sum() == 0:
        return {
            "site_Accuracy": float("nan"),
            "site_Precision": float("nan"),
            "site_Recall": float("nan"),
            "site_F1": float("nan"),
            "site_AUC": float("nan")
        }

    return {
        "site_Accuracy": accuracy_score(true_mask, pred_binary),
        "site_Precision": precision_score(true_mask, pred_binary, zero_division=0),
        "site_Recall": recall_score(true_mask, pred_binary, zero_division=0),
        "site_F1": f1_score(true_mask, pred_binary, zero_division=0),
        "site_AUC": roc_auc_score(true_mask, pred_probs)
    }


def topk_accuracy_softmax(logits, target_indices, batch, k=1):
    probs = softmax(logits, batch)
    num_graphs = batch.max().item() + 1
    topk_acc = 0

    for graph_id in range(num_graphs):
        node_mask = (batch == graph_id)
        topk_nodes = probs[node_mask].topk(k).indices
        true_index = target_indices[graph_id]

        if true_index in topk_nodes:
            topk_acc += 1

    return topk_acc / num_graphs


def evaluate_model(model, dataloader, device, yield_min, yield_max):
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

    mask = y_true != 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    y_true = y_true * (yield_max - yield_min) + yield_min
    y_pred = y_pred * (yield_max - yield_min) + yield_min

    site_logits = torch.cat(all_site_logits)
    site_masks = torch.cat(all_site_masks)

    # evaluate
    metrics = {}
    metrics.update(evaluate_yield(y_true, y_pred))
    metrics.update(evaluate_borylation_site(site_logits, site_masks))

    return metrics, y_true, y_pred
