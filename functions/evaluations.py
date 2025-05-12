import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from scipy.stats import spearmanr, pearsonr

def evaluate_yield(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"yield_MSE": mse, "yield_MAE": mae, "yield_R2": r2}


def evaluate_borylation_site(pred_logits, true_mask):
    pred_probs = torch.sigmoid(pred_logits).detach().cpu().numpy()
    true_mask = true_mask.detach().cpu().numpy()
    
    pred_binary = (pred_probs >= 0.5).astype(int)

    return {
        "site_Accuracy": accuracy_score(true_mask, pred_binary),
        "site_Precision": precision_score(true_mask, pred_binary, zero_division=0),
        "site_Recall": recall_score(true_mask, pred_binary, zero_division=0),
        "site_F1": f1_score(true_mask, pred_binary, zero_division=0),
        "site_AUC": roc_auc_score(true_mask, pred_probs) if len(set(true_mask)) > 1 else float("nan")
    }


def evaluate_reactivity(pred_score, true_score):
    pred = pred_score.detach().cpu().numpy()
    true = true_score.detach().cpu().numpy()
    
    mse = mean_squared_error(true, pred)
    spearman_corr = spearmanr(true, pred).correlation
    pearson_corr = pearsonr(true, pred)[0]
    
    return {
        "react_MSE": mse,
        "react_Spearman": spearman_corr,
        "react_Pearson": pearson_corr
    }

def evaluate_model(model, dataloader, device):
    model.eval()
    
    all_y_true = []
    all_y_pred = []
    
    all_site_logits = []
    all_site_masks = []

    all_reactivity_pred = []
    all_reactivity_true = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            p_borylation, reactivity_score, predicted_yield = model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            )

            # Yield
            all_y_true.append(batch.y.cpu())
            all_y_pred.append(predicted_yield.cpu())

            # Borylation mask
            all_site_logits.append(p_borylation)
            all_site_masks.append(batch.borylation_mask)

            # Reactivity
            all_reactivity_pred.append(reactivity_score)
            all_reactivity_true.append(batch.reactivity)

    # concat
    y_true = torch.cat(all_y_true).numpy()
    y_pred = torch.cat(all_y_pred).numpy()
    
    site_logits = torch.cat(all_site_logits)
    site_masks = torch.cat(all_site_masks)
    
    react_pred = torch.cat(all_reactivity_pred)
    react_true = torch.cat(all_reactivity_true)

    # evaluate
    metrics = {}
    metrics.update(evaluate_yield(y_true, y_pred))
    metrics.update(evaluate_borylation_site(site_logits, site_masks))
    metrics.update(evaluate_reactivity(react_pred, react_true))

    return metrics
