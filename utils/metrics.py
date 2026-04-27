import torch

def compute_metrics(preds, labels):
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)

    # Accuracy
    acc = (preds == labels).float().mean().item()

    # Confusion components
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    # Precision
    precision = tp / (tp + fp + 1e-8)

    # Recall (VERY IMPORTANT)
    recall = tp / (tp + fn + 1e-8)

    # F1 Score
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def confusion_matrix(preds, labels):
    cm = [[0, 0], [0, 0]]
    for p, l in zip(preds, labels):
        cm[l][p] += 1
    return cm