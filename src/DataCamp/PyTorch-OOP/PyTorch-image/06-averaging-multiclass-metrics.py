import torch
from torchmetrics.classification import F1Score, Precision, Recall


def print_metric_group(metric_name: str, per_class, micro, macro, weighted, class_to_idx):
    print(f"\n{metric_name}")
    print(f"{'-' * len(metric_name)}")

    # Convert per-class tensor to readable class-name dictionary.
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    per_class_dict = {
        idx_to_class[i]: round(float(score), 4)
        for i, score in enumerate(per_class.tolist())
    }

    print(f"Per class: {per_class_dict}")
    print(f"Micro:    {float(micro):.4f}")
    print(f"Macro:    {float(macro):.4f}")
    print(f"Weighted: {float(weighted):.4f}")


def run_evaluation_loop(num_classes: int, class_to_idx):
    # Simulated model outputs (logits) and labels in mini-batches.
    logits_batches = [
        torch.tensor([[3.2, 0.6, 0.2], [0.9, 2.4, 0.8], [0.1, 1.2, 2.0]]),
        torch.tensor([[2.3, 1.1, 0.4], [0.3, 2.5, 0.2], [0.2, 0.7, 2.1], [1.8, 1.7, 0.1]]),
    ]
    label_batches = [torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2, 1])]

    metric_precision_macro = Precision(task="multiclass", num_classes=num_classes, average="macro")
    metric_recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
    metric_precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)

    with torch.no_grad():
        for logits, labels in zip(logits_batches, label_batches):
            preds = torch.argmax(logits, dim=1)
            metric_precision_macro.update(preds, labels)
            metric_recall_macro.update(preds, labels)
            metric_precision_per_class.update(preds, labels)

    precision_macro = metric_precision_macro.compute()
    recall_macro = metric_recall_macro.compute()
    precision_per_class = metric_precision_per_class.compute()

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    precision_by_class = {
        idx_to_class[i]: round(float(score), 4)
        for i, score in enumerate(precision_per_class.tolist())
    }

    print("\nEvaluation loop")
    print("---------------")
    print(f"Precision (macro): {float(precision_macro):.4f}")
    print(f"Recall (macro):    {float(recall_macro):.4f}")
    print(f"Precision per class: {precision_by_class}")

    metric_precision_macro.reset()
    metric_recall_macro.reset()
    metric_precision_per_class.reset()


if __name__ == "__main__":
    # Example multiclass predictions and targets (class indices).
    predictions = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0, 1, 2])
    targets = torch.tensor([0, 1, 1, 1, 2, 2, 2, 0, 0, 2])
    num_classes = 3
    class_to_idx = {"class_0": 0, "class_1": 1, "class_2": 2}

    # Recall metrics.
    recall_per_class = Recall(task="multiclass", num_classes=num_classes, average=None)
    recall_micro = Recall(task="multiclass", num_classes=num_classes, average="micro")
    recall_macro = Recall(task="multiclass", num_classes=num_classes, average="macro")
    recall_weighted = Recall(task="multiclass", num_classes=num_classes, average="weighted")

    # Precision metrics.
    precision_per_class = Precision(task="multiclass", num_classes=num_classes, average=None)
    precision_micro = Precision(task="multiclass", num_classes=num_classes, average="micro")
    precision_macro = Precision(task="multiclass", num_classes=num_classes, average="macro")
    precision_weighted = Precision(task="multiclass", num_classes=num_classes, average="weighted")

    # F1-score metrics.
    f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)
    f1_micro = F1Score(task="multiclass", num_classes=num_classes, average="micro")
    f1_macro = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    f1_weighted = F1Score(task="multiclass", num_classes=num_classes, average="weighted")

    # Compute each metric.
    r_pc = recall_per_class(predictions, targets)
    r_mi = recall_micro(predictions, targets)
    r_ma = recall_macro(predictions, targets)
    r_wt = recall_weighted(predictions, targets)

    p_pc = precision_per_class(predictions, targets)
    p_mi = precision_micro(predictions, targets)
    p_ma = precision_macro(predictions, targets)
    p_wt = precision_weighted(predictions, targets)

    f_pc = f1_per_class(predictions, targets)
    f_mi = f1_micro(predictions, targets)
    f_ma = f1_macro(predictions, targets)
    f_wt = f1_weighted(predictions, targets)

    print_metric_group("Recall", r_pc, r_mi, r_ma, r_wt, class_to_idx)
    print_metric_group("Precision", p_pc, p_mi, p_ma, p_wt, class_to_idx)
    print_metric_group("F1", f_pc, f_mi, f_ma, f_wt, class_to_idx)
    run_evaluation_loop(num_classes, class_to_idx)