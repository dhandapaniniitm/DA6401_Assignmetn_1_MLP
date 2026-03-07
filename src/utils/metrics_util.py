"""
metrics_util.py
Python File to have all metrics related functions which is used across the pipelin
"""
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report,balanced_accuracy_score

def get_predictions(model, x):
    logits = model.forward(x)
    return np.argmax(logits, axis=1)

def get_accuracyy(model,x,y):
    preds = get_predictions(model, x)
    labels = np.argmax(y,axis=1)
    return np.mean(preds == labels)

def get_f1(model, x, y):
    preds = get_predictions(model, x)
    labels = np.argmax(y, axis=1)
    return f1_score(labels, preds, average="macro", zero_division=0)

def evaluate_model_core(model, x, y):
    
    ##Forward pass to the model
    logits = model.forward(x)
    ## getting loss for reference from the model
    loss = model.compute_loss(y, logits)

    y_pred = np.argmax(logits, axis=1)
    y_true = np.argmax(y, axis=1)

    results = {
        "logits": logits,
        "y_true": y_true,
        "y_pred": y_pred,
        "loss": float(loss),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
    return results


def log_eval_results(results, wandb=None, prefix="test"):
    print(f"\n[{prefix.upper()} RESULTS]")
    print("Accuracy:", results["accuracy"])
    print("precision:", results["precision"])
    print("recall:", results["recall"])
    print("f1:", results["f1"])
    print("Balanced Accuracy:", results["balanced_accuracy"])
    print("Confusion matrix\n", results["confusion_matrix"])

    c_report = classification_report(results["y_true"],results["y_pred"], zero_division=0)

    print("Classification report\n", c_report)

    if wandb is not None:
        wandb.log({
            f"{prefix}/accuracy": results["accuracy"],
            f"{prefix}/f1": results["f1"],
            f"{prefix}/precision": results["precision"],
            f"{prefix}/recall": results["recall"],
            f"{prefix}/balanced_accuracy": results["balanced_accuracy"],
            f"{prefix}/confusion_matrix":
                wandb.plot.confusion_matrix(
                    preds=results["y_pred"],
                    y_true=results["y_true"]
                ),
        })

        wandb.summary[f"{prefix}/classification_report"] = c_report