"""
plot metrics
"""

import os
import wandb
import matplotlib.pyplot as plt

def plot_training_curves(history,output_path,run=None,):
    
    plotting_path = os.path.join(output_path, "plots")
    os.makedirs(plotting_path, exist_ok=True)
    
    train_losses = history["train_losses"]
    val_losses = history["val_losses"]
    train_accs = history["train_accs"]
    val_accs = history["val_accs"]
    
    ##Loss Curvess
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    loss_plot = os.path.join(plotting_path, "loss_vs_epoch_curve.png")
    plt.savefig(loss_plot)
    plt.close()

    ##Accuracy Curvess
    plt.figure()
    plt.plot(train_accs, label="Train")
    plt.plot(val_accs, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    acc_plot = os.path.join(plotting_path, "accuracy_vs_epoch_curve.png")
    plt.savefig(acc_plot)
    plt.close()

    ## WandBLogginggs
    if run is not None:
        run.log({
            "loss_curve": wandb.Image(loss_plot),
            "accuracy_curve": wandb.Image(acc_plot),
        })

    return {
        "loss_plot": loss_plot,
        "accuracy_plot": acc_plot,
    }