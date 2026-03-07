import os
import json
import numpy as np


def save_training_models(args, last_weights, best_weights=None):
    """
    Save last epoch and best model weights.

    Parameters
    ----------
    args : argparse.Namespace
        Training arguments (contains paths).
    last_weights : object
        Weights from final epoch.
    best_weights : object, optional
        Best model weights (based on validation F1).
    """

    os.makedirs(args.model_save_path, exist_ok=True)

    ##Last epoch saving
    last_path = os.path.join(
        args.model_save_path,
        "last_epoch_model.npy"
    )

    np.save(last_path, last_weights, allow_pickle=True)

    print("Saved last epoch model:", last_path)

    #Best model saving
    if best_weights is not None:

        best_path = os.path.join(
            args.model_save_path,
            "best_model.npy"
        )

        np.save(best_path, best_weights, allow_pickle=True)

        print("Saved best model:", best_path)

        ##saving args as a config file
        os.makedirs(os.path.dirname(args.config_saved_path), exist_ok=True)

        with open(args.config_saved_path, "w") as f:
            json.dump(vars(args), f, indent=4)

        print("Saved config:", args.config_saved_path)