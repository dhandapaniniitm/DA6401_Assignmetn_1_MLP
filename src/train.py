"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from utils.wand_util import setup_wandb
from utils.plot_util import plot_training_curves
from utils.model_io import save_training_models
from datetime import datetime
from utils.metrics_util import evaluate_model_core , log_eval_results
import os
import wandb

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument("-d","--dataset",type=str,default="mnist")
    parser.add_argument("-vs","--val_size_from_train",type=float,default=0.1)
    parser.add_argument("-e","--epochs",type=int,default=20)
    parser.add_argument("-b","--batch_size",type=int,default=64)
    parser.add_argument("-l","--loss",type=str,default="cross_entropy")
    parser.add_argument("-o","--optimizer",type=str,default="rmsprop")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.0005)
    parser.add_argument("-wd","--weight_decay",type=float,default=0.0)
    parser.add_argument("-nhl","--num_layers",type=int,default=3)
    parser.add_argument("-sz","--hidden_size",type=int,nargs='+',default=[128,128,64])
    parser.add_argument("-a","--activation",type=str,nargs='+',default=["relu","relu","relu"])
    parser.add_argument("-as","--activations",type=str,nargs='+',default=["relu","relu","relu"])
    parser.add_argument("-w_i","--weight_init",type=str,default="xavier")
    
    parser.add_argument("-s","--seed",type=int,default=42,help="Choose a seed")
    parser.add_argument("-op", "--output_path", type=str, default="output/")
    parser.add_argument("-ms","--model_save_path",type=str,default="output/models/",help="path to save the trained model")
    parser.add_argument("-cs","--config_saved_path",type=str,default="output/best_config.json",help="path to save the config of args used for training")
    
    parser.add_argument("-wa","--wandb_api_key",type=str,default=None)
    parser.add_argument("-wp","--wandb_project",type=str,default="DA6401_assn_1_mlp_trail_1")
    parser.add_argument("--sweep",action="store_true",help="Enables Weights & Biases hyperparameter sweep")
    
    return parser.parse_args()

def train_once(args, run=None):
    """
    Runs one training experiment.
    Used by both normal training and wandb sweep for the experiments.
    """
    ##Load data
    X_train, y_train, X_test, y_test = load_data(args.dataset)
    
    if run is not None:
        wandb.log({
            "dataset/train_samples": X_train.shape[0],
            "dataset/features": X_train.shape[1],
            "dataset/classes": y_train.shape[1],
        })
    
    ##Initialize model and use the dim on data to decide the input and output dim.
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = NeuralNetwork(args, input_dim=input_dim, output_dim=output_dim)

    ##Shuffle the training data
    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    y_train = y_train[perm]

    ##Validation split of 10% from train set
    val_size = int(args.val_size_from_train * X_train.shape[0])
    X_val, y_val = X_train[:val_size], y_train[:val_size]
    X_train_sub, y_train_sub = X_train[val_size:], y_train[val_size:]

    os.makedirs(args.model_save_path, exist_ok=True)

    ## sending the data to train function in Neural networks
    history, best_weights, last_weights = model.train(
        X_train_sub,
        y_train_sub,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_val,
        y_val=y_val,
        wandb=run
    )

    ##We'll get the metric, best and last checkpoint from the model and will Save it using the below function
    save_training_models(
        args,
        last_weights=last_weights,
        best_weights=best_weights
    )

    ##Plotting the training curves
    plot_training_curves(
        history,
        output_path=args.output_path,
        run=run
    )
    
    print("\nRunning Final evaluation...")
    ##Evaluate last model checkpoint on test data
    model.set_weights(last_weights)
    last_test_results = evaluate_model_core(model, X_test, y_test)
    
    ##Evaluate best model checkpoint on test data
    model.set_weights(best_weights)
    best_test_results = evaluate_model_core(model, X_test, y_test)
    
    ##test_results = evaluate_model_core(model, X_test, y_test)
    ##Logging last and best results on wanbd
    log_eval_results(
        last_test_results,
        wandb=wandb if run else None,
        prefix="test_last"
    )

    log_eval_results(
        best_test_results,
        wandb=wandb if run else None,
        prefix="test_best"
    )

    return history

def run_sweep_training():
    """
    Sweep training entrypoint (WandB)
    """
    with wandb.init() as run:

        config = wandb.config

        #convert config to args setup for training
        class Args:
            pass

        args = Args()

        for k, v in config.items():
            setattr(args, k, v)

        ## DEFAULTS parameter for sweep training
        args.dataset = "mnist"
        args.val_size_from_train = 0.1
        args.seed = 42

        ##Architecture expansion
        args.hidden_size = [args.hidden_size] * args.num_layers
        args.activation = [args.activation] * args.num_layers
        args.activations = [args.activation]

        ##unique paths to save in wandb and local
        run_id = run.id

        args.model_save_path = f"output/sweeps/{run_id}/models/"
        args.output_path = f"output/sweeps/{run_id}/plots/"
        args.config_saved_path = f"output/sweeps/{run_id}/config.json"

        os.makedirs(args.model_save_path, exist_ok=True)
        os.makedirs(args.output_path, exist_ok=True)

        np.random.seed(args.seed)

        ## Using the Train function
        history = train_once(args, run)

        ##logging FINAL METRICS for summary.
        wandb.log({
            "final/train_accuracy": history["train_accs"][-1],
            "final/val_accuracy": history["val_accs"][-1],
            "final/train_loss": history["train_losses"][-1],
            "final/val_loss": history["val_losses"][-1],
        })

## Different Hyperparameter tuning values.
def get_sweep_config():
    return {
        "method": "random",
        "metric": {
            "name": "val/accuracy",
            "goal": "maximize"
        },
        "parameters": {

            "epochs": {"values": [10,20,30]},            
            "batch_size": {"values": [32, 64, 128]},

            "learning_rate": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-1
            },

            "optimizer": {
                "values": ["sgd", "momentum", "nag", "rmsprop"]
            },

            "activation": {
                "values": ["sigmoid", "tanh", "relu"]
            },

            "weight_init": {
                "values": ["random", "xavier"]
            },

            "num_layers": {"values": [2, 3, 4, 5, 6]},
            "hidden_size": {"values": [32, 64, 128]},

            "loss": {
                "values": ["mse", "ce"]
            },

            "weight_decay": {
                "values": [0.0, 1e-4, 1e-3]
            }
        }
    }

def main():
    """
    Main training function.
    """
    ##Sweep project name 
    #SWEEP_PROJECT = "mnist-sweep"
    SWEEP_PROJECT = "mnist-sweep-100"
    
    args = parse_arguments()
    args.activations = args.activation
    
    np.random.seed(args.seed)
    
    if len(args.hidden_size) != args.num_layers:
        raise ValueError("hidden_size length must equal num_layers")
    
    ## based on Sweep flag from cli , we perform Sweep training
    if args.sweep:
        setup_wandb(args.wandb_api_key)
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(
            sweep_config,
            project=SWEEP_PROJECT,
        )
        wandb.agent(
            sweep_id,
            function=run_sweep_training,
            #count=10
            count=100
        )
        
        ## After sweep training , It shows top 5 runs based on validation accuracy.
        print("\nFetching Top 5 Runs...")

        api = wandb.Api()
    
        runs = api.runs(
            f"{wandb.api.default_entity}/{SWEEP_PROJECT}",
            order="-summary_metrics.val/accuracy"
        )
        top5_runs = runs[:5]
    
        for i, run in enumerate(top5_runs, 1):
            print(f"\nTop {i} Run")
            print("Run:", run.name)
            print("Val Accuracy:", run.summary.get("val/accuracy"))
            print("Config:", run.config)

        
        
        return   ##It will stop normal training
    
    ## If Sweep is not enabled , then normal training is activated and it also check whether wandb logging is required or not.
    if args.wandb_api_key:
        setup_wandb(args.wandb_api_key)
        run = wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"train_{args.optimizer}_lr{args.learning_rate}_{datetime.now().strftime('%H%M%S')}"
        )
    else:
        run = None
    train_once(args, run)
    if run is not None:
        run.finish()
    print("Training complete!")

if __name__ == '__main__':
    main()

"""
Command to run
python src/train.py -d mnist -e 20 -b 64 -l ce -o rmsprop -lr 0.0005 -nhl 3 -sz 128 128 64 -a relu relu relu -w_i xavier -s 42
"""