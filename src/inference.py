"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import os
import json
import numpy as np
import wandb
from types import SimpleNamespace
from datetime import datetime
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data
from utils.metrics_util import evaluate_model_core , log_eval_results
#from utils.model_util import save_model, load_model
from utils.wand_util import setup_wandb

def parse_arguments():
    """
    Parse command-line arguments for inference.
    
    TODO: Implement argparse with:
    - model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
    - dataset: Dataset to evaluate on
    - batch_size: Batch size for inference
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    
    parser.add_argument("-d","--dataset",type=str,default="mnist")
    parser.add_argument("-vs","--val_size_from_train",type=float,default=0.1)
    parser.add_argument("-e","--epochs",type=int,default=20)
    parser.add_argument("-b","--batch_size",type=int,default=64)
    parser.add_argument("-l","--loss",type=str,default="ce")
    parser.add_argument("-o","--optimizer",type=str,default="rmsprop")
    parser.add_argument("-lr","--learning_rate",type=float,default=0.0005)
    parser.add_argument("-wd","--weight_decay",type=float,default=0.0)
    parser.add_argument("-nhl","--num_layers",type=int,default=3)
    parser.add_argument("-sz","--hidden_size",type=int,nargs='+',default=[128,128,64])
    parser.add_argument("-a","--activation",type=str,nargs='+',default=["relu","relu","relu"])
    parser.add_argument("-as","--activations",type=str,nargs='+',default=["relu","relu","relu"])
    parser.add_argument("-w_i","--weight_init",type=str,default="xavier")
    
    parser.add_argument("-s","--seed",type=int,default=42)
    parser.add_argument("-op","--output_path",type=str,default="output/")
    parser.add_argument("-ms","--model_save_path",type=str,default="output/models/")
    parser.add_argument("-cs","--config_saved_path",type=str,default="output/best_config.json")
    #parser.add_argument("-cs","--config_saved_path",type=str,default="best_config.json")
    
    parser.add_argument("-wa","--wandb_api_key",type=str,default=None)
    parser.add_argument("-wp","--wandb_project",type=str,default="DA6401_assn_1_mlp")
    
    parser.add_argument("--model_path",type=str,default="output/models/best_model.npy",help="Relative path to saved model weights")
    #parser.add_argument("--model_path",type=str,default="best_model.npy",help="Relative path to saved model weights")
    return parser.parse_args()


def build_and_load_model(args, input_dim, output_dim):
    """
    load saved training config and initalize the model
    """
    if os.path.exists(args.config_saved_path):
        with open(args.config_saved_path, "r") as f:
            saved_config = json.load(f)

        for k, v in saved_config.items():
            setattr(args, k, v)
            
    cli_args = SimpleNamespace(**vars(args))

    ##recreate architecture
    model = NeuralNetwork(
        cli_args,
        input_dim=input_dim,
        output_dim=output_dim
    )

    ##load weights using our model archtiecture and weights
    load_model(model, args.model_path)

    print(f"Loaded model from: {args.model_path}")

    return model

def load_model(model, model_path):
    weights = np.load(model_path, allow_pickle=True).item()
    model.set_weights(weights)
    return model

def evaluate_model(model, X_test, y_test,wandb_run=None): 
    """
    Evaluate model on test data.
        
    TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    results = evaluate_model_core(model, X_test, y_test)
    
    return results

def main():
    """
    Main inference function.

    TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()
    args.activations = args.activation
    if args.wandb_api_key:
        setup_wandb(args.wandb_api_key)
        run = wandb.init(
            project=args.wandb_project,
            name=f"inference_{datetime.now().strftime('%H%M%S')}",
            config=vars(args),
        )
    else:
        run = None

    # ---------- Load dataset ----------
    X_train, y_train, X_test, y_test = load_data(args.dataset)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    ##output_dim = len(np.unique(y_train))

    # ---------- Load model ----------
    model = build_and_load_model(args, input_dim, output_dim)
    
    assert len(args.hidden_size) == args.num_layers, "hidden_size count must equal num_layers"
    
    # ---------- Evaluate ----------
    results = evaluate_model(model, X_test, y_test, run)
    
    log_eval_results(
        results,
        wandb=wandb if run else None,
        prefix="test"
    )
    
    print("\n===== TEST RESULTS =====")
    print(f"Loss      : {results['loss']:.4f}")
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1 Score  : {results['f1']:.4f}")

    if run:
        run.finish()

    print("\nEvaluation complete!")

    return results

if __name__ == '__main__':
    main()
