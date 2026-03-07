# Assignment 1: Multi-Layer Perceptron for Image Classification

**Course Name:** DA6401\
**Author:** Dhadapani N\
**Roll No:** BT25S018

## Links

**Link to WandB Report:**
[WandB Experiment Report](https://wandb.ai/bt25s018-iitm/mnist-sweep-100/reports/DA6401-Assignment_1_MLP--VmlldzoxNjEzMzAyOQ?accessToken=c51rfr7cab263k49xrxxc79i3jn24qxhpq3tnrxw1ert131z4p3jpqunk7wb822o)

**Link to GitHub Repository:**
[Project GitHub Repository](https://github.com/dhandapaniniitm/DA6401_Assignmetn_1_MLP.git)

------------------------------------------------------------------------

## Overview

This assignment implements a **Multi-Layer Perceptron (MLP)** from
scratch using **NumPy only**.\
All neural network components including layers, activations, optimizers,
and loss functions are manually implemented.

The model is trained and evaluated on:

-   MNIST
-   Fashion-MNIST

Weights & Biases (W&B) is used for experiment tracking and hyperparameter sweeps.

------------------------------------------------------------------------

## Project Structure

	.
	├── models
	├── output
	│   ├── best_config.json
	│   ├── models
	│   │   ├── best_model.npy
	│   │   └── last_epoch_model.npy
	│   └── plots
	│       ├── accuracy_vs_epoch_curve.png
	│       └── loss_vs_epoch_curve.png
	├── README.md
	├── requirements.txt
	└── src
	    ├── ann
	    │   ├── activations.py
	    │   ├── __init__.py
	    │   ├── metrics_util.py
	    │   ├── neural_layer.py
	    │   ├── neural_network.py
	    │   ├── objective_functions.py
	    │   ├── optimizers.py
	    ├── best_config.json
	    ├── best_model.npy
	    ├── inference.py
	    ├── train.py
	    └── utils
		├── data_loader.py
		├── __init__.py
		├── metrics_util.py
		├── model_io.py
		├── model_util.py
		└── plot_util.py


------------------------------------------------------------------------

## Installation

Install all dependencies using:

    pip install -r requirements.txt
    		
    		or

    pip install numpy matplotlib tensorflow keras wandb scikit-learn
------------------------------------------------------------------------

# Command Line Interface (CLI)

The project provides CLI commands for training, inference, and
hyperparameter sweeps.

------------------------------------------------------------------------

## Training

Run the training script:

    python3 src/train.py [arguments]

### Example Training Command

    python3 src/train.py -d mnist -e 20 -b 64 -l ce -o rmsprop -lr 0.0005 -nhl 3 -sz 128 128 64 -a relu relu relu -w_i xavier -s 42 -wa Wandb_API_Key

Configuration:

-   Dataset: MNIST
-   Epochs: 20
-   Batch size: 64
-   Optimizer: RMSProp
-   Learning rate: 0.0005
-   Hidden layers: 128 → 128 → 64
-   Activation: ReLU
-   Weight Initialization: Xavier
-   Seed: 42

------------------------------------------------------------------------

## Inference

Run inference using a saved model:

    python3 inference.py -d mnist -b 64 -sz 128 128 64 -a relu relu relu --model_path output/models/best_model.npy -wa Wandb_API_Key

This loads the trained model and evaluates it on the dataset.

------------------------------------------------------------------------

## Hyperparameter Sweep (Weights & Biases)

To run hyperparameter sweep experiments:

    python3 train.py -wa Wandb_API_Key --sweep

This runs multiple training jobs with different hyperparameter
configurations.

------------------------------------------------------------------------

# CLI Parameters

  Argument   Description                       Default
  ---------- --------------------------------- ---------------------------
  -d         Dataset (mnist / fashion_mnist)   mnist
  -vs        Validation split                  0.1
  -e         Epochs                            20
  -b         Batch size                        64
  -l         Loss function (ce / mse)          ce
  -o         Optimizer                         rmsprop
  -lr        Learning rate                     0.0005
  -wd        Weight decay                      0
  -nhl       Number of hidden layers           3
  -sz        Hidden layer sizes                128 128 64
  -a         Activation functions              relu relu relu
  -w_i       Weight initialization             xavier
  -s         Random seed                       42
  -ms        Model save path                   output/models/
  -cs        Config save path                  output/best_config.json
  -wa        WandB API key                     None
  -wp        WandB project name                DA6401_assn_1_mlp_trail_1
  --sweep    Enable sweep                      False

------------------------------------------------------------------------

# Results

Using the above configuration:

Accuracy: **0.9782**\
Precision: **0.9781839142720715**\
Recall: **0.9780153151902345**\
F1 Score: **0.9780400187995639**\
Balanced Accuracy: **0.9780153151902345**

------------------------------------------------------------------------

# Learning Objectives

-   Understand forward propagation
-   Implement backpropagation manually
-   Implement optimizers: SGD, Momentum, NAG, RMSProp, Adam, Nadam
-   Implement activation functions and derivatives
-   Train neural networks from scratch
-   Track experiments using Weights & Biases

------------------------------------------------------------------------

## Contact

For questions or issues, please contact the course teaching staff or
post on the course forum.
