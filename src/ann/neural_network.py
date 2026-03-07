"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from .activations import Activation
from .neural_layer import DenseLayer
from .objective_functions import get_loss
from .optimizers import get_optimizer
from .metrics_util import evaluate_model_core

class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args, input_dim=None, output_dim=None):
        # --- Ensure input/output dims for autograder ---
        input_dim = input_dim or 784     # default for MNIST
        output_dim = output_dim or 10    # default number of classes

        self.lr = cli_args.learning_rate
        self.layers = []
        self.loss_name = cli_args.loss
        self.loss_fn = get_loss(cli_args.loss)
        self.optimizer_name = cli_args.optimizer
        self.weight_decay = cli_args.weight_decay
        self.optimizer = None  # will initialize after adding layers

        # Setup layer sizes
        layer_sizes = [input_dim] + cli_args.hidden_size + [output_dim]
        num_layers = len(layer_sizes) - 1  # number of Dense layers

        # --- Ensure activations list exists ---
        activations = getattr(cli_args, 'activations', None)
        if activations is None:
            activations = getattr(cli_args, 'activation', ["relu"])  # fallback
        if not isinstance(activations, list):
            activations = [activations]

        # Repeat single activation for hidden layers if needed
        if len(activations) == 1 and num_layers > 1:
            activations = activations * (num_layers - 1)

        # Add output layer activation if missing
        if len(activations) == num_layers - 1:
            activations.append("linear")

        # Final check
        assert len(activations) == num_layers, \
            f"Expected {num_layers} activations, got {len(activations)}"

        # Create layers
        for i in range(num_layers):
            self.layers.append(DenseLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                activation=activations[i],
                weight_init=cli_args.weight_init
            ))

        # Initialize optimizer
        self.optimizer = get_optimizer(cli_args.optimizer, lr=self.lr)
        
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns grad_Ws, grad_bs.
        Automatically converts y_true to one-hot if needed.
        """
        #print("Beforev y_true and y_pred")
        #print(y_true,y_pred)       
        # Check if y_true is one-hot
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            # Convert to one-hot
            num_classes = y_pred.shape[1]
            y_one_hot = np.zeros((y_true.shape[0], num_classes))
            y_indices = y_true.flatten().astype(int)
            y_one_hot[np.arange(y_true.shape[0]), y_indices] = 1
            y_true = y_one_hot

        #print("After y_true and y_pred")
        #print(y_true,y_pred)       
    
        grad_W_list = []
        grad_b_list = []
    
        # Backprop from output to input
        dA = self.loss_fn.backward(y_true, y_pred)
            
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            grad_W_list.insert(0, layer.grad_W.copy())
            grad_b_list.insert(0, layer.grad_b.copy())  # keep 2D, do NOT flatten        
        # Store lists directly, do NOT make np.array
        self.grad_W = grad_W_list
        self.grad_b = grad_b_list
        
        # Debug prints
        #print("Gradients shapes:")
        #for i in range(len(self.grad_b)):
        #    print(f"Layer {i} grad_b shape: {self.grad_b[i].shape}, grad_W shape: {self.grad_W[i].shape}")
    
        # Debug prints
        #print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        #print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
    
        return self.grad_W, self.grad_b
    
    # def backward(self, y_true, y_pred):
    #     print("y_true and y_pred")
    #     print(y_true,y_pred)
    #     grad_W_list = []    
    #     # Backprop from output to inputgrad_W_list = []
    #     grad_b_list = []
        
    #     dA = self.loss_fn.backward(y_true, y_pred)
        
    #     for layer in reversed(self.layers):
    #         dA = layer.backward(dA)
    #         grad_W_list.insert(0, layer.grad_W.copy())
    #         grad_b_list.insert(0, layer.grad_b.flatten())  # flatten to 1D
        
    #     self.grad_W = np.array(grad_W_list, dtype=object)
    #     self.grad_b = np.array(grad_b_list, dtype=object)
    #     print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
    #     print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        
    #    return self.grad_W, self.grad_b

    def update_weights(self):
        ###Check whether optimizer is present
        if self.optimizer is None:
            raise ValueError("Optimizer not set.")
        
        #Check weight decya value and apply it
        if self.weight_decay > 0:
            for layer in self.layers:
                #L2 regularization gradient as mentioned in assignement
                ##layer.dW += self.weight_decay * layer.W
                layer.grad_W += self.weight_decay * layer.W
    
        # -------- OPTIMIZER STEP ----------
        self.optimizer.step(self.layers)

    #def train(self, X_train, y_train, epochs=1, batch_size=32):
    def train(self, X_train, y_train, epochs=1, batch_size=32, X_val=None, y_val=None, wandb=None):
        history = {
            "train_losses": [],
            "val_losses": [],
            "train_accs": [],
            "val_accs": [],
        }
        
        n_samples = X_train.shape[0]
        #best_f1 = 0
        best_f1 = -np.inf 
        best_weights = None

        for epoch in range(epochs):
            # Shuffle dataset
            ##indices = np.arange(n_samples)
            ##np.random.shuffle(indices)
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            epoch_losses = []
                
            # Mini-batch gradient descent - Training loop
            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]

                # Forward + Backward
                y_pred = self.forward(X_batch)
                loss = self.compute_loss(y_batch, y_pred)
                epoch_losses.append(loss)
                self.backward(y_batch, y_pred)
                self.update_weights()

            # Epoch metrics
            avg_train_loss = np.mean(epoch_losses)
            train_results = self.evaluate(X_train, y_train)
            history["train_losses"].append(avg_train_loss)
            history["train_accs"].append(train_results["accuracy"])

            # Validation metrics
            if X_val is not None and y_val is not None:
                val_results = self.evaluate(X_val, y_val)
                history["val_losses"].append(val_results["loss"])
                history["val_accs"].append(val_results["accuracy"])

                # Track best F1
                if val_results["f1"] > best_f1:
                    best_f1 = val_results["f1"]
                    best_weights = self.get_weights()#.copy()

            # WandB logging
            if wandb is not None:
                log_data = {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "train/accuracy": train_results["accuracy"],
                    ##"optimizer/learning_rate": self.lr, 
                    "optimizer/learning_rate": self.optimizer.lr
                }
                if X_val is not None:
                    log_data.update({
                        "val/loss": val_results["loss"],
                        "val/accuracy": val_results["accuracy"],
                        "val/f1": val_results["f1"],
                    })
                    
                ##Logging Gradient norms
                total_grad_norm = 0
                for i, layer in enumerate(self.layers):
                    ##grad_norm = np.linalg.norm(layer.dW)
                    grad_norm = np.linalg.norm(layer.grad_W)
                    log_data[f"grad_norm/layer_{i}"] = grad_norm
                    total_grad_norm += grad_norm ** 2 
                log_data["grad_norm/total"] = np.sqrt(total_grad_norm)
    
                ## Logging Weight norms
                for i, layer in enumerate(self.layers):
                    weight_norm = np.linalg.norm(layer.W)
                    log_data[f"weight_norm/layer_{i}"] = weight_norm
    
                ##Wandb logging of Activation statistics
                for i, layer in enumerate(self.layers):

                    if hasattr(layer, "A") and layer.A is not None:
                        A = layer.A

                        log_data[f"activation_mean/layer_{i}"] = np.mean(A)
                        log_data[f"activation_std/layer_{i}"] = np.std(A)

                        ##dead neuron detection (ReLU only)
                        if layer.activation_name == "relu":
                            zero_frac = np.mean(A <= 0)
                            log_data[f"dead_neurons/layer_{i}"] = zero_frac
                            
                        elif layer.activation_name == "sigmoid":
                            sat_frac = np.mean((A < 0.05) | (A > 0.95))
                            log_data[f"saturation/layer_{i}"] = sat_frac
                        
                        elif layer.activation_name == "tanh":
                            sat_frac = np.mean(np.abs(A) > 0.95)
                            log_data[f"saturation/layer_{i}"] = sat_frac
                    
                wandb.log(log_data)
                

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_results['accuracy']:.4f}")

        last_weights = self.get_weights()
        
        if wandb is not None and X_val is not None:
            wandb.summary["best_val_f1"] = best_f1
            wandb.summary["final_train_acc"] = history["train_accs"][-1]
            wandb.summary["final_val_acc"] = history["val_accs"][-1]
        return history, best_weights, last_weights
    
    def evaluate(self, X, y):
        return evaluate_model_core(self, X, y)
    
    def compute_loss(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

