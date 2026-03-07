"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np

def get_loss(name):
    ##used for multi class
    if name in ["ce", "cross_entropy"]:
        return CrossEntropy()
    ##used for binary class
    elif name == "bce":
        return BinaryCrossEntropy()
    ##Mainly used for regression , we are experimenting it for classification without using softmax.
    elif name == "rmse":
        return RMSE()
    elif name in ["mse","mean_squared_error"]:
        return MSE()
    else:
        raise ValueError(f" Un--supported loss name: {name}")

class CrossEntropy:
    def forward(self, y_true, logits):
        m = y_true.shape[0]
        # Numerical stability
        z = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z)
        self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)  # store for backward
        # Compute loss
        loss = -np.sum(y_true * np.log(self.probs + 1e-15)) / m
        return loss

    def backward(self, y_true, logits=None):
        """
        Compute gradient using stored softmax probabilities.
        logits argument is optional; probs must exist from forward pass.
        """
        if not hasattr(self, "probs"):
            # compute probs if backward called first (rare, but autograder may do this)
            z = logits - np.max(logits, axis=1, keepdims=True)
            exp_z = np.exp(z)
            self.probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        m = y_true.shape[0]
        return (self.probs - y_true) / m

    # optional: allow __call__ as shorthand
    def __call__(self, y_true, logits):
        return self.forward(y_true, logits)

class BinaryCrossEntropy:
    ##one class object is used for forward and backward
    ##Forward pass -> equivalent to loss_fn.foward
    def __call__(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) # Numerical stability => log(0) = -intifinity => so we keep the range between eps to 1-eps
        
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )
        return loss
    def backward(self, y_true, y_pred):
        m = y_true.shape[0]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) #Numerical stability => prevents division by zero from loss fn's denominator
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * m)

class RMSE:
    ##one class object is used for forward and backward
    ##Forward pass -> equivalent to loss_fn.foward
    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean((y_pred - y_true) ** 2))
    def backward(self, y_true, y_pred):
        m = y_true.size
        diff = y_pred - y_true
        rmse = np.sqrt(np.mean(diff ** 2)) + 1e-15  # Adding small epsilon of 1e-15 to avoid division by zero.
        return diff / (m * rmse)
    
class MSE:
    ##one class object is used for forward and backward
    ##Forward passses
    def __call__(self, y_true, y_pred):
        return np.mean((y_pred - y_true) ** 2)

    ##Backward pas , derivative
    def backward(self, y_true, y_pred):
        m = y_true.size   #total number of elements used in the batch
        return 2 * (y_pred - y_true) / m
    
"""     
This one expects probability from model -after performing softmax at model end  
class CrossEntropy:
    ##one class object is used for forward and backward
    ##Forward pass -> equivalent to loss_fn.foward => predictions -> loss
    def __call__(self, y_true, y_pred):
        # TO get number of samples for averaging
        m = y_true.shape[0]
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps) # Numerical stability => log(0) = -intifinity => so we keep the range between eps to 1-eps
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss
    ##Backward pass => loss -> gradients
    ## y_pred - y_true => y_hat - y
    def backward(self, y_true, y_pred): #here the gradient doesn't have denominator,so No need for numerical stabiloty using clip
        m = y_true.shape[0]
        return (y_pred - y_true) / m 
"""