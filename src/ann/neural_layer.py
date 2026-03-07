"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np
from .activations import Activation

class DenseLayer:
    def __init__(self, in_features, out_features,
                 activation="relu",
                 weight_init="xavier"):
        self.in_features = in_features
        self.out_features = out_features
        self.activation_name = activation
        # Weight initialization
        if weight_init == "xavier":
            limit = np.sqrt(1.0 / in_features)
            self.W = np.random.randn(in_features, out_features) * limit
        elif weight_init == "he":
            limit = np.sqrt(2.0 / in_features)
            self.W = np.random.randn(in_features, out_features) * limit
        else:
            self.W = np.random.randn(in_features, out_features) * 0.01

        self.b = np.zeros((1, out_features))
        ##cache to store the x/Z and its previous value
        self.Z = None
        ##Input to the model or activation output from previous layer
        self.A_prev = None

        # cache to save the gradients valuesss
        self.grad_W = None
        self.grad_b = None
    
    def forward(self, A_prev):
        ##Z = x(a_prev) . W + b
        self.A_prev = A_prev
        self.Z = A_prev @ self.W + self.b
    
        if self.activation_name == "linear":
            return self.Z
        ##apllying actyivation non-linear to the A=F(Z)
        return Activation.forward(self.activation_name, self.Z)
    
    
    def backward(self, dA):
    
        if self.activation_name == "linear":
            dZ = dA
        else:
            ##backward pass logic , chainrule
            dZ = Activation.backward(self.activation_name, self.Z, dA)
    
        m = self.A_prev.shape[0]
        ##Updating weights gradients
        
        self.grad_W = (self.A_prev.T @ dZ) / m
        self.grad_b = np.sum(dZ, axis=0, keepdims=True) / m
    
        dA_prev = dZ @ self.W.T
        return dA_prev