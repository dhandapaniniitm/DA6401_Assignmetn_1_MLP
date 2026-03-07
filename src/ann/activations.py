"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np

class Activation:
    @staticmethod
    def forward(name, Z):
        ##keep positive values, set negative ones to 0
        if name == "relu":
            return np.maximum(0, Z)
        ##squashes values between 0 and 1
        elif name == "sigmoid":
            return 1 / (1 + np.exp(-Z))
        ##scale values between -1 and 1
        elif name == "tanh":
            return np.tanh(Z)
        ##convert outputs into probability distribution within 0 to 1
        elif name == "softmax":
            #Numerical Stability to avoid overflow issue with Z - RuntimeWarning: overflow encountered in exp
            Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
            exp_Z = np.exp(Z_shifted)
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        ##simply returns the input
        elif name == "linear":
            return Z
        else:
            raise ValueError(f"Unsupported activation: {name}")
            
    @staticmethod
    def backward(name, Z, dA):
        ##Almost predefined ones based on activation
        if name == "relu":
            dZ = dA * (Z > 0)
        elif name == "sigmoid":
            sig = 1 / (1 + np.exp(-Z))
            dZ = dA * sig * (1 - sig)
        elif name == "tanh":
            t = np.tanh(Z)
            dZ = dA * (1 - t ** 2)
        elif name == "softmax":
            # Usually handled in CE loss, but safe fallback
            dZ = dA
        elif name == "linear":
            dZ = dA
        else:
            raise ValueError(f"Unsupported activation: {name}")
        return dZ