"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""
import numpy as np


def get_optimizer(name, **kwargs):
    name = name.lower()
    if name == "sgd":
        return SGD(lr=kwargs.get("lr", 0.01))
    elif name == "momentum":
        return Momentum(lr=kwargs.get("lr", 0.01),momentum=kwargs.get("momentum", 0.9),)
    elif name == "nag":
        return NAG(lr=kwargs.get("lr", 0.01),momentum=kwargs.get("momentum", 0.9),)
    elif name == "rmsprop":
        return RMSProp(lr=kwargs.get("lr", 0.001),beta=kwargs.get("beta", 0.9),)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
        
##This can handle both mini-batch GD and SGD where it is updated for each sample
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    ##step function used for back propp
    def step(self, layers):
        for layer in layers:
            
            ## thetha = thetha - lr * delta_thetha => GD
            ##Theta can be either weights or biases
#            layer.W -= self.lr * layer.dW
#            layer.b -= self.lr * layer.db
            layer.W -= self.lr * layer.grad_W
            layer.b -= self.lr * layer.grad_b

class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            ##Initalizing velocity of theta with 0 at the start.
            if i not in self.v:
                ##self.v[i] = {"dW": np.zeros_like(layer.W),"db": np.zeros_like(layer.b),}
                self.v[i] = {"grad_W": np.zeros_like(layer.W),"grad_b": np.zeros_like(layer.b)}
            
            ##Introducting velocity for momentum of theta to achieve minima
            self.v[i]["grad_W"] = (self.momentum * self.v[i]["grad_W"]+ self.lr * layer.grad_W)
            self.v[i]["grad_b"] = (self.momentum * self.v[i]["grad_b"]+ self.lr * layer.grad_b)
            
            ##Applying GD after calculating the updated theta
            layer.W -= self.v[i]["grad_W"]
            layer.b -= self.v[i]["grad_b"]

##Nesterov Accelerated Gradient(NAG)
class NAG:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            ## Intializing velocity for theta of each layer
            if i not in self.v:
                self.v[i] = {"grad_W": np.zeros_like(layer.W),"grad_b": np.zeros_like(layer.b),}
            ##Saving previous velocity of theta for lookahead
            v_prev_W = self.v[i]["grad_W"]
            v_prev_b = self.v[i]["grad_b"]
            
            self.v[i]["grad_W"] = (self.momentum * self.v[i]["grad_W"] + self.lr * layer.grad_W)
            self.v[i]["grad_b"] = (self.momentum * self.v[i]["grad_b"]+ self.lr * layer.grad_b)

            #calculating gradient for future position => lookahead correction
            layer.W -= (-self.momentum * v_prev_W+ (1 + self.momentum) * self.v[i]["grad_W"])
            layer.b -= (-self.momentum * v_prev_b+ (1 + self.momentum) * self.v[i]["grad_b"])
 
class RMSProp:
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta ##decay rate for moving average (usually 0.9)
        self.eps = eps ##a small constant to avoid zero division error
        self.s = {}

    def step(self, layers):
        for i, layer in enumerate(layers):
            ## Intializing average squared gradients of the theta for each layer.
            if i not in self.s:
                self.s[i] = {"grad_W": np.zeros_like(layer.W),"grad_b": np.zeros_like(layer.b),}

            ##Calculating average of squared gradients 
            self.s[i]["grad_W"] = (self.beta * self.s[i]["grad_W"] + (1 - self.beta) * (layer.grad_W ** 2) )
            self.s[i]["grad_b"] = (self.beta * self.s[i]["grad_b"]+ (1 - self.beta) * (layer.grad_b ** 2))

            ##Adaptive update when s_t large => update would be small and vice versa
            layer.W -= self.lr * layer.grad_W / (np.sqrt(self.s[i]["grad_W"]) + self.eps)
            layer.b -= self.lr * layer.grad_b / (np.sqrt(self.s[i]["grad_b"]) + self.eps)