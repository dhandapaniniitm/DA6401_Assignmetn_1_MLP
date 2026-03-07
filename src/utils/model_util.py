"""
function to load and save model weights to numpy
"""

import os
import numpy as np

##This was created before TA shared load and set weights
def save_model(model,model_save_path):
    os.makedirs(os.path.dirname(model_save_path),exist_ok=True)
    
    Weightss = {}
    
    for i ,layer in enumerate(model.layers):
        Weightss[f"w{i}"] = layer.w
        Weightss[f"b{i}"] = layer.b
    np.save(model_save_path, Weightss,allow_pickle=True)
    #print(f"Model has been saved to the given location: {model_save_path}")

def load_model(model, path):
    weights = np.load(path, allow_pickle=True).item()
    for i, layer in enumerate(model.layers):
        layer.w = weights[f"w{i}"]
        layer.b = weights[f"b{i}"]
        
