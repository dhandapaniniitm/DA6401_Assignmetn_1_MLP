"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np
from keras.datasets import fashion_mnist , mnist

## perform one hot encoding for the output class
def one_hot(y, num_classes=10):
    out = np.zeros((y.size, num_classes))
    out[np.arange(y.size), y] = 1
    return out

def preprocess(x, y, num_classes=10):
    # flatten the input images to a vector
    x = x.reshape(x.shape[0], -1)    
    # normalize the images such that it keeps it in the range of 0 to 1
    x = x / 255.0
    # one-hot encode label to match network output format
    y = one_hot(y, num_classes)
    return x, y

def load_data(dataset):
    ##Loading the data based on cli argument
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == "f_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'mnist' or 'f_mnist'.")
    
    #we are apply preprocessing or transformation to the dataset
    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)
    
    return x_train, y_train, x_test, y_test