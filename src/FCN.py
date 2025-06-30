import numpy as np

class FCN:
    """ Fully Connected Layer (FCN).
    This layer performs a linear transformation
    followed by an optional bias addition.
    It is typically used in feedforward neural networks.    
    Attributes:
        W (np.ndarray): Weights of the layer, shape (out_features, in_features).
        b (np.ndarray): Biases of the layer, shape (out_features,).
        dW (np.ndarray): Gradient of the weights, shape (out_features, in_features).
        db (np.ndarray): Gradient of the biases, shape (out_features,).
        x (np.ndarray): Cached input for backpropagation.
    """
    def __init__(self, in_features, out_features):
        
        # Kaiming-ish initialisation for stability
        self.W = np.random.randn(out_features, in_features) * np.sqrt(2 / in_features)
        self.b = np.zeros(out_features)
        
        # Gradients will be written here after back-prop
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        # Saved activations for backward use
        self.x  = None

    def forward(self, x):
        """
        x : shape (batch, in_features)
        returns y : shape (batch, out_features)
        """
        self.x = x                        # cache for back-prop
        return x @ self.W.T + self.b      # y

    def backward(self, dL_dy):
        """
        dL_dy : gradient coming from next layer (shape = y)
        returns dL_dx : gradient to propagate further backward
        """
        # weight & bias gradients
        self.dW = dL_dy.T @ self.x        # (out, in)
        self.db = dL_dy.sum(0)            # (out,)
        # input gradient
        dL_dx = dL_dy @ self.W            # (batch, in)
        return dL_dx