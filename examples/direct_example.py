import numpy as np

class Linear:
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


def mse_loss(pred, target):
    diff = pred - target
    loss = np.mean(diff**2)
    dL_dpred = 2 * diff / diff.size      # derivative of MSE
    return loss, dL_dpred

# Tiny example: linear regression y = Wx + b on random data
np.random.seed(0)
N, Din, Dout = 4, 3, 2
x  = np.random.randn(N, Din)
true_W = np.array([[1.5, -2.0, 0.3],
                   [0.7,  0.2, 1.1]])
true_b = np.array([0.5, -1.2])
y_true = x @ true_W.T + true_b

layer = Linear(Din, Dout)
lr = 1e-1
for epoch in range(200):
    # FORWARD
    y_pred = layer.forward(x)
    loss, dL_dy = mse_loss(y_pred, y_true)

    # BACKWARD
    layer.backward(dL_dy)

    # SGD update
    layer.W -= lr * layer.dW
    layer.b -= lr * layer.db

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {loss:.4f}")