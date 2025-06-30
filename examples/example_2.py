import numpy as np
import matplotlib.pyplot as plt
import sys
import tqdm
import time
sys.path.append('../src/')  # Adjust path to your src directory
from FCN import FCN
from loss import mse_loss
from activation import ReLU

# ----------------------------------------------------------------------
# 1.  Tiny synthetic regression dataset
# ----------------------------------------------------------------------
np.random.seed(0)
N, Din, Dout = 64, 10, 1          # 64 samples, 10-D input, 1-D output
X  = np.random.randn(N, Din)
true_W = np.random.randn(Dout, Din)
true_b = np.random.randn(Dout)
y_true = X @ true_W.T + true_b    # linear ground truth with noise
y_true += 0.05 * np.random.randn(*y_true.shape)

# ----------------------------------------------------------------------
# 2.  Define a 3-layer fully-connected network: 10 → 32 → 16 → 1
# ----------------------------------------------------------------------
fc1, relu1 = FCN(Din, 32), ReLU()
fc2, relu2 = FCN(32, 16), ReLU()
fc3        = FCN(16, Dout)

def forward_pass(x):
    out = fc1.forward(x)
    out = relu1.forward(out)
    out = fc2.forward(out)
    out = relu2.forward(out)
    out = fc3.forward(out)
    return out                                    # final prediction

def backward_pass(grad_out):
    grad = fc3.backward(grad_out)
    grad = relu2.backward(grad)
    grad = fc2.backward(grad)
    grad = relu1.backward(grad)
    fc1.backward(grad)                            # returns grad to inputs (unused)

# ----------------------------------------------------------------------
# 3.  Training loop (plain SGD)
# ----------------------------------------------------------------------
lr = 1e-2
epochs = 1000
for ep in range(1, epochs + 1):
    # -------- forward --------
    y_pred = forward_pass(X)
    loss, dL_dy = mse_loss(y_pred, y_true)

    # -------- backward -------
    backward_pass(dL_dy)

    # -------- SGD update -----
    for layer in (fc1, fc2, fc3):
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db

    # -------- monitor --------
    if ep % 100 == 0 or ep == 1:
        print(f"epoch {ep:4d}  |  loss {loss:.6f}")

print("\nfinished training")