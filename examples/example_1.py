import numpy as np
import matplotlib.pyplot as plt
import sys
import tqdm
import time
sys.path.append('../src/')

from FCN import FCN
from loss import mse_loss

# ----------------------------------------------------------------------
# 1. Tiny example: linear regression y = Wx + b on random data
# ----------------------------------------------------------------------
np.random.seed(0)
N, Din, Dout = 4, 3, 2
x  = np.random.randn(N, Din)
true_W = np.array([[1.5, -2.0, 0.3],
                   [0.7,  0.2, 1.1]])
true_b = np.array([0.5, -1.2])
y_true = x @ true_W.T + true_b

# ----------------------------------------------------------------------
# 2. Define a linear layer
# ----------------------------------------------------------------------
layer = FCN(Din, Dout)
lr = 1e-1
losses = []
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b-')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss vs Epoch')
ax.set_xlim(0, 200)
ax.set_ylim(0, 1)

# ----------------------------------------------------------------------
# 3. Training loop (plain SGD)
# ----------------------------------------------------------------------
plt.show()
plt.pause(0.1)  # Pause to allow the plot to render

for epoch in tqdm.tqdm(range(200)):
    # FORWARD
    y_pred = layer.forward(x)
    loss, dL_dy = mse_loss(y_pred, y_true)

    # BACKWARD
    layer.backward(dL_dy)

    # SGD update
    layer.W -= lr * layer.dW
    layer.b -= lr * layer.db

    losses.append(loss)
    line.set_data(range(len(losses)), losses)
    ax.set_ylim(0, max(losses) * 1.1)

    # ----------- These lines are important! -------------
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    # Optional for smoothness:
    time.sleep(0.01)
    # ----------------------------------------------------

    if epoch % 40 == 0:
        print(f"epoch {epoch:3d}  loss {loss:.4f}")

plt.ioff()
plt.show()
