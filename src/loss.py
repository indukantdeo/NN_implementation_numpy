import numpy as np

def mse_loss(pred, target):
    """
    Mean Squared Error (MSE) loss function.
    Args:
        pred (np.ndarray): Predicted values.
        target (np.ndarray): Target values.
    Returns:
        loss (float): Computed MSE loss.
        dL_dpred (np.ndarray): Gradient of the loss with respect to predictions.
    """
    diff = pred - target
    loss = np.mean(diff**2)
    dL_dpred = 2 * diff / diff.size      # derivative of MSE
    return loss, dL_dpred