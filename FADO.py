import numpy as np
from tensorflow.keras.losses import BinaryCrossentropy

class FADO:
    def __init__(self, alpha=1.0, beta=0.9, gamma=0.001, epsilon=1e-8, update_interval=1):
        """
        Initialize AWOA parameters.
        :param alpha: Controls the learning step size for weight updates.
        :param beta: Decay rate for moving average of squared biases.
        :param gamma: Learning rate for weight adjustments.
        :param epsilon: Small constant for numerical stability.
        :param update_interval: Number of epochs between weight updates.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.lambda_k = 0
        self.v_moving_avg = 0
        self.update_interval = update_interval

    def initialize_weights(self, n_majority, n_minority):
        total_samples = n_majority + n_minority
        w0 = n_minority / total_samples
        w1 = n_majority / total_samples
        return w0, w1

    def compute_bias(self, y_true, y_pred):
        pred_minority = np.mean(y_pred[y_true == 1] >= 0.5)
        true_minority = np.mean(y_true == 1)
        violation = pred_minority - true_minority
        return violation

    def update_weights(self, bias):
        self.v_moving_avg = self.beta * self.v_moving_avg + (1 - self.beta) * (bias ** 2)
        self.lambda_k -= self.gamma * bias / (np.sqrt(self.v_moving_avg) + self.epsilon)
        w1 = np.exp(self.alpha * self.lambda_k) / (np.exp(self.alpha * self.lambda_k) + np.exp(-self.alpha * self.lambda_k))
        w0 = 1 - w1
        return w0, w1

    def weighted_loss(self, y_true, y_pred, w0, w1):
        bce = BinaryCrossentropy()
        sample_weights = np.where(y_true == 1, w1, w0)
        return bce(y_true, y_pred, sample_weight=sample_weights)
