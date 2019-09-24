import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """
    a = predictions
    if predictions.ndim == 1:
        a_mod = a - np.max(a)
        a_exp = np.exp(a_mod)
        return a_exp / np.sum(a_exp)
    else:
        a_mod = a - np.max(a, axis=1)[:, None]
        a_exp = np.exp(a_mod)
        return a_exp / np.sum(a_exp, axis=1)[:, None]


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """
    h = -np.log(probs)
    if isinstance(target_index, int):
        return h[target_index]
    else:
        if target_index.ndim == 1:
            target_index = target_index[:, None]
        return np.mean(np.take_along_axis(h, target_index, axis=1))


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # Softmax and cross-entropy loss
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    # Gradient
    dprediction = np.zeros(probs.shape)
    if isinstance(target_index, int):
        dprediction[target_index] = 1
        dprediction = probs - dprediction
    else:
        if target_index.ndim == 1:
            target_index = target_index[:, None]
        np.put_along_axis(dprediction, target_index, 1, axis=1)
        dprediction = (probs - dprediction) / dprediction.shape[0]

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        self.X = X
        return np.where(self.X > 0, self.X, 0)

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        return d_out * np.where(self.X > 0, 1, 0)

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X
        return self.X @ self.W.value + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        d_input = d_out @ self.W.value.T
        dW = self.X.T @ d_out
        dB = d_out.sum(axis=0)

        # Add gradients of W and B to their `grad` attribute
        self.W.grad += dW
        self.B.grad += dB

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
