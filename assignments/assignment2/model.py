import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg

        # Network architecture
        self.layers = {'Linear1': FullyConnectedLayer(n_input, hidden_layer_size),
                       'ReLU1': ReLULayer(),
                       'Linear2': FullyConnectedLayer(hidden_layer_size, n_output),
                       'ReLU2': ReLULayer()}

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        
        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        layers = list(self.layers.values())
        for layer in layers:
            X = layer.forward(X)
        loss, grad = softmax_with_cross_entropy(X, y)
        for layer in layers[::-1]:
            grad = layer.backward(grad)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}

        for layer_label, layer in self.layers.items():
            for param_label, param in layer.params().items():
                result[f'{layer_label}_{param_label}'] = param

        return result
