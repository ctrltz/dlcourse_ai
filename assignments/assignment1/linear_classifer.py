import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
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
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    h = -np.log(probs)
    if isinstance(target_index, int):
        return h[target_index]
    else:
        return np.mean(np.take_along_axis(h, target_index, axis=1))


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # Softmax and cross-entropy loss
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)

    # Gradient
    dprediction = np.zeros(probs.shape)
    if isinstance(target_index, int):
        dprediction[target_index] = 1
        dprediction = probs - dprediction
    else:
        np.put_along_axis(dprediction, target_index, 1, axis=1)
        dprediction = (probs - dprediction) / dprediction.shape[0]

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)

    if target_index.ndim == 1:
        target_index = target_index[:, None]
    loss, dpredictions = softmax_with_cross_entropy(predictions, target_index)
    dW = np.dot(X.T, dpredictions)
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1, verbose=True, val_X=None, val_y=None):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        validation = val_X is not None and val_y is not None

        loss_history = []
        val_loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # Start training epoch
            loss_during_epoch = np.zeros((len(batches_indices),))
            for i, batch_idx in enumerate(batches_indices):
                # Implement generating batches from indices
                batch = X[batch_idx, :]
                batch_target = y[batch_idx]

                # Compute loss and gradients
                cross_entropy_loss, cross_entropy_grad = linear_softmax(batch, self.W, batch_target)
                reg_loss, reg_grad = l2_regularization(self.W, reg)

                loss = cross_entropy_loss + reg_loss
                grad = cross_entropy_grad + reg_grad

                # Apply gradient to weights using learning rate
                self.W = self.W - learning_rate * grad

                loss_during_epoch[i] = loss

            # End training epoch - all batches visited
            loss = np.mean(loss_during_epoch)
            if verbose:
                print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

            if validation:
                val_loss, _ = linear_softmax(val_X, self.W, val_y)
                val_loss_history.append(val_loss) 

        if validation:
            return loss_history, val_loss_history
        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        probs = softmax(np.dot(X, self.W))
        y_pred = np.argmax(probs, axis=1).astype(int)

        return y_pred



                
                                                          

            

                
