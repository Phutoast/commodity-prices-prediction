import numpy as np

class PerformanceMetric(object):
    """
    A collection of Losses 
        used for calculating performance of the data, 
        where all of the function arguments and returns are the same 
    
    Args:
        true: Testing Data Label
        pred: Prediction of the model (possibly including the uncertainty estimate)
    
    Return:
        perf: Performance of the model given its prediction and true label
    """

    def dummy_loss(self, true, pred):
        return -1

    def square_error(self, true, pred):
        """
        Square Error (se) calculated as:

            \sum^N_{i=1} (true[i] - pred[i])**2

        """
        pred = np.array(pred["mean"].to_numpy())
        return np.sum(np.square(true-pred))


