import torch as ch
import logging
from torch.autograd import Variable


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LinearRegressionClassifier(object):
    """Objects of this class is a LinearRegressionClassifier.

   Examples
   --------
   >>> import torch as ch
   >>> x = ch.Tensor([[24], [50], [15], [55], [14], [12], [18], [62]])
   >>> y = ch.Tensor([[21.54945196], [47.46446305], [17.21865634], [52.789],
                    [16.1234], [12.789], [19.5649], [60.5793278]])
   >>> lr = omg.linear_regression.LinearRegressionClassifier(x,y)
   >>> lr.fit('Adam')
   >>> lr.predict(x)

   """

    def __init__(self, seed=None):
        """instantiate a linear regression object.

        """
        if seed:
            ch.manual_seed(seed)

    def fit(self, train_x, train_y, optimizer, iterations=1000):
        """ Finds suitable value of parameters which best fit the given output values.

        Parameters
        ----------
        train_x : list of (int/float torch.Tensor).
                  The input values on which linear regression is performed.
        train_y : list of (int/float torch.Tensor).
                  The target labels corresponding to the input values.
        optimizers : SGD, Adam, Analytical
                     Trains parameters to minimize the loss function.
        iterations : int
                     The number of steps to run an optimizer.

        """

        if optimizer == 'Analytical':
            cov_x_y = []
            var_x = []

            mean_x = ch.mean(train_x)
            mean_y = ch.mean(train_y)

            for x, y in zip(train_x, train_y):
                cov_x_y.append((x - mean_x) * (y - mean_y))
                var_x.append((x - mean_x) ** 2)

            self.beta = sum(cov_x_y) / sum(var_x)
            self.alpha = mean_y - self.beta * mean_x
            return

    def predict(self, test_data):
            """This function predicts the target label on the test dataset.

            Parameters
            ----------
            test_data : list of (int/float torch.Tensor).
                The test data on which the results are evaluated generally after each
                epoch.

            """
            self.test_data = Variable(ch.FloatTensor(test_data), requires_grad=False)
            for td in self.test_data:
                    predicted_value = float(self.beta) * float(td) + float(self.alpha)
                    logger.info("Predicted value for test_data {} slope {} and bias "
                                "{} is {}".format(td, self.alpha, self.beta, predicted_value))

    def save_model(self, file_path):
        """This function saves the model in a file for loading it in future.

        Parameters
        ----------
        file_path : str
            The path to file where the model should be saved.

        """
        ch.save(self.__dict__, file_path)
        return

    def load_model(self, file_path):
        """This function loads the saved model from a file.

        Parameters
        ----------
        file_path : str
            The path of file from where the model should be retrieved.

        """
        self.__dict__ = ch.load(file_path)
        return
