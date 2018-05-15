import numpy as np
from utils.loss_function import LossFunction
from utils.activation import Activation


class Perceptron(object):
        """This is a single layer neural network classifier.
        Parameters:
        -----------
        train_x: input numpy array
            e.g., train_x = np.array([0.3]).
        train_x: actual output numpy array
            e.g., train_y = np.array([0.3]).
        hidden_neurons: integer
            number of neurons/Perceptron
        """

        def __init__(self, train_x, train_y, hidden_neurons=1, bias=0):
            self.train_x = train_x
            self.train_y = train_y

            self.hidden_neurons = hidden_neurons
            self.input_nodes = np.shape(train_x)[0]
            self.output_nodes = np.shape(train_y)[0]
            self.bias = bias

            # seed for the fixed random values
            np.random.seed(1)
            self.W_in = np.random.normal(0.0, 0.1, (self.input_nodes, self.hidden_neurons))
            print(self.W_in)

        def forward_prop(self):
            self.X = np.dot(self.train_x.T, self.W_in) + self.bias
            self.A = Activation.sigmoid(self.X)
            self.E = LossFunction.L2_loss(np.array(self.A), np.array(self.train_y))

#       backpropagation
        def back_prop(self):
            self.loss = self.E
            self.dloss = - (self.train_y - self.A)*self.A*(1-self.A)*self.train_x

#       optimization
        """Optimizer function for tuning the weight by the gradient of the error.
        Parameters:
        -----------
        lr: float
        learning rate
        """
        def optimization(self, lr=0.01):
                if self.A - self.train_y > 0:
                    self.W_in = self.W_in + lr * self.dloss
                else:
                    self.W_in = self.W_in - lr * self.dloss

#       train the network
        """Trains the neural network.
        Parameters:
        -----------
        n_iterations: int
        number of times to update the weight
        """
        def train(self, n_iterations=10):
            for i in range(n_iterations):
                # forward pass
                self.forward_prop()
                self.back_prop()
                self.optimization()
            print(self.W_in, "trained Weight")
            print(self.A, "Predicted Output")
