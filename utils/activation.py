import numpy as np
import logging


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Activation(object):

    @staticmethod
    def relu_activation(x):
        return max(0, x)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return (1 - np.exp(x)) / (1 + np.exp(x))

    @staticmethod
    def softmax(x):
        x_new = [np.exp(i) for i in x]
        sum_x_new = sum(x_new)
        return [sum_x_new / (i) for i in x_new]

    @staticmethod
    def derivate_relu(x):
        if x > 0:
            return 1
        else:
            return 0

    @staticmethod
    def derivate_sigmoid(x):
        return (Activation.sigmoid(x)) * (1 - Activation.sigmoid(x))

    @staticmethod
    def derivate_tanh(x):
        return - np.exp(x) / (1 + np.exp(x)) ** 2
