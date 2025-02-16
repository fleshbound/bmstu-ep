import abc

from numpy.random import exponential


class Distribution(abc.ABC):
    @abc.abstractmethod
    def generate(self):
        return NotImplementedError


class ExponentialDistribution(Distribution):
    lambda_: float

    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def generate(self):
        return exponential(self.lambda_)
