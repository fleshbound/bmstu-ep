import abc
import math
from random import random

from numpy.random import exponential


class Distribution(abc.ABC):
    @abc.abstractmethod
    def generate(self):
        return NotImplementedError


class PoissonDistribution(Distribution):
    """
    Класс, представляющий распределение Пуассона.

    Attributes:
        lambd (float): Параметр lambda (среднее значение) распределения.

    Methods:
        generate(): Генерирует случайное значение Пуассоновской случайной величины.
    """
    lambda_: float

    def __init__(self, lambda_):
        """
        Конструктор класса PoissonDistribution.

        Args:
            lambd (float): Параметр lambda (среднее значение) распределения.
        """
        if lambda_ <= 0:
            raise ValueError("Lambda must be a positive value.")
        self.lambda_ = lambda_

    def generate(self):
        """
        Генерирует случайное значение Пуассоновской случайной величины.
        Использует алгоритм, основанный на равномерном распределении.

        Returns:
            int: Случайное значение Пуассоновской случайной величины.
        """
        L = math.exp(-self.lambda_)
        k = 0
        p = 1
        while p > L:
            k += 1
            p *= random()
        return k - 1
