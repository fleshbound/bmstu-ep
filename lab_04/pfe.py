from typing import List

import numpy as np

from lab_03.distribution import PoissonDistribution
from lab_03.generator import Generator
from lab_03.processor import Processor
from lab_03.request import REQUEST_TYPE_ONE, REQUEST_TYPE_TWO
from lab_03.simulator import Simulator


class Pfe:
    factors_number: int = 4
    N: int = 2 ** factors_number
    dfe_num_of_experiments: int
    ockp_num_of_experiments: int
    linear_coef_num: int = factors_number + 1
    nonlinear_coef_num: int = N
    test_num: int = 100

    lambda1_min: float
    lambda1_max: float
    lambda2_min: float
    lambda2_max: float
    mu1_min: float
    mu1_max: float
    mu2_min: float
    mu2_max: float

    ockp_coef_a: float
    ockp_coef_alpha: float

    plan_matrix: np.ndarray
    dfe_plan_matrix: np.ndarray
    ockp_plan_matrix: np.ndarray
    b_coefs_vector_1: np.ndarray
    b_coefs_vector_1_dfe: np.ndarray
    b_coefs_vector_1_natural: np.ndarray
    b_coefs_vector_1_natural_dfe: np.ndarray
    linear_y_approx_1: np.ndarray
    linear_y_approx_1_dfe: np.ndarray
    nonlinear_y_approx_1: np.ndarray
    nonlinear_y_approx_1_dfe: np.ndarray
    y_minus_linear_y_approx_1: np.ndarray
    y_minus_linear_y_approx_square_1: np.ndarray
    y_minus_nonlinear_y_approx_1: np.ndarray
    y_minus_nonlinear_y_approx_square_1: np.ndarray
    y_minus_linear_y_approx_1_dfe: np.ndarray
    y_minus_linear_y_approx_square_1_dfe: np.ndarray
    y_minus_nonlinear_y_approx_1_dfe: np.ndarray
    y_minus_nonlinear_y_approx_square_1_dfe: np.ndarray
    b_coefs_vector_2: np.ndarray
    b_coefs_vector_2_dfe: np.ndarray
    b_coefs_vector_2_natural: np.ndarray
    b_coefs_vector_2_natural_dfe: np.ndarray
    linear_y_approx_2: np.ndarray
    linear_y_approx_2_dfe: np.ndarray
    nonlinear_y_approx_2: np.ndarray
    nonlinear_y_approx_2_dfe: np.ndarray
    y_minus_linear_y_approx_2: np.ndarray
    y_minus_linear_y_approx_square_2: np.ndarray
    y_minus_nonlinear_y_approx_2: np.ndarray
    y_minus_nonlinear_y_approx_square_2: np.ndarray
    y_minus_linear_y_approx_2_dfe: np.ndarray
    y_minus_linear_y_approx_square_2_dfe: np.ndarray
    y_minus_nonlinear_y_approx_2_dfe: np.ndarray
    y_minus_nonlinear_y_approx_square_2_dfe: np.ndarray
    
    result_matrix: np.ndarray
    dfe_result_matrix: np.ndarray
    y_experiment_results_2: np.ndarray
    y_experiment_results_1: np.ndarray

    sum_r_linear_2: float
    sum_r_nonlinear_2: float

    sum_r_linear_1: float
    sum_r_nonlinear_1: float

    sum_r_linear_2_dfe: float
    sum_r_nonlinear_2_dfe: float

    sum_r_linear_1_dfe: float
    sum_r_nonlinear_1_dfe: float

    num_requests: int

    # Коэффициенты для нормированных факторов (ОЦКП)
    b_coefs_vector_1_ockp: np.ndarray  # для первого отклика (y1)
    b_coefs_vector_2_ockp: np.ndarray  # для второго отклика (y2)

    # Натуральные коэффициенты (ОЦКП)
    b_coefs_vector_1_natural_ockp: np.ndarray
    b_coefs_vector_2_natural_ockp: np.ndarray

    # Аппроксимации и ошибки для ОЦКП
    nonlinear_y_approx_1_ockp: np.ndarray
    nonlinear_y_approx_2_ockp: np.ndarray
    y_minus_nonlinear_y_approx_1_ockp: np.ndarray
    y_minus_nonlinear_y_approx_square_1_ockp: np.ndarray
    y_minus_nonlinear_y_approx_2_ockp: np.ndarray
    y_minus_nonlinear_y_approx_square_2_ockp: np.ndarray

    # Матрица плана и результатов ОЦКП
    ockp_result_matrix: np.ndarray

    # Для анализа адекватности модели ОЦКП
    sum_r_nonlinear_1_ockp: float
    sum_r_nonlinear_2_ockp: float
    y_minus_linear_y_approx_1_ockp: float
    y_minus_linear_y_approx_2_ockp: float

    def __init__(self, lambda1_min: float, lambda1_max: float, lambda2_min: float, lambda2_max: float,
                 mu1_min: float, mu1_max: float, mu2_min: float, mu2_max: float, num_requests: int):
        self.lambda1_min = lambda1_min   # x1_min_natural
        self.lambda1_max = lambda1_max   # x1_max_natural
        self.lambda2_min = lambda2_min   # x2_min_natural
        self.lambda2_max = lambda2_max   # x2_max_natural
        self.mu1_min = mu1_min   # x3_min_natural
        self.mu1_max = mu1_max   # x3_max_natural
        self.mu2_min = mu2_min   # x4_min_natural
        self.mu2_max = mu2_max   # x4_max_natural
        self.factors_max_natural = [lambda1_max, lambda2_max, mu1_max, mu2_max]
        self.factors_min_natural = [lambda1_min, lambda2_min, mu1_min, mu2_min]
        self.num_requests = num_requests

    def normal_to_natural(self, x_norm, x_nat_min, x_nat_max):
        I = (x_nat_max - x_nat_min) / 2  # интервал варьирования фактора
        return x_norm * I + (x_nat_max + x_nat_min) / 2

    def create_plan_matrix(self):
        self.plan_matrix = np.array([[1] * self.N for _ in range(self.N)])

        for j in range(1, self.factors_number + 1):
            period = pow(2, self.factors_number - j)
            for start_minus in range(period, self.N, 2 * period):
                for i in range(start_minus, start_minus + period):
                    self.plan_matrix[i][j] = -1

        self.plan_matrix *= -1
        self.plan_matrix[:, 0] *= -1
        # pairs
        self.plan_matrix[:, 5] = self.plan_matrix[:, 1] * self.plan_matrix[:, 2]
        self.plan_matrix[:, 6] = self.plan_matrix[:, 1] * self.plan_matrix[:, 3]
        self.plan_matrix[:, 7] = self.plan_matrix[:, 1] * self.plan_matrix[:, 4]
        self.plan_matrix[:, 8] = self.plan_matrix[:, 2] * self.plan_matrix[:, 3]
        self.plan_matrix[:, 9] = self.plan_matrix[:, 2] * self.plan_matrix[:, 4]
        self.plan_matrix[:, 10] = self.plan_matrix[:, 3] * self.plan_matrix[:, 4]

        # triplets
        self.plan_matrix[:, 11] = self.plan_matrix[:, 1] * self.plan_matrix[:, 2] * self.plan_matrix[:, 3]
        self.plan_matrix[:, 12] = self.plan_matrix[:, 1] * self.plan_matrix[:, 2] * self.plan_matrix[:, 4]
        self.plan_matrix[:, 13] = self.plan_matrix[:, 1] * self.plan_matrix[:, 3] * self.plan_matrix[:, 4]
        self.plan_matrix[:, 14] = self.plan_matrix[:, 2] * self.plan_matrix[:, 3] * self.plan_matrix[:, 4]

        # fours
        self.plan_matrix[:, 15] = self.plan_matrix[:, 1] * self.plan_matrix[:, 2] * self.plan_matrix[:, 3] * self.plan_matrix[:, 4]

    def calculate_b_coefs(self):
        self.b_coefs_vector_1 = np.array([
            np.mean((self.plan_matrix[:, j] * self.y_experiment_results_1)) for j in range(self.N)
        ])
        self.b_coefs_vector_2 = np.array([
            np.mean((self.plan_matrix[:, j] * self.y_experiment_results_2)) for j in range(self.N)
        ])

        return self.b_coefs_vector_1, self.b_coefs_vector_2

    def calculate_b_coefs_dfe(self):
        self.b_coefs_vector_1_dfe = np.array([
            np.mean((self.dfe_plan_matrix[:, j] * self.y_experiment_results_1)) for j in range(self.dfe_num_of_experiments + 1)
        ])

        self.b_coefs_vector_2_dfe = np.array([
            np.mean((self.dfe_plan_matrix[:, j] * self.y_experiment_results_2)) for j in range(self.dfe_num_of_experiments + 1)
        ])

        return self.b_coefs_vector_1_dfe, self.b_coefs_vector_2_dfe

    def create_dfe_plan_matrix(self):
        self.factors_number = 4
        self.dfe_num_of_experiments = 8  # 2^(4-1) для полуреплики
        # Инициализация матрицы: N строк, 9 столбцов (x0, x1, x2, x3, x4, x1x2, x2x3, x1x3, x1x2x3)
        self.dfe_plan_matrix = np.ones((self.dfe_num_of_experiments, 9), dtype=int)

        # Заполнение x1, x2, x3 как для полного ФЭ первых трёх факторов
        for j in range(1, self.factors_number):  # Столбцы 1, 2, 3 (x1, x2, x3)
            period = 2 ** (3 - (j - 1))  # Период смены знака
            for start_minus in range(period, self.dfe_num_of_experiments, 2 * period):
                for i in range(start_minus, start_minus + period):
                    self.dfe_plan_matrix[i][j] = -1

        # x4 = x1 * x2 * x3 (генератор)
        self.dfe_plan_matrix[:, 4] = (self.dfe_plan_matrix[:, 1] * 
                                     self.dfe_plan_matrix[:, 2] * 
                                     self.dfe_plan_matrix[:, 3])

        # Парные взаимодействия
        self.dfe_plan_matrix[:, 5] = self.dfe_plan_matrix[:, 1] * self.dfe_plan_matrix[:, 2]  # x1x2
        self.dfe_plan_matrix[:, 6] = self.dfe_plan_matrix[:, 2] * self.dfe_plan_matrix[:, 3]  # x2x3
        self.dfe_plan_matrix[:, 7] = self.dfe_plan_matrix[:, 1] * self.dfe_plan_matrix[:, 3]  # x1x3

        # Тройное взаимодействие
        self.dfe_plan_matrix[:, 8] = (self.dfe_plan_matrix[:, 1] * 
                                     self.dfe_plan_matrix[:, 2] * 
                                     self.dfe_plan_matrix[:, 3])  # x1x2x3

        return self.dfe_plan_matrix

    def create_ockp_plan_matrix(self):
        self.factors_number = 4
        self.ockp_num_of_experiments = 15  # 2^(4-1) ядро + 6 звёздных + 1 центральная
        # Инициализация матрицы: 15 строк, 11 столбцов
        self.ockp_plan_matrix = np.ones((self.ockp_num_of_experiments, 11))

        self.ockp_coef_alpha = 1.215  # alpha = sqrt(sqrt(N0*N)/2-N0/2)
        self.ockp_coef_a = 0.8  # a = sqrt(N0/N)

        # Ядро плана (первые 8 строк) - ДФЭ 2^(4-1)
        for j in range(1, 4):  # Столбцы x1, x2, x3
            period = 2 ** (3 - (j - 1))
            for start_minus in range(period, 8, 2 * period):
                for i in range(start_minus, start_minus + period):
                    self.ockp_plan_matrix[i][j] = -1

        # Генератор x4 = x1*x2*x3
        self.ockp_plan_matrix[:8, 4] = (self.ockp_plan_matrix[:8, 1] *
                                        self.ockp_plan_matrix[:8, 2] *
                                        self.ockp_plan_matrix[:8, 3])

        # Парные взаимодействия для ядра
        self.ockp_plan_matrix[:8, 5] = self.ockp_plan_matrix[:8, 1] * self.ockp_plan_matrix[:8, 2]  # x1x2
        self.ockp_plan_matrix[:8, 6] = self.ockp_plan_matrix[:8, 2] * self.ockp_plan_matrix[:8, 3]  # x2x3
        self.ockp_plan_matrix[:8, 7] = self.ockp_plan_matrix[:8, 1] * self.ockp_plan_matrix[:8, 3]  # x1x3

        # Квадраты факторов для ядра (x1², x2², x3²)
        for i in range(8):
            self.ockp_plan_matrix[i, 8] = self.ockp_plan_matrix[i, 1] ** 2 - self.ockp_coef_a
            self.ockp_plan_matrix[i, 9] = self.ockp_plan_matrix[i, 2] ** 2 - self.ockp_coef_a
            self.ockp_plan_matrix[i, 10] = self.ockp_plan_matrix[i, 3] ** 2 - self.ockp_coef_a

        # Звёздные точки (строки 8-13)
        star_points = [
            (8, -self.ockp_coef_alpha, 0, 0),  # x1 = -α
            (9, self.ockp_coef_alpha, 0, 0),  # x1 = +α
            (10, 0, -self.ockp_coef_alpha, 0),  # x2 = -α
            (11, 0, self.ockp_coef_alpha, 0),  # x2 = +α
            (12, 0, 0, -self.ockp_coef_alpha),  # x3 = -α
            (13, 0, 0, self.ockp_coef_alpha)  # x3 = +α
        ]

        for row, x1, x2, x3 in star_points:
            self.ockp_plan_matrix[row, 1:4] = [x1, x2, x3]
            # Квадраты для звёздных точек
            self.ockp_plan_matrix[row, 8] = x1 ** 2 - self.ockp_coef_a
            self.ockp_plan_matrix[row, 9] = x2 ** 2 - self.ockp_coef_a
            self.ockp_plan_matrix[row, 10] = x3 ** 2 - self.ockp_coef_a

        # Центральная точка (строка 14)
        # Все x1-x3 = 0, x0 = 1, квадраты = -a
        self.ockp_plan_matrix[14, 8:11] = [-self.ockp_coef_a] * 3

        # Обнуление взаимодействий для звёздных и центральной точек (строки 8-14)
        self.ockp_plan_matrix[8:14, 4:8] = 0  # x4, x1x2, x2x3, x1x3

        self.ockp_plan_matrix[14, 1:8] = 0

    def calculate_linear_y_approx_dfe(self):
        self.linear_y_approx_1_dfe = np.array([
            sum(self.dfe_plan_matrix[i, :self.linear_coef_num] * self.b_coefs_vector_1_dfe[:self.linear_coef_num])
            for i in range(self.dfe_num_of_experiments)
        ])
        self.linear_y_approx_2_dfe = np.array([
            sum(self.dfe_plan_matrix[i, :self.linear_coef_num] * self.b_coefs_vector_2_dfe[:self.linear_coef_num])
            for i in range(self.dfe_num_of_experiments)
        ])

        return self.linear_y_approx_1_dfe, self.linear_y_approx_2_dfe

    def calculate_nonlinear_y_approx_dfe(self):
        self.nonlinear_y_approx_1_dfe = np.array([
            sum(self.dfe_plan_matrix[i, :] * self.b_coefs_vector_1_dfe[:])
            for i in range(self.dfe_num_of_experiments)
        ])
        self.nonlinear_y_approx_2_dfe = np.array([
            sum(self.dfe_plan_matrix[i, :] * self.b_coefs_vector_2_dfe[:])
            for i in range(self.dfe_num_of_experiments)
        ])

        return self.nonlinear_y_approx_1_dfe, self.nonlinear_y_approx_2_dfe

    def calculate_result_matrix_dfe(self, experiment_results_1: np.ndarray, experiment_results_2: np.ndarray):
        self.y_experiment_results_1 = experiment_results_1
        self.y_experiment_results_2 = experiment_results_2

        self.calculate_b_coefs_dfe()
        self.calculate_linear_y_approx_dfe()
        self.calculate_nonlinear_y_approx_dfe()

        self.y_minus_linear_y_approx_1_dfe = self.y_experiment_results_1 - self.linear_y_approx_1_dfe
        self.y_minus_linear_y_approx_2_dfe = self.y_experiment_results_2 - self.linear_y_approx_2_dfe
        self.y_minus_linear_y_approx_square_1_dfe = np.square(self.y_minus_linear_y_approx_1_dfe)
        self.y_minus_linear_y_approx_square_2_dfe = np.square(self.y_minus_linear_y_approx_2_dfe)

        self.y_minus_nonlinear_y_approx_1_dfe = self.y_experiment_results_1 - self.nonlinear_y_approx_1_dfe
        self.y_minus_nonlinear_y_approx_square_1_dfe = np.square(self.y_minus_nonlinear_y_approx_1_dfe)
        self.y_minus_nonlinear_y_approx_2_dfe = self.y_experiment_results_2 - self.nonlinear_y_approx_2_dfe
        self.y_minus_nonlinear_y_approx_square_2_dfe = np.square(self.y_minus_nonlinear_y_approx_2_dfe)

        self.dfe_result_matrix = np.c_[
            self.dfe_plan_matrix,
            self.y_experiment_results_1,
            self.linear_y_approx_1_dfe,
            self.nonlinear_y_approx_1_dfe,
            np.abs(self.y_minus_linear_y_approx_1_dfe),
            np.abs(self.y_minus_nonlinear_y_approx_1_dfe),
            self.y_minus_linear_y_approx_square_1_dfe,
            self.y_minus_nonlinear_y_approx_square_1_dfe,
            self.y_experiment_results_2,
            self.linear_y_approx_2_dfe,
            self.nonlinear_y_approx_2_dfe,
            np.abs(self.y_minus_linear_y_approx_2_dfe),
            np.abs(self.y_minus_nonlinear_y_approx_2_dfe),
            self.y_minus_linear_y_approx_square_2_dfe,
            self.y_minus_nonlinear_y_approx_square_2_dfe,
        ]

        return self.dfe_result_matrix, self.b_coefs_vector_1_dfe, self.b_coefs_vector_2_dfe

    def calculate_b_coefs_natural(self):
        unknown_number = self.factors_number + 1
        self.b_coefs_vector_1_natural = np.zeros(unknown_number)
        self.b_coefs_vector_2_natural = np.zeros(unknown_number)

        self.b_coefs_vector_1_natural[0] = self.b_coefs_vector_1[0]
        self.b_coefs_vector_2_natural[0] = self.b_coefs_vector_2[0]

        for i in range(1, unknown_number):
            x_nat_max_i = self.factors_max_natural[i]
            x_nat_min_i = self.factors_min_natural[i]
            self.b_coefs_vector_1_natural[0] += (self.b_coefs_vector_1[i] *
                                                 (x_nat_max_i + x_nat_min_i) / (x_nat_max_i - x_nat_min_i))
            self.b_coefs_vector_2_natural[0] += (self.b_coefs_vector_2[i] *
                                                 (x_nat_max_i + x_nat_min_i) / (x_nat_max_i - x_nat_min_i))

        for i in range(1, unknown_number):
            self.b_coefs_vector_1_natural[i] = (self.b_coefs_vector_1[i] * 2 /
                                                (self.factors_max_natural[i] - self.factors_min_natural[i]))
            self.b_coefs_vector_2_natural[i] = (self.b_coefs_vector_2[i] * 2 /
                                                (self.factors_max_natural[i] - self.factors_min_natural[i]))

        return self.b_coefs_vector_1_natural, self.b_coefs_vector_2_natural

    def calculate_linear_y_approx(self):
        self.linear_y_approx_1 = np.array([
            sum(self.plan_matrix[i, :self.linear_coef_num] * self.b_coefs_vector_1[:self.linear_coef_num])
            for i in range(self.N)
        ])
        self.linear_y_approx_2 = np.array([
            sum(self.plan_matrix[i, :self.linear_coef_num] * self.b_coefs_vector_2[:self.linear_coef_num])
            for i in range(self.N)
        ])

        return self.linear_y_approx_1, self.linear_y_approx_2

    def calculate_nonlinear_y_approx(self):
        self.nonlinear_y_approx_1 = np.array([
            sum(self.plan_matrix[i, :] * self.b_coefs_vector_1[:])
            for i in range(self.N)
        ])
        self.nonlinear_y_approx_2 = np.array([
            sum(self.plan_matrix[i, :] * self.b_coefs_vector_2[:])
            for i in range(self.N)
        ])

        return self.nonlinear_y_approx_1, self.nonlinear_y_approx_2

    def calculate_result_matrix(self, experiment_results_1: np.ndarray, experiment_results_2: np.ndarray):
        self.y_experiment_results_1 = experiment_results_1
        self.y_experiment_results_2 = experiment_results_2

        self.calculate_b_coefs()
        self.calculate_linear_y_approx()
        self.calculate_nonlinear_y_approx()

        self.y_minus_linear_y_approx_1 = self.y_experiment_results_1 - self.linear_y_approx_1
        self.y_minus_linear_y_approx_2 = self.y_experiment_results_2 - self.linear_y_approx_2
        self.y_minus_linear_y_approx_square_1 = np.square(self.y_minus_linear_y_approx_1)
        self.y_minus_linear_y_approx_square_2 = np.square(self.y_minus_linear_y_approx_2)

        self.y_minus_nonlinear_y_approx_1 = self.y_experiment_results_1 - self.nonlinear_y_approx_1
        self.y_minus_nonlinear_y_approx_square_1 = np.square(self.y_minus_nonlinear_y_approx_1)
        self.y_minus_nonlinear_y_approx_2 = self.y_experiment_results_2 - self.nonlinear_y_approx_2
        self.y_minus_nonlinear_y_approx_square_2 = np.square(self.y_minus_nonlinear_y_approx_2)

        self.result_matrix = np.c_[
            self.plan_matrix,
            self.y_experiment_results_1,
            self.linear_y_approx_1,
            self.nonlinear_y_approx_1,
            np.abs(self.y_minus_linear_y_approx_1),
            np.abs(self.y_minus_nonlinear_y_approx_1),
            self.y_minus_linear_y_approx_square_1,
            self.y_minus_nonlinear_y_approx_square_1,
            self.y_experiment_results_2,
            self.linear_y_approx_2,
            self.nonlinear_y_approx_2,
            np.abs(self.y_minus_linear_y_approx_2),
            np.abs(self.y_minus_nonlinear_y_approx_2),
            self.y_minus_linear_y_approx_square_2,
            self.y_minus_nonlinear_y_approx_square_2,
        ]

        return self.result_matrix, self.b_coefs_vector_1, self.b_coefs_vector_2

    def calculate_sum_remaining(self):
        self.sum_r_linear_1 = float(np.transpose(self.y_minus_linear_y_approx_1) @ self.y_minus_linear_y_approx_1)
        self.sum_r_nonlinear_1 = float(np.transpose(self.y_minus_nonlinear_y_approx_1) @ self.y_minus_nonlinear_y_approx_1)
        self.sum_r_linear_2 = float(np.transpose(self.y_minus_linear_y_approx_2) @ self.y_minus_linear_y_approx_2)
        self.sum_r_nonlinear_2 = float(np.transpose(self.y_minus_nonlinear_y_approx_2) @ self.y_minus_nonlinear_y_approx_2)

        return self.sum_r_linear_1, self.sum_r_nonlinear_1, self.sum_r_linear_2, self.sum_r_nonlinear_2

    def calculate_sum_remaining_dfe(self):
        self.sum_r_linear_1_dfe = float(np.transpose(self.y_minus_linear_y_approx_1_dfe) @ self.y_minus_linear_y_approx_1_dfe)
        self.sum_r_nonlinear_1_dfe = float(np.transpose(self.y_minus_nonlinear_y_approx_1_dfe) @ self.y_minus_nonlinear_y_approx_1_dfe)
        self.sum_r_linear_2_dfe = float(np.transpose(self.y_minus_linear_y_approx_2_dfe) @ self.y_minus_linear_y_approx_2_dfe)
        self.sum_r_nonlinear_2_dfe = float(np.transpose(self.y_minus_nonlinear_y_approx_2_dfe) @ self.y_minus_nonlinear_y_approx_2_dfe)

        return self.sum_r_linear_1_dfe, self.sum_r_nonlinear_1_dfe, self.sum_r_linear_2_dfe, self.sum_r_nonlinear_2_dfe

    def run_dfe(self):
        self.create_dfe_plan_matrix()
        experiment_results_1 = np.zeros(self.dfe_num_of_experiments)
        experiment_results_2 = np.zeros(self.dfe_num_of_experiments)

        for e_number, e_params in enumerate(self.dfe_plan_matrix):
            lambda1_natural = self.normal_to_natural(e_params[1], self.lambda1_min, self.lambda1_max)
            lambda2_natural = self.normal_to_natural(e_params[2], self.lambda2_min, self.lambda2_max)
            mu1_natural = self.normal_to_natural(e_params[3], self.mu1_min, self.mu1_max)
            mu2_natural = self.normal_to_natural(e_params[4], self.mu2_min, self.mu2_max)

            wait_times = [0, 0]

            print(f'R = {lambda1_natural/mu1_natural + lambda2_natural/mu2_natural:.5f}\np1 = {lambda1_natural/mu1_natural}')
            for _ in range(self.test_num):
                generators = [
                    Generator(PoissonDistribution(lambda1_natural), REQUEST_TYPE_ONE),
                    Generator(PoissonDistribution(lambda2_natural), REQUEST_TYPE_TWO),
                ]

                processors = [
                    Processor([PoissonDistribution(mu1_natural), PoissonDistribution(mu2_natural)])
                ]

                for gen in generators:
                    gen.receivers = processors.copy()

                simulator = Simulator(generators, processors)
                result_stats = simulator.simulate(self.num_requests)

                wait_times[REQUEST_TYPE_ONE] += result_stats.avg_wait_times[REQUEST_TYPE_ONE]
                wait_times[REQUEST_TYPE_TWO] += result_stats.avg_wait_times[REQUEST_TYPE_TWO]

            experiment_results_1[e_number] = wait_times[REQUEST_TYPE_ONE] / self.test_num
            experiment_results_2[e_number] = wait_times[REQUEST_TYPE_TWO] / self.test_num

        return self.calculate_result_matrix_dfe(experiment_results_1, experiment_results_2)

    def run(self):
        self.create_plan_matrix()
        experiment_results_1 = np.zeros(self.N)
        experiment_results_2 = np.zeros(self.N)

        for e_number, e_params in enumerate(self.plan_matrix):
            lambda1_natural = self.normal_to_natural(e_params[1], self.lambda1_min, self.lambda1_max)
            lambda2_natural = self.normal_to_natural(e_params[2], self.lambda2_min, self.lambda2_max)
            mu1_natural = self.normal_to_natural(e_params[3], self.mu1_min, self.mu1_max)
            mu2_natural = self.normal_to_natural(e_params[4], self.mu2_min, self.mu2_max)

            wait_times = [0, 0]

            print(f'R = {lambda1_natural/mu1_natural + lambda2_natural/mu2_natural:.5f}\np1 = {lambda1_natural/mu1_natural}')
            for _ in range(self.test_num):
                generators = [
                    Generator(PoissonDistribution(lambda1_natural), REQUEST_TYPE_ONE),
                    Generator(PoissonDistribution(lambda2_natural), REQUEST_TYPE_TWO),
                ]

                processors = [
                    Processor([PoissonDistribution(mu1_natural), PoissonDistribution(mu2_natural)])
                ]

                for gen in generators:
                    gen.receivers = processors.copy()

                simulator = Simulator(generators, processors)
                result_stats = simulator.simulate(self.num_requests)

                wait_times[REQUEST_TYPE_ONE] += result_stats.avg_wait_times[REQUEST_TYPE_ONE]
                wait_times[REQUEST_TYPE_TWO] += result_stats.avg_wait_times[REQUEST_TYPE_TWO]

            experiment_results_1[e_number] = wait_times[REQUEST_TYPE_ONE] / self.test_num
            experiment_results_2[e_number] = wait_times[REQUEST_TYPE_TWO] / self.test_num

        return self.calculate_result_matrix(experiment_results_1, experiment_results_2)

    def check(self, l1_norm: float, l2_norm: float, mu1_norm: float, mu2_norm: float):
        lambda1_natural = self.normal_to_natural(l1_norm, self.lambda1_min, self.lambda1_max)
        lambda2_natural = self.normal_to_natural(l2_norm, self.lambda2_min, self.lambda2_max)
        mu1_natural = self.normal_to_natural(mu1_norm, self.mu1_min, self.mu1_max)
        mu2_natural = self.normal_to_natural(mu2_norm, self.mu2_min, self.mu2_max)

        wait_times = [0, 0]

        for _ in range(self.test_num):
            generators = [
                Generator(PoissonDistribution(lambda1_natural), REQUEST_TYPE_ONE),
                Generator(PoissonDistribution(lambda2_natural), REQUEST_TYPE_TWO),
            ]

            processors = [
                Processor([PoissonDistribution(mu1_natural), PoissonDistribution(mu2_natural)])
            ]

            for gen in generators:
                gen.receivers = processors.copy()

            simulator = Simulator(generators, processors)
            result_stats = simulator.simulate(self.num_requests)

            wait_times[REQUEST_TYPE_ONE] += result_stats.avg_wait_times[REQUEST_TYPE_ONE]
            wait_times[REQUEST_TYPE_TWO] += result_stats.avg_wait_times[REQUEST_TYPE_TWO]

        experiment_result_1 = wait_times[REQUEST_TYPE_ONE] / self.test_num
        experiment_result_2 = wait_times[REQUEST_TYPE_TWO] / self.test_num

        experiment_plan_row = np.array([1, l1_norm, l2_norm, mu1_norm, mu2_norm,
                                        l1_norm * l2_norm, l1_norm * mu1_norm, l1_norm * mu2_norm, l2_norm * mu1_norm, l2_norm * mu2_norm, mu1_norm * mu2_norm,
                                        l1_norm * l2_norm * mu1_norm, l1_norm * l2_norm * mu2_norm,
                                        l1_norm * mu1_norm * mu2_norm, l2_norm * mu1_norm * mu2_norm, l1_norm * mu1_norm * mu2_norm * l2_norm
                                        ])
        nonlinear_y_approx_1 = sum(experiment_plan_row * self.b_coefs_vector_1)
        nonlinear_y_approx_2 = sum(experiment_plan_row * self.b_coefs_vector_2)
        linear_y_approx_1 = sum(experiment_plan_row[:self.linear_coef_num] * self.b_coefs_vector_1[:self.linear_coef_num])
        linear_y_approx_2 = sum(experiment_plan_row[:self.linear_coef_num] * self.b_coefs_vector_2[:self.linear_coef_num])

        total = list(experiment_plan_row)
        total.extend([experiment_result_1, linear_y_approx_1, nonlinear_y_approx_1,
                      experiment_result_1 - linear_y_approx_1, experiment_result_1 - nonlinear_y_approx_1,
                      (experiment_result_1 - linear_y_approx_1) ** 2, (experiment_result_1 - nonlinear_y_approx_1) ** 2,
                      experiment_result_2, linear_y_approx_2, nonlinear_y_approx_2,
                      experiment_result_2 - linear_y_approx_2, experiment_result_2 - nonlinear_y_approx_2,
                      (experiment_result_2 - linear_y_approx_2) ** 2, (experiment_result_2 - nonlinear_y_approx_2) ** 2,
                      ])

        return np.array(total)

    def calculate_b_coefs_ockp(self):
        """Вычисление коэффициентов регрессии для ОЦКП (аналог calculate_b_coefs_dfe)"""
        self.b_coefs_vector_1_ockp = np.array([
            np.mean((self.ockp_plan_matrix[:, j] * self.y_experiment_results_1))
            for j in range(self.ockp_plan_matrix.shape[1])
        ])

        self.b_coefs_vector_2_ockp = np.array([
            np.mean((self.ockp_plan_matrix[:, j] * self.y_experiment_results_2))
            for j in range(self.ockp_plan_matrix.shape[1])
        ])

        return self.b_coefs_vector_1_ockp, self.b_coefs_vector_2_ockp

    def calculate_b_coefs_natural_ockp(self):
        """Вычисление натуральных коэффициентов регрессии для ОЦКП"""
        unknown_number = self.factors_number + 1  # Количество неизвестных (b0, b1, b2, b3)

        # Инициализация векторов натуральных коэффициентов
        self.b_coefs_vector_1_natural_ockp = np.zeros(unknown_number)
        self.b_coefs_vector_2_natural_ockp = np.zeros(unknown_number)

        # Расчет свободного члена (b0) с учетом преобразования
        self.b_coefs_vector_1_natural_ockp[0] = self.b_coefs_vector_1_ockp[0]
        self.b_coefs_vector_2_natural_ockp[0] = self.b_coefs_vector_2_ockp[0]

        for i in range(1, unknown_number):
            x_nat_max_i = self.factors_max_natural[i - 1]
            x_nat_min_i = self.factors_min_natural[i - 1]

            # Коррекция свободного члена
            self.b_coefs_vector_1_natural_ockp[0] += (self.b_coefs_vector_1_ockp[i] *
                                                      (x_nat_max_i + x_nat_min_i) /
                                                      (x_nat_max_i - x_nat_min_i))

            self.b_coefs_vector_2_natural_ockp[0] += (self.b_coefs_vector_2_ockp[i] *
                                                      (x_nat_max_i + x_nat_min_i) /
                                                      (x_nat_max_i - x_nat_min_i))

        # Расчет линейных коэффициентов
        for i in range(1, unknown_number):
            self.b_coefs_vector_1_natural_ockp[i] = (self.b_coefs_vector_1_ockp[i] * 2 /
                                                     (self.factors_max_natural[i - 1] - self.factors_min_natural[
                                                         i - 1]))

            self.b_coefs_vector_2_natural_ockp[i] = (self.b_coefs_vector_2_ockp[i] * 2 /
                                                     (self.factors_max_natural[i - 1] - self.factors_min_natural[
                                                         i - 1]))

        return self.b_coefs_vector_1_natural_ockp, self.b_coefs_vector_2_natural_ockp

    def calculate_sum_remaining_ockp(self):
        """Вычисление остаточных сумм для ОЦКП (аналог calculate_sum_remaining_dfe)"""
        # Для первого отклика (y1)
        # self.sum_r_linear_1_ockp = float(np.transpose(self.y_minus_linear_y_approx_1_ockp) @
        #                                  self.y_minus_linear_y_approx_1_ockp)
        self.sum_r_nonlinear_1_ockp = float(np.transpose(self.y_minus_nonlinear_y_approx_1_ockp) @
                                            self.y_minus_nonlinear_y_approx_1_ockp)

        # Для второго отклика (y2)
        # self.sum_r_linear_2_ockp = float(np.transpose(self.y_minus_linear_y_approx_2_ockp) @
        #                                  self.y_minus_linear_y_approx_2_ockp)
        self.sum_r_nonlinear_2_ockp = float(np.transpose(self.y_minus_nonlinear_y_approx_2_ockp) @
                                            self.y_minus_nonlinear_y_approx_2_ockp)

        # return (self.sum_r_linear_1_ockp, self.sum_r_nonlinear_1_ockp,
        #         self.sum_r_linear_2_ockp, self.sum_r_nonlinear_2_ockp)
        return self.sum_r_nonlinear_1_ockp, self.sum_r_nonlinear_2_ockp

    def calculate_nonlinear_y_approx_ockp(self):
        """Вычисление аппроксимации нелинейной модели для ОЦКП (аналог calculate_nonlinear_y_approx_dfe)"""
        self.nonlinear_y_approx_1_ockp = np.array([
            sum(self.ockp_plan_matrix[i, :] * self.b_coefs_vector_1_ockp[:])
            for i in range(self.ockp_num_of_experiments)
        ])

        self.nonlinear_y_approx_2_ockp = np.array([
            sum(self.ockp_plan_matrix[i, :] * self.b_coefs_vector_2_ockp[:])
            for i in range(self.ockp_num_of_experiments)
        ])

        return self.nonlinear_y_approx_1_ockp, self.nonlinear_y_approx_2_ockp

    def calculate_result_matrix_ockp(self, experiment_results_1: np.ndarray, experiment_results_2: np.ndarray):
        """Формирование результирующей матрицы для ОЦКП с расчетом остатков"""
        self.y_experiment_results_1 = experiment_results_1
        self.y_experiment_results_2 = experiment_results_2

        self.calculate_b_coefs_ockp()
        self.calculate_nonlinear_y_approx_ockp()

        # Расчет ошибок для первого отклика
        self.y_minus_nonlinear_y_approx_1_ockp = (self.y_experiment_results_1 -
                                                  self.nonlinear_y_approx_1_ockp)
        self.y_minus_nonlinear_y_approx_square_1_ockp = np.square(
            self.y_minus_nonlinear_y_approx_1_ockp)

        # Расчет ошибок для второго отклика
        self.y_minus_nonlinear_y_approx_2_ockp = (self.y_experiment_results_2 -
                                                  self.nonlinear_y_approx_2_ockp)
        self.y_minus_nonlinear_y_approx_square_2_ockp = np.square(
            self.y_minus_nonlinear_y_approx_2_ockp)

        # Расчет остаточных сумм
        self.calculate_sum_remaining_ockp()

        # Формирование результирующей матрицы
        self.ockp_result_matrix = np.c_[
            self.ockp_plan_matrix,
            self.y_experiment_results_1,
            self.nonlinear_y_approx_1_ockp,
            np.abs(self.y_minus_nonlinear_y_approx_1_ockp),
            self.y_minus_nonlinear_y_approx_square_1_ockp,
            self.y_experiment_results_2,
            self.nonlinear_y_approx_2_ockp,
            np.abs(self.y_minus_nonlinear_y_approx_2_ockp),
            self.y_minus_nonlinear_y_approx_square_2_ockp
        ]

        return self.ockp_result_matrix, self.b_coefs_vector_1_ockp, self.b_coefs_vector_2_ockp

    def run_ockp(self):
        """Запуск экспериментов по плану ОЦКП"""
        self.create_ockp_plan_matrix()
        experiment_results_1 = np.zeros(self.ockp_num_of_experiments)
        experiment_results_2 = np.zeros(self.ockp_num_of_experiments)

        for e_number, e_params in enumerate(self.ockp_plan_matrix):
            lambda1_natural = self.normal_to_natural(e_params[1], self.lambda1_min, self.lambda1_max)
            lambda2_natural = self.normal_to_natural(e_params[2], self.lambda2_min, self.lambda2_max)
            mu1_natural = self.normal_to_natural(e_params[3], self.mu1_min, self.mu1_max)
            mu2_natural = self.normal_to_natural(e_params[4], self.mu2_min, self.mu2_max)

            wait_times = [0, 0]

            print(
                f'R = {lambda1_natural / mu1_natural + lambda2_natural / mu2_natural:.5f}\np1 = {lambda1_natural / mu1_natural}')
            for _ in range(self.test_num):
                generators = [
                    Generator(PoissonDistribution(lambda1_natural), REQUEST_TYPE_ONE),
                    Generator(PoissonDistribution(lambda2_natural), REQUEST_TYPE_TWO),
                ]

                processors = [
                    Processor([PoissonDistribution(mu1_natural), PoissonDistribution(mu2_natural)])
                ]

                for gen in generators:
                    gen.receivers = processors.copy()

                simulator = Simulator(generators, processors)
                result_stats = simulator.simulate(self.num_requests)

                wait_times[REQUEST_TYPE_ONE] += result_stats.avg_wait_times[REQUEST_TYPE_ONE]
                wait_times[REQUEST_TYPE_TWO] += result_stats.avg_wait_times[REQUEST_TYPE_TWO]

                experiment_results_1[e_number] = wait_times[REQUEST_TYPE_ONE] / self.test_num
                experiment_results_2[e_number] = wait_times[REQUEST_TYPE_TWO] / self.test_num

        return self.calculate_result_matrix_ockp(experiment_results_1, experiment_results_2)

    def check_ockp(self, l1_norm: float, l2_norm: float, mu1_norm: float, mu2_norm: float):
        """Проверка модели ОЦКП в заданной точке"""
        lambda1_natural = self.normal_to_natural(l1_norm, self.lambda1_min, self.lambda1_max)
        lambda2_natural = self.normal_to_natural(l2_norm, self.lambda2_min, self.lambda2_max)
        mu1_natural = self.normal_to_natural(mu1_norm, self.mu1_min, self.mu1_max)
        mu2_natural = self.normal_to_natural(mu2_norm, self.mu2_min, self.mu2_max)

        wait_times = [0, 0]

        for _ in range(self.test_num):
            generators = [
                Generator(PoissonDistribution(lambda1_natural), REQUEST_TYPE_ONE),
                Generator(PoissonDistribution(lambda2_natural), REQUEST_TYPE_TWO),
            ]

            processors = [
                Processor([PoissonDistribution(mu1_natural), PoissonDistribution(mu2_natural)])
            ]

            for gen in generators:
                gen.receivers = processors.copy()

            simulator = Simulator(generators, processors)
            result_stats = simulator.simulate(self.num_requests)

            wait_times[REQUEST_TYPE_ONE] += result_stats.avg_wait_times[REQUEST_TYPE_ONE]
            wait_times[REQUEST_TYPE_TWO] += result_stats.avg_wait_times[REQUEST_TYPE_TWO]

        experiment_result_1 = wait_times[REQUEST_TYPE_ONE] / self.test_num
        experiment_result_2 = wait_times[REQUEST_TYPE_TWO] / self.test_num

        # Формирование вектора факторов для проверки (включая квадратичные эффекты)
        experiment_plan_row = np.array([
            1,  # x0
            l1_norm, l2_norm, mu1_norm, mu2_norm,  # линейные эффекты
            l1_norm * l2_norm, l2_norm * mu1_norm, l1_norm * mu1_norm,  # парные взаимодействия
            l1_norm ** 2 - self.ockp_coef_a, l2_norm ** 2 - self.ockp_coef_a, mu1_norm ** 2 - self.ockp_coef_a
            # квадратичные эффекты
        ])

        nonlinear_y_approx_1 = sum(experiment_plan_row * self.b_coefs_vector_1_ockp)
        nonlinear_y_approx_2 = sum(experiment_plan_row * self.b_coefs_vector_2_ockp)

        total = list(experiment_plan_row)
        total.extend([
            experiment_result_1, nonlinear_y_approx_1,
            experiment_result_1 - nonlinear_y_approx_1,
            (experiment_result_1 - nonlinear_y_approx_1) ** 2,
            experiment_result_2, nonlinear_y_approx_2,
            experiment_result_2 - nonlinear_y_approx_2,
            (experiment_result_2 - nonlinear_y_approx_2) ** 2
        ])

        return np.array(total)

    def check_dfe(self, l1_norm: float, l2_norm: float, mu1_norm: float, mu2_norm: float):
        lambda1_natural = self.normal_to_natural(l1_norm, self.lambda1_min, self.lambda1_max)
        lambda2_natural = self.normal_to_natural(l2_norm, self.lambda2_min, self.lambda2_max)
        mu1_natural = self.normal_to_natural(mu1_norm, self.mu1_min, self.mu1_max)
        mu2_natural = self.normal_to_natural(mu2_norm, self.mu2_min, self.mu2_max)

        wait_times = [0, 0]

        for _ in range(self.test_num):
            generators = [
                Generator(PoissonDistribution(lambda1_natural), REQUEST_TYPE_ONE),
                Generator(PoissonDistribution(lambda2_natural), REQUEST_TYPE_TWO),
            ]

            processors = [
                Processor([PoissonDistribution(mu1_natural), PoissonDistribution(mu2_natural)])
            ]

            for gen in generators:
                gen.receivers = processors.copy()

            simulator = Simulator(generators, processors)
            result_stats = simulator.simulate(self.num_requests)

            wait_times[REQUEST_TYPE_ONE] += result_stats.avg_wait_times[REQUEST_TYPE_ONE]
            wait_times[REQUEST_TYPE_TWO] += result_stats.avg_wait_times[REQUEST_TYPE_TWO]

        experiment_result_1 = wait_times[REQUEST_TYPE_ONE] / self.test_num
        experiment_result_2 = wait_times[REQUEST_TYPE_TWO] / self.test_num

        experiment_plan_row = np.array([1, l1_norm, l2_norm, mu1_norm, mu2_norm,
                                        l1_norm * l2_norm, l1_norm * mu1_norm, l2_norm * mu1_norm,
                                        l1_norm * l2_norm * mu1_norm
                                        ])
        nonlinear_y_approx_1 = sum(experiment_plan_row * self.b_coefs_vector_1_dfe)
        nonlinear_y_approx_2 = sum(experiment_plan_row * self.b_coefs_vector_2_dfe)
        linear_y_approx_1 = sum(experiment_plan_row[:self.linear_coef_num] * self.b_coefs_vector_1_dfe[:self.linear_coef_num])
        linear_y_approx_2 = sum(experiment_plan_row[:self.linear_coef_num] * self.b_coefs_vector_2_dfe[:self.linear_coef_num])

        total = list(experiment_plan_row)
        total.extend([experiment_result_1, linear_y_approx_1, nonlinear_y_approx_1,
                      experiment_result_1 - linear_y_approx_1, experiment_result_1 - nonlinear_y_approx_1,
                      (experiment_result_1 - linear_y_approx_1) ** 2, (experiment_result_1 - nonlinear_y_approx_1) ** 2,
                      experiment_result_2, linear_y_approx_2, nonlinear_y_approx_2,
                      experiment_result_2 - linear_y_approx_2, experiment_result_2 - nonlinear_y_approx_2,
                      (experiment_result_2 - linear_y_approx_2) ** 2, (experiment_result_2 - nonlinear_y_approx_2) ** 2,
                      ])

        return np.array(total)
