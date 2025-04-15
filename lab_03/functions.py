from typing import List, Dict

from numpy.ma.core import arange

from lab_03.distribution import PoissonDistribution
from lab_03.generator import Generator
from lab_03.processor import Processor
from lab_03.request import REQUEST_TYPE_ONE, REQUEST_TYPE_TWO
from lab_03.simulator import Simulator, Statistics


class SimulationResults:
    stats: Statistics
    load_ratio_type_one: float
    load_ratio_type_two: float

    def __init__(self, stats: Statistics, load_ratios: List[float]):
        self.stats = stats
        self.load_ratio_type_one = load_ratios[REQUEST_TYPE_ONE]
        self.load_ratio_type_two = load_ratios[REQUEST_TYPE_TWO]


def run_simulation_once(
        requests_per_generator: int,
        requests_processed: int,
        lambda1: float,
        lambda2: float,
        mu1: float,
        mu2: float
):
    generators = [
        Generator(PoissonDistribution(lambda1), REQUEST_TYPE_ONE, requests_per_generator),
        Generator(PoissonDistribution(lambda2), REQUEST_TYPE_TWO, requests_per_generator),
    ]

    processors = [
        Processor([PoissonDistribution(mu1), PoissonDistribution(mu2)])
    ]

    for gen in generators:
        gen.receivers = processors.copy()

    simulator = Simulator(generators, processors)
    result_stats = simulator.simulate(requests_processed)
    theoretical_load_ratios = [lambda1 / mu1, lambda2 / mu2 + lambda1 / mu1]

    return SimulationResults(result_stats, theoretical_load_ratios)


class PassiveSimulationResults:
    # stats: Statistics
    load_ratios_type_one: List[float]
    load_ratios_type_two: List[float]
    wait_times_type_one: List[float]
    wait_times_type_two: List[float]
    lambda1_list: List[float]
    lambda2_list: List[float]

    def __init__(self, result_load_ratios: List[List[float]], result_wait_times: List[List[float]],
                 lambda1_list: List[float], lambda2_list: List[float]):
        # self.stats = stats
        self.load_ratios_type_one = result_load_ratios[REQUEST_TYPE_ONE]
        self.load_ratios_type_two = result_load_ratios[REQUEST_TYPE_TWO]
        self.wait_times_type_one = result_wait_times[REQUEST_TYPE_ONE]
        self.wait_times_type_two = result_wait_times[REQUEST_TYPE_TWO]
        self.lambda1_list = lambda1_list
        self.lambda2_list = lambda2_list

def run_simulation_passive(
        requests_processed: int,
        lambda1_dict: Dict,
        mu1: float,
        mu2: float,
        num_tests
):
    result_load_ratios = [[], []]
    result_wait_times = [[], []]
    lambda1_list = []
    lambda2_list = []


    for lambda1 in arange(lambda1_dict['min'], lambda1_dict['max'], lambda1_dict['step']):
        lambda1_list.append(lambda1)
        lambda2 = lambda1 * 1.5
        lambda2_list.append(lambda2)
        result_load_ratio_1 = lambda1/mu1
        result_load_ratio_2 = lambda2/mu2
        result_load_ratios[REQUEST_TYPE_ONE].append(result_load_ratio_1)
        result_load_ratios[REQUEST_TYPE_TWO].append(result_load_ratio_2)

        sum_avg_wait_times_type_one = 0
        sum_avg_wait_times_type_two = 0
        for i in range(num_tests):
            results = run_simulation_once(requests_processed * 2000, requests_processed, lambda1, lambda2, mu1, mu2)
            sum_avg_wait_times_type_one += results.stats.avg_wait_times[REQUEST_TYPE_ONE]
            sum_avg_wait_times_type_two += results.stats.avg_wait_times[REQUEST_TYPE_TWO]

        result_wait_times[REQUEST_TYPE_ONE].append(sum_avg_wait_times_type_one / num_tests)
        result_wait_times[REQUEST_TYPE_TWO].append(sum_avg_wait_times_type_two / num_tests)

    return PassiveSimulationResults(result_load_ratios, result_wait_times, lambda1_list, lambda2_list)

class PassiveSimulationResultsByFactor:
    lambda1_list: List[float]
    lambda2_list: List[float]
    mu1_list: List[float]
    mu2_list: List[float]
    result_wait_time_lambda1: List[List[float]]
    result_wait_time_lambda2: List[List[float]]
    result_wait_time_mu1: List[List[float]]
    result_wait_time_mu2: List[List[float]]

    def __init__(self, factor_list: List[List[float]], result_wait_times: List[List[List[float]]]):
        self.lambda1_list = factor_list[0]
        self.lambda2_list = factor_list[1]
        self.mu1_list = factor_list[2]
        self.mu2_list = factor_list[3]
        self.result_wait_time_lambda1 = result_wait_times[0]
        self.result_wait_time_lambda2 = result_wait_times[1]
        self.result_wait_time_mu1 = result_wait_times[2]
        self.result_wait_time_mu2 = result_wait_times[3]


def run_simulation_passive_by_factor(
        requests_processed: int,
        factor_dict: Dict,
        num_tests
):
    result_wait_times = [[[], []], [[], []], [[], []], [[], []]]
    factor_list = [[], [], [], []]

    for lambda1 in arange(factor_dict['min'], factor_dict['max'], factor_dict['step']):
        factor_list[0].append(lambda1)

        sum_avg_wait_times_type_one = 0
        sum_avg_wait_times_type_two = 0
        for i in range(num_tests):
            results = run_simulation_once(requests_processed * 2000, requests_processed,
                                          lambda1, factor_dict['fix'], factor_dict['fix'], factor_dict['fix'])
            sum_avg_wait_times_type_one += results.stats.avg_wait_times[REQUEST_TYPE_ONE]
            sum_avg_wait_times_type_two += results.stats.avg_wait_times[REQUEST_TYPE_TWO]

        result_wait_times[0][REQUEST_TYPE_ONE].append(sum_avg_wait_times_type_one / num_tests)
        result_wait_times[0][REQUEST_TYPE_TWO].append(sum_avg_wait_times_type_two / num_tests)

    for lambda2 in arange(factor_dict['min'], factor_dict['max'], factor_dict['step']):
        factor_list[1].append(lambda2)

        sum_avg_wait_times_type_one = 0
        sum_avg_wait_times_type_two = 0
        for i in range(num_tests):
            results = run_simulation_once(requests_processed * 2000, requests_processed,
                                          factor_dict['fix'], lambda2, factor_dict['fix'], factor_dict['fix'])
            sum_avg_wait_times_type_one += results.stats.avg_wait_times[REQUEST_TYPE_ONE]
            sum_avg_wait_times_type_two += results.stats.avg_wait_times[REQUEST_TYPE_TWO]

        result_wait_times[1][REQUEST_TYPE_ONE].append(sum_avg_wait_times_type_one / num_tests)
        result_wait_times[1][REQUEST_TYPE_TWO].append(sum_avg_wait_times_type_two / num_tests)

    for mu1 in arange(factor_dict['min'], factor_dict['max'], factor_dict['step']):
        factor_list[2].append(mu1)

        sum_avg_wait_times_type_one = 0
        sum_avg_wait_times_type_two = 0
        for i in range(num_tests):
            results = run_simulation_once(requests_processed * 2000, requests_processed,
                                          factor_dict['fix'], factor_dict['fix'], mu1, factor_dict['fix'])
            sum_avg_wait_times_type_one += results.stats.avg_wait_times[REQUEST_TYPE_ONE]
            sum_avg_wait_times_type_two += results.stats.avg_wait_times[REQUEST_TYPE_TWO]

        result_wait_times[2][REQUEST_TYPE_ONE].append(sum_avg_wait_times_type_one / num_tests)
        result_wait_times[2][REQUEST_TYPE_TWO].append(sum_avg_wait_times_type_two / num_tests)

    for mu2 in arange(factor_dict['min'], factor_dict['max'], factor_dict['step']):
        factor_list[3].append(mu2)

        sum_avg_wait_times_type_one = 0
        sum_avg_wait_times_type_two = 0
        for i in range(num_tests):
            results = run_simulation_once(requests_processed * 2000, requests_processed,
                                          factor_dict['fix'], factor_dict['fix'], factor_dict['fix'], mu2)
            sum_avg_wait_times_type_one += results.stats.avg_wait_times[REQUEST_TYPE_ONE]
            sum_avg_wait_times_type_two += results.stats.avg_wait_times[REQUEST_TYPE_TWO]

        result_wait_times[3][REQUEST_TYPE_ONE].append(sum_avg_wait_times_type_one / num_tests)
        result_wait_times[3][REQUEST_TYPE_TWO].append(sum_avg_wait_times_type_two / num_tests)

    return PassiveSimulationResultsByFactor(factor_list, result_wait_times)
