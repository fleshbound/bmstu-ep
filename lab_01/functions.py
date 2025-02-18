from typing import List, Dict

from numpy.ma.core import arange

from lab_01.distribution import PoissonDistribution
from lab_01.generator import Generator
from lab_01.processor import Processor
from lab_01.request import REQUEST_TYPE_ONE, REQUEST_TYPE_TWO
from lab_01.simulator import Simulator, Statistics


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
    stats: Statistics
    load_ratios_type_one: List[float]
    load_ratios_type_two: List[float]
    wait_times_type_one: List[float]
    wait_times_type_two: List[float]

    def __init__(self, stats: Statistics, result_load_ratios: List[List[float]], result_wait_times: List[List[float]]):
        self.stats = stats
        self.load_ratios_type_one = result_load_ratios[REQUEST_TYPE_ONE]
        self.load_ratios_type_two = result_load_ratios[REQUEST_TYPE_TWO]
        self.wait_times_type_one = result_wait_times[REQUEST_TYPE_ONE]
        self.wait_times_type_two = result_wait_times[REQUEST_TYPE_TWO]


def run_simulation_passive(
        requests_processed: int,
        lambda1_dict: Dict,
        lambda2_dict: Dict,
        mu1_dict: Dict,
        mu2_dict: Dict
):
    result_load_ratios = [[], []]
    result_wait_times = [[], []]

    for lambda1 in arange(lambda1_dict['min'], lambda1_dict['max'], lambda1_dict['step']):
        lambda2 = lambda1
        for mu1 in arange(mu1_dict['min'], mu1_dict['max'], mu1_dict['step']):
            mu2 = mu1
            results = run_simulation_once(requests_processed + 1000, requests_processed, lambda1, lambda2, mu1, mu2)

            result_load_ratio_1 = lambda1/mu1
            result_load_ratio_2 = lambda2/mu2 + lambda1/mu1

            result_load_ratios[REQUEST_TYPE_ONE].append(result_load_ratio_1)
            result_load_ratios[REQUEST_TYPE_TWO].append(result_load_ratio_2)

            result_wait_times[REQUEST_TYPE_ONE].append(results.stats.avg_wait_times[REQUEST_TYPE_ONE])
            result_wait_times[REQUEST_TYPE_TWO].append(results.stats.avg_wait_times[REQUEST_TYPE_TWO])

    return PassiveSimulationResults(result_load_ratios, result_wait_times)
