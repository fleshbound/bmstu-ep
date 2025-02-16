from typing import List

from generator import Generator
from processor import Processor
from queue import Queue


class Statistics:
    refused: int
    processed: int
    created: int
    total_time: float
    waiting_times: List[float]
    avg_wait_time: float

    def __init__(self):
        self.refused = 0
        self.processed = 0
        self.created = 0
        self.total_time = 0
        self.avg_wait_time = 0
        self.waiting_times = []


class Simulator:
    generators: List[Generator]
    processors: List[Processor]
    stats: Statistics

    def __init__(self, generators: List[Generator], processors: List[Processor]):
        self.generators = generators
        self.processors = processors
        self.stats = Statistics()

    def simulate(self, num_requests: int) -> Statistics:
        """
        Start event mode simulation
        :param num_requests: Number of processed requests
        :return: statistics
        """

        for generator in self.generators:
            generator.receivers = self.processors.copy()
