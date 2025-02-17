from typing import List

from generator import Generator
from lab_01.request import REQUEST_TYPE_ONE
from processor import Processor


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

    def clear(self):
        self.refused = 0
        self.processed = 0
        self.created = 0
        self.total_time = 0
        self.avg_wait_time = 0
        self.waiting_times = []

    def calculate_avg_waiting_time(self):
        for t in self.waiting_times:
            self.avg_wait_time += t

        self.avg_wait_time /= len(self.waiting_times)


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

        self.stats.clear()

        for generator in self.generators:
            generator.next = generator.next_time_interval()

        self.processors[0].next = self.processors[0].next_time_interval(REQUEST_TYPE_ONE)
        elements = self.generators + self.processors

        while self.stats.processed <= num_requests:
            cur_sim_time = self.generators[0].next

            for elem in elements:
                if 0 < elem.next < cur_sim_time:
                    cur_sim_time = elem.next

            for elem in elements:
                if elem.next == cur_sim_time:
                    if isinstance(elem, Processor):
                        # если очередной элемент - ОА
                        elem.end_processing()
                        self.stats.processed += 1

                        cur_request = elem.start_processing(cur_sim_time)
                        self.stats.waiting_times.append(cur_request.waiting_time_interval)
                        elem.next = elem.next_time_interval(cur_request.type)
                    else:
                        # если очередной элемент - генератор
                        processor = elem.generate(cur_sim_time)
                        self.stats.created += 1

                        if processor.next == 0:
                            processor.next = cur_sim_time + processor.next_time_interval(elem.type)
                        else:
                            self.stats.refused += 1

                        elem.next = cur_sim_time + elem.next_time_interval()

        self.stats.calculate_avg_waiting_time()

        return self.stats
