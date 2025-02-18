from typing import List

from fontTools.merge.util import current_time

from generator import Generator
from lab_01.request import REQUEST_TYPE_ONE, REQUEST_TYPE_TWO
from processor import Processor


class Statistics:
    refused: int
    processed: int
    created: int
    total_time: float
    waiting_times: List[List[float]]
    avg_wait_times: List[float]
    proc_by_type: List[float]
    crea_by_type: List[float]
    exp_lambda_1: float
    exp_lambda_2: float
    exp_mu_1: float
    exp_mu_2: float

    def __init__(self):
        self.refused = 0
        self.processed = 0
        self.created = 0
        self.total_time = 0
        self.avg_wait_times = [0, 0]
        self.waiting_times = [[], []]
        self.proc_by_type = [0, 0]
        self.crea_by_type = [0, 0]
        self.sim_time = 0

    def clear(self):
        self.refused = 0
        self.processed = 0
        self.created = 0
        self.total_time = 0
        self.avg_wait_times = [0, 0]
        self.waiting_times = [[], []]
        self.proc_by_type = [0, 0]
        self.crea_by_type = [0, 0]
        self.sim_time = 0

    def calculate_avg_waiting_time(self, type: int):
        for t in self.waiting_times[type]:
            self.avg_wait_times[type] += t

        self.avg_wait_times[type] /= len(self.waiting_times[type])

    def calculate_avg_waiting_times(self):
        self.calculate_avg_waiting_time(REQUEST_TYPE_ONE)
        self.calculate_avg_waiting_time(REQUEST_TYPE_TWO)

    def calculate_exp_intensities(self):
        self.exp_lambda_1 = self.crea_by_type[REQUEST_TYPE_ONE] / self.sim_time
        self.exp_lambda_2 = self.crea_by_type[REQUEST_TYPE_TWO] / self.sim_time
        self.exp_mu_1 = self.proc_by_type[REQUEST_TYPE_ONE] / self.sim_time
        self.exp_mu_2 = self.proc_by_type[REQUEST_TYPE_TWO] / self.sim_time


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
        cur_sim_time = 0

        for generator in self.generators:
            generator.next = generator.next_time_interval()

        # self.processors[0].next = self.processors[0].next_time_interval(REQUEST_TYPE_ONE)
        elements = self.generators + self.processors

        while self.stats.processed <= num_requests:
            cur_sim_time = self.generators[0].next

            for elem in elements:
                if 0 < elem.next < cur_sim_time:
                    cur_sim_time = elem.next

            for elem in elements:
                if elem.next == cur_sim_time:
                    # print(f"[{cur_sim_time:.2f}]")
                    if elem.machine_type == "processor":
                        # если очередной элемент - ОА
                        if elem.current_request is not None:
                            self.stats.proc_by_type[elem.current_request.type] += 1
                            self.stats.processed += 1
                            elem.end_processing()

                        cur_request = elem.start_processing(cur_sim_time)

                        if cur_request:
                            self.stats.waiting_times[cur_request.type].append(cur_request.waiting_time_interval)
                            elem.next = cur_sim_time + elem.next_time_interval(cur_request.type)
                            # print(f", конец: {elem.next:.2f}")
                        else:
                            elem.next = 0
                    elif elem.machine_type == "generator":
                        # если очередной элемент - генератор
                        processor = elem.generate(cur_sim_time)
                        self.stats.crea_by_type[elem.type] += 1
                        self.stats.created += 1

                        if processor.next == 0:
                            processor.start_processing(cur_sim_time)
                            processor.next = cur_sim_time + processor.next_time_interval(elem.type)
                            # print(f", конец: {processor.next:.2f}")
                        else:
                            # # print(f'{processor.current_request.id}: отклонена')
                            self.stats.refused += 1

                        elem.next = cur_sim_time + elem.next_time_interval()

        self.stats.calculate_avg_waiting_times()
        self.stats.sim_time = cur_sim_time
        self.stats.calculate_exp_intensities()

        return self.stats
