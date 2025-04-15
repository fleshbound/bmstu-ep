from typing import List

from lab_03.distribution import Distribution
from lab_03.processor import Processor
from lab_03.request import Request


class Generator:
    distribution: Distribution
    type: int
    receivers: List[Processor]
    next: float
    machine_type: str = "generator"

    def __init__(self, d: Distribution, type: int, num_requests: int = 3000):
        self.distribution = d
        self.type = type

    def next_time_interval(self):
        return self.distribution.generate()

    def generate(self, cur_sim_time: float):
        # if self.n_requests <= 0:
        #     return None
        # self.n_requests -= 1

        new_request = Request(self.type, cur_sim_time)

        receiver_min = self.receivers[0]
        min_q_size = self.receivers[0].queue.size()

        for receiver in self.receivers:
            if receiver.queue.size() < min_q_size:
                min_q_size = receiver.queue.size()
                receiver_min = receiver

        receiver_min.receive(new_request)

        return receiver_min
