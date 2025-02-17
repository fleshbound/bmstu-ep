from typing import List

from distribution import Distribution
from processor import Processor
from request import Request


class Generator:
    distribution: Distribution
    type: int
    receivers: List[Processor]
    next: float

    def __init__(self, d: Distribution, type: int):
        self.distribution = d
        self.type = type

    def next_time_interval(self):
        return self.distribution.generate()

    def generate(self, cur_sim_time: float):
        new_request = Request(self.type, cur_sim_time)

        receiver_min = self.receivers[0]
        min_q_size = self.receivers[0].queue.size()

        for receiver in self.receivers:
            if receiver.queue.size() < min_q_size:
                min_q_size = receiver.queue.size()
                receiver_min = receiver

        receiver_min.receive(new_request)

        return receiver_min
