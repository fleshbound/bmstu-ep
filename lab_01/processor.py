from typing import List, Optional

from distribution import Distribution
from lab_01.request import REQUEST_TYPE_ONE
from my_queue import Queue
from request import Request


class Processor:
    current_request: Optional[Request]
    is_data_losing: bool
    generators: List[Distribution]
    queue: Queue
    max_reached_queue_size: int
    processed: int
    next: float
    is_aborting_to_tail: bool
    machine_type: str = "processor"

    def __init__(self, generators: List[Distribution], queue_size: int = -1,
                 is_aborting_to_tail: bool = False, is_data_losing: bool = False):
        self.current_request = None
        self.generators = generators
        self.queue = Queue(max_size=queue_size)
        self.max_reached_queue_size = 0
        self.is_aborting_to_tail = is_aborting_to_tail
        self.is_data_losing = is_data_losing
        self.next = 0
        self.processed = 0

    def first_high_priority_queue_index(self):
        for i in range(0, self.queue.size(), 1):
            if self.queue.requests[i].type == REQUEST_TYPE_ONE:
                return i

        return -1

    def receive(self, r: Request) -> None:
        if self.is_busy() and r.type < self.current_request.type:  # 2 > 1 - less prior
            high_priority_index = self.first_high_priority_queue_index()

            if high_priority_index > 0:
                self.queue.enqueue(r)
            else:
                self.abort(r.create_time)
                self.queue.enqueue_head(r)
                self.next = r.create_time
        else:
            self.queue.enqueue(r)
        # print(f'{r.id}: принята')

        if self.queue.size() > self.max_reached_queue_size:
            self.max_reached_queue_size = self.queue.size()

    def abort(self, cur_sim_time: float):
        self.current_request.calc_remaining_time(cur_sim_time)
        # print(f'{self.current_request.id}: аборт')

        if not self.is_data_losing:
            if self.is_aborting_to_tail:
                self.queue.enqueue(self.current_request)
            else:
                self.queue.enqueue_head(self.current_request)

        self.current_request = None

    def is_busy(self) -> bool:
        return self.current_request is not None

    def next_time_interval(self, type: int) -> float:
        return self.generators[type - 1].generate()

    def start_processing(self, cur_time: float) -> Optional[Request]:
        if self.queue.is_empty():
            return None

        high_priority_index = self.first_high_priority_queue_index()

        if high_priority_index > 0:
            request = self.queue.dequeue_emergency(high_priority_index)
        else:
            request = self.queue.dequeue()

        # print(f'{request.id}: начало обслуживания', end="")
        request.calc_waiting_time(cur_time)
        self.current_request = request

        return self.current_request

    def end_processing(self):
        # print(f'{self.current_request.id}: конец обслуживания')
        self.current_request = None
        self.processed += 1
