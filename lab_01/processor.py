from typing import List, Optional

from distribution import Distribution
from queue import Queue
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

    def last_equal_priority_queue_index(self, type: int):
        for i in range(self.queue.size() - 1, -1, -1):
            if self.queue.requests[i].type <= type:
                return i

        return -1

    def receive(self, r: Request) -> None:
        if self.is_busy() and r.type < self.current_request.type:  # 2 > 1 - less prior
            last_more_prior_pos = self.last_equal_priority_queue_index(r.type)

            if last_more_prior_pos < 0:
                self.abort(r.create_time)
                self.queue.enqueue_head(r)
                self.next = r.create_time
            else:
                self.queue.enqueue(r)
        else:
            self.queue.enqueue(r)

        if self.queue.size() > self.max_reached_queue_size:
            self.max_reached_queue_size = self.queue.size()

    def abort(self, cur_sim_time: float):
        self.current_request.calc_remaining_time(cur_sim_time)

        if not self.is_data_losing:
            if self.is_aborting_to_tail:
                self.queue.enqueue(self.current_request)
            else:
                self.queue.enqueue_head(self.current_request)

    def is_busy(self) -> bool:
        return self.current_request is not None

    def next_time_interval(self, type: int) -> float:
        return self.generators[type - 1].generate()

    def start_processing(self, cur_time: float) -> Optional[Request]:
        if self.queue.is_empty():
            return None

        request = self.queue.dequeue()
        request.calc_waiting_time(cur_time)
        self.current_request = request

        return self.current_request

    def end_processing(self):
        self.current_request = None
        self.processed += 1
