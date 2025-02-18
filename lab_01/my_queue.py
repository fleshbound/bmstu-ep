from typing import List

from request import Request


class Queue:
    requests: List[Request]
    max_size: int

    def __init__(self, max_size: int = -1):
        self.requests = []
        self.max_size = max_size

    def size(self) -> int:
        return len(self.requests)

    def is_inf(self):
        return self.max_size == -1

    def is_empty(self) -> bool:
        return len(self.requests) == 0

    def enqueue(self, r: Request) -> bool:
        """
        :param r: request to be enqueued
        :return: True, if r is enqueued successfully
        """
        if not self.is_inf() and self.size() > self.max_size:
            return False

        self.requests.append(r)
        return True

    def enqueue_head(self, r: Request) -> bool:
        """
        :param r: request to be enqueued (to head)
        :return: True, if r is enqueued successfully
        """
        if not self.is_inf() and self.size() > self.max_size:
            return False

        self.requests.insert(0, r)
        return True

    def dequeue_emergency(self, index: int):
        if not self.is_empty():
            return self.requests.pop(index)

    def dequeue(self) -> Request:
        if not self.is_empty():
            return self.requests.pop(0)

    def clear(self) -> None:
        self.requests.clear()
