import uuid


class Request:
    id: uuid.UUID
    type: int
    create_time: float  # placed to queue simulation time
    waiting_time_interval: float  # wait in queue time interval
    remaining_time_interval: float  # remaining time interval after aborting
    processing_time_interval: float  # current task random processing time interval
    last_abort_time: float
    was_aborted: bool

    def __init__(self, type: int, create_time: float):
        self.id = uuid.uuid4()
        self.type = type
        self.create_time = create_time
        self.waiting_time_interval = 0
        self.remaining_time_interval = 0
        self.processing_time_interval = 0
        self.was_aborted = False
        # print(f'{self.id}: создана ({self.type + 1})')

    def calc_waiting_time(self, cur_sim_time: float):
        self.waiting_time_interval += cur_sim_time - self.create_time

    def set_processing_time_interval(self, processing_time_interval: float):
        self.processing_time_interval = processing_time_interval

    def calc_remaining_time(self, cur_abort_time: float):
        if self.remaining_time_interval == 0:
            self.was_aborted = True
            self.last_abort_time = cur_abort_time
            self.remaining_time_interval = cur_abort_time - self.processing_time_interval
        else:
            self.remaining_time_interval -= cur_abort_time - self.last_abort_time


REQUEST_TYPE_ONE = 0
REQUEST_TYPE_TWO = 1
