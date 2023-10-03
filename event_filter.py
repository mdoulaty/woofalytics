import time


class EventFilter:
    def __init__(
        self,
        min_time_between_events_in_seconds=1,
        max_number_of_events_in_time_t=4,
        time_t_for_max_events_seconds=60,
    ):
        self.min_time_between_events_in_seconds = min_time_between_events_in_seconds
        self.max_number_of_events_in_time_t = max_number_of_events_in_time_t
        self.time_t_for_max_events_seconds = time_t_for_max_events_seconds
        self.event_timestamps = []

    def fire(self):
        current_time = time.time()
        if len(self.event_timestamps) > 0:
            if (
                current_time - self.event_timestamps[-1]
                < self.min_time_between_events_in_seconds
            ):
                print("Cool down period, event not registered.")
                return False

        self.event_timestamps.append(current_time)

        # remove older events
        self.event_timestamps = [
            t
            for t in self.event_timestamps
            if current_time - t <= self.time_t_for_max_events_seconds
        ]
        if len(self.event_timestamps) > self.max_number_of_events_in_time_t:
            print("Max triggers in time T reached")
            return False

        return True
