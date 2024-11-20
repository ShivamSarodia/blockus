import json
from collections import namedtuple

Event = namedtuple('Event', [
    'timestamp',
    'name',
    'params',
]) 

def load_events(log_path):
    events = []
    with open(log_path) as f:
        logs = f.readlines()
        for line in logs:
            # Skip lines that are not events.
            if not line.startswith("event | "):
                continue

            _, timestamp, event, params = line.strip().split(" | ")
            events.append((float(timestamp), event, json.loads(params)))

    # Sort events by timestamp.
    events.sort(key=lambda x: x[0])

    # Adjust timestamps to start at 0.
    start_time = events[0][0]
    print(f"Start time: {start_time}")

    return [
        Event(
            timestamp=timestamp - start_time,
            name=event,
            params=params,
        )
        for timestamp, event, params in events
    ]

def filter_events(events, event_name):
    return [event for event in events if event.name == event_name]