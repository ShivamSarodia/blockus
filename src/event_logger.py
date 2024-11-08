import time
import json

def log_event(event_name, parameters={}):
    print(
        f"{time.time()} | {event_name} | {json.dumps(parameters)}"
    )