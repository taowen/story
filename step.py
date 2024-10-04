import threading
import queue
from typing import Any, Callable, Tuple
import flow_state

if not flow_state.has_key('message_queue'):
    flow_state.update_key(message_queue = queue.Queue())
message_queue = flow_state.get_key('message_queue')

# Initialize a thread-safe counter
step_counter = 0
counter_lock = threading.Lock()

def get_next_step_id() -> str:
    global step_counter
    with counter_lock:
        step_counter += 1
        return f's{step_counter}'

class StepBeginEvent:
    def __init__(self, step_id: str, func: Callable, args: Tuple, kwargs: dict, caller_step_id: str | None = None):
        self.step_id = step_id
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.caller_step_id = caller_step_id

class StepEndEvent:
    def __init__(self, step_id: str, result: Any):
        self.step_id = step_id
        self.result = result

def step(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        # Generate a unique step_id
        step_id = get_next_step_id()
        
        thread_local = threading.local()
        # Initialize the stack if it doesn't exist
        if not hasattr(thread_local, 'stack'):
            thread_local.stack = []
        
        # Push the current step_id onto the stack
        caller_step_id = thread_local.stack[-1] if thread_local.stack else None
        thread_local.stack.append(step_id)
        
        # Capture the input arguments as a StepData object
        message_queue.put(("step_begin", StepBeginEvent(step_id, func, args, kwargs, caller_step_id)))
        
        # Run the function and capture its output
        output_data = func(*args, **kwargs)
        
        # Send input and output as separate messages to the main thread
        message_queue.put(("step_end", StepEndEvent(step_id, output_data)))
        
        # Pop the step_id from the stack
        thread_local.stack.pop()
        
        return output_data
    return wrapper