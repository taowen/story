import threading
import queue
from typing import Any, Callable, Tuple
import flow_state
import traceback
import inspect

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
    def __init__(self, step_id: str, func: Callable, kwargs: dict, caller_step_id: str | None = None):
        self.step_id = step_id
        self.func = func
        self.kwargs = kwargs
        self.caller_step_id = caller_step_id

class StepEndEvent:
    def __init__(self, step_id: str, result: Any, exception: str):
        self.step_id = step_id
        self.result = result
        self.exception = exception

thread_local = threading.local()

def step(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        # Generate a unique step_id
        step_id = get_next_step_id()
        
        # Initialize the stack if it doesn't exist
        if not hasattr(thread_local, 'stack'):
            thread_local.stack = []
        
        # Push the current step_id onto the stack
        caller_step_id = thread_local.stack[-1] if thread_local.stack else None
        thread_local.stack.append(step_id)
        
        combined_kwargs = {**dict(zip(inspect.getfullargspec(func).args, args)), **kwargs}
        message_queue.put(("step_begin", StepBeginEvent(step_id, func, combined_kwargs, caller_step_id)))
        try:        
            result = func(**combined_kwargs)
            message_queue.put(("step_end", StepEndEvent(step_id, result, '')))
            return result
        except:
            message_queue.put(("step_end", StepEndEvent(step_id, None, traceback.format_exc())))
            raise
        finally:
            thread_local.stack.pop()
    return wrapper