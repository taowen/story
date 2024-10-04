import threading
import queue
import time
from typing import Any, Callable, Tuple
import cache
import random
import streamlit as st
import inspect
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout

if not cache.has_cache('flow_key'):
    cache.update_cache(flow_key='initial flow key')

if not cache.has_cache('nodes'):
    cache.update_cache(nodes={})
nodes = cache.get_cache('nodes')

if not cache.has_cache('steps'):
    cache.update_cache(steps={})
steps = cache.get_cache('steps')

if not cache.has_cache('edges'):
    cache.update_cache(edges={})
edges = cache.get_cache('edges')

if not cache.has_cache('message_queue'):
    cache.update_cache(message_queue = queue.Queue())
message_queue = cache.get_cache('message_queue')

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

# Example usage
@step
def example_function(a: int, b: int) -> int:
    time.sleep(10)
    return a + b

def worker_thread():
    example_function(1, 2)

def rerun():
    flowkey = cache.get_cache('flow_key')
    del st.session_state[flowkey]
    cache.update_cache(flow_key=f'hackable_flow_{random.randint(0, 1000)}')
    st.rerun()

def _format_node_content(func_name, **kwargs):
    """Formats the content to be displayed in a node."""
    content = f'# {func_name}\n'
    content += '\n'.join([f"* {k}={v}"[:200] for k, v in kwargs.items()])
    return content

def on_step_begin(event: StepBeginEvent):
    func = event.func
    args = event.args
    kwargs = event.kwargs
    combined_kwargs = {**dict(zip(inspect.getfullargspec(func).args, args)), **kwargs}
    content = _format_node_content(event.func.__name__, **combined_kwargs)
    nodes[event.step_id] = StreamlitFlowNode(id=event.step_id, pos=(0, 0), data={'content': content})
    steps[event.step_id] = {
        'func': func, 
        'kwargs': combined_kwargs
    }
    if event.caller_step_id:
        edge_key = f"{event.caller_step_id}->{event.step_id}"
        if edge_key not in edges:
            edges[edge_key] = StreamlitFlowEdge(edge_key, event.caller_step_id, event.step_id)
    rerun()

def on_step_end(event: StepEndEvent):
    steps[event.step_id]['result'] = event.result

def visualize_value(value: Any):
    if hasattr(value, 'visualize'):
        value.visualize()
    elif isinstance(value, dict):
        keys = {str(k): k for k, v in value.items()}
        value = {str(k): v for k, v in value.items()}
        kvs = list(value.items())
        if kvs:
            if len(kvs) < 5:
                tabs = st.tabs([k for k, _ in kvs])
                for i in range(len(kvs)):
                    with tabs[i]:
                        visualize_value(keys[kvs[i][0]])
                        visualize_value(kvs[i][1])
            else:
                selected_key = st.selectbox('Select key', [k for k, _ in kvs])
                visualize_value(keys[selected_key])
                visualize_value(value[selected_key])
        else:
            st.write('empty dict')
    elif isinstance(value, list):
        if len(value) < 5:
            tabs = st.tabs([f'[{i}] {v}' for i, v in enumerate(value)])
            for i in range(len(value)):
                with tabs[i]:
                    visualize_value(value[i])
        else:
            selected_index = st.selectbox('Select index', [f'[{i}] {v}' for i, v in enumerate(value)])
            selected_index = int(selected_index.split(' ')[0][1:-1])
            visualize_value(value[selected_index])
    elif isinstance(value, tuple):
        visualize_value(list(value))
    else:
        st.write(value)

def visualize_step(step: dict):
    st.title('Input')
    visualize_value(step['kwargs'])
    st.title('Output')
    if 'result' in step:
        visualize_value(step['result'])
    else:
        st.write('not end yet')

def visualize_steps():
    st.set_page_config(layout="wide")

    flow_key = cache.get_cache('flow_key')

    # Create two columns
    col1, col2 = st.columns(2)

    # Flowchart in the first column
    with col1:
        selected_id = streamlit_flow(flow_key, list(nodes.values()), list(edges.values()), layout=TreeLayout(direction='down'), fit_view=True, height=800, get_node_on_click=True)

    # Node information in the second column
    with col2:
        if selected_id:
            selected_node: StreamlitFlowNode = nodes[selected_id]
            st.markdown(selected_node.data['content'])
            visualize_step(steps[selected_id])
    
def main_thread(worker_thread: Callable):
    if not cache.has_cache('worker'):
        worker = threading.Thread(target=worker_thread)
        worker.start()
        cache.update_cache(worker=worker)

    visualize_steps()
    worker = cache.get_cache('worker')
    while worker.is_alive() or not message_queue.empty():
        try:
            message_type, event = message_queue.get(timeout=0.1)
            if message_type == "step_begin":
                on_step_begin(event)
            elif message_type == "step_end":
                on_step_end(event)
            else:
                raise Exception(f'unknown event: {message_type}')
        except queue.Empty:
            pass

if __name__ == "__main__":
    main_thread(worker_thread)
