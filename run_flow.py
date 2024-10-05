import threading
import queue
from typing import Any, Callable
import flow_state
import random
import streamlit as st
import inspect
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from step import StepBeginEvent, StepEndEvent, message_queue, WorkerEndEvent, wrapped_worker_thread

if not flow_state.has_key('flow_key'):
    flow_state.update_key(flow_key='initial flow key')

if not flow_state.has_key('nodes'):
    flow_state.update_key(nodes={})
nodes = flow_state.get_key('nodes')

if not flow_state.has_key('steps'):
    flow_state.update_key(steps={})
steps = flow_state.get_key('steps')

if not flow_state.has_key('edges'):
    flow_state.update_key(edges={})
edges = flow_state.get_key('edges')

def rerun():
    flowkey = flow_state.get_key('flow_key')
    del st.session_state[flowkey]
    flow_state.update_key(flow_key=f'hackable_flow_{random.randint(0, 1000)}')
    st.rerun()

def _format_node_content(func_name, **kwargs):
    """Formats the content to be displayed in a node."""
    content = f'# {func_name}\n'
    content += '\n'.join([f"* {k}={v}"[:200] for k, v in kwargs.items()])
    return content

def on_step_begin(event: StepBeginEvent):
    kwargs = event.kwargs
    content = _format_node_content(event.func.__name__, **kwargs)
    nodes[event.step_id] = StreamlitFlowNode(id=event.step_id, pos=(0, 0), data={'content': content})
    steps[event.step_id] = {
        'step_begin_event': event
    }
    if event.caller_step_id:
        edge_key = f"{event.caller_step_id}->{event.step_id}"
        if edge_key not in edges:
            edges[edge_key] = StreamlitFlowEdge(edge_key, event.caller_step_id, event.step_id)
    rerun()

def on_step_end(event: StepEndEvent):
    steps[event.step_id]['step_end_event'] = event

def on_worker_end(event: WorkerEndEvent):
    flow_state.update_key(worker_end_event=event)
    rerun()

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
    visualize_value(step['step_begin_event'].kwargs)
    if 'step_end_event' in step:
        if step['step_end_event'].exception:
            st.title('Failed')
            st.markdown(f'```\n{step["step_end_event"].exception}\n```')
        else:
            st.title('Output')
            visualize_value(step['step_end_event'].result)
    else:
        st.title('Output')
        st.write('not end yet')

def visualize_steps():
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        .stImage {
            padding: 8px;
            background-color: #d3d3d3;
            display: inline-block;
        }
        </style>
        """, unsafe_allow_html=True)
    flow_key = flow_state.get_key('flow_key')

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
        else:
            if flow_state.has_key('worker_end_event'):
                if flow_state.get_key('worker_end_event').exception:
                    st.title('Worker Failed')
                    st.markdown(f'```\n{flow_state.get_key("worker_end_event").exception}\n```')
                else:
                    st.write('worker finished')
            else:
                st.write('worker is still working')

def run_flow(worker_thread: Callable):
    if not flow_state.has_key('worker'):
        worker = threading.Thread(target=lambda: wrapped_worker_thread(worker_thread))
        worker.start()
        flow_state.update_key(worker=worker)

    visualize_steps()
    worker = flow_state.get_key('worker')
    while worker.is_alive() or not message_queue.empty():
        try:
            message_type, event = message_queue.get(timeout=0.1)
            if message_type == "step_begin":
                on_step_begin(event)
            elif message_type == "step_end":
                on_step_end(event)
            elif message_type == 'worker_end':
                on_worker_end(event)
            else:
                raise Exception(f'unknown event: {message_type}')
        except queue.Empty:
            pass