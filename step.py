import streamlit as st
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from get_hash_key import get_hash_key
import cache
import random
import functools
from execute_step import execute_step

if not cache.has_cache('flow_key'):
    cache.update_cache(flow_key='initial flow key')

if not cache.has_cache('nodes'):
    cache.update_cache(nodes={})
nodes = cache.get_cache('nodes')

if not cache.has_cache('edges'):
    cache.update_cache(edges={})
edges = cache.get_cache('edges')

if not cache.has_cache('stack'):
    cache.update_cache(stack=[])
stack = cache.get_cache('stack')

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

def step(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hash_key, kwargs = get_hash_key(func, args, kwargs)
        updated = False
        if hash_key not in nodes:
            content = _format_node_content(func.__name__, **kwargs)
            nodes[hash_key] = StreamlitFlowNode(id=hash_key, pos=(0, 0), data={'content': content})
            updated = True
        if stack:
            edge_key = f"{stack[-1]}->{hash_key}"
            if edge_key not in edges:
                edges[edge_key] = StreamlitFlowEdge(edge_key, stack[-1], hash_key)
                updated = True
        if updated:
            cache.update_cache(continue_step_data={
                "func": func,
                "args": args,
                "kwargs": kwargs
            })
            rerun()
        raise Exception('should not enter here')     
    return wrapper

def continue_step():
    data = cache.get_cache('continue_step_data')
    hash_key, _ = get_hash_key(**data)
    stack.append(hash_key)
    try:
        step_dict = execute_step(**data)
        result = step_dict['result']
        return result
    finally:
        stack.pop()
    