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

stack = []

def rerun():
    flowkey = cache.get_cache('flow_key')
    del st.session_state[flowkey]
    cache.update_cache(flow_key=f'hackable_flow_{random.randint(0, 1000)}')
    st.rerun()

def step(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hash_key, kwargs = get_hash_key(func, args, kwargs)
        updated = False
        if hash_key not in nodes:
            content = f'# {func.__name__}\n'
            content += '\n'.join([f"* {k}={v}" for k, v in kwargs.items()])
            nodes[hash_key] = StreamlitFlowNode(id=hash_key, pos=[0, 0], data={'content': content})
            updated = True
        if stack:
            edge_key = f"{stack[-1]}->{hash_key}"
            if edge_key not in edges:
                edges[edge_key] = StreamlitFlowEdge(edge_key, stack[-1], hash_key)
                updated = True
        if updated:
            print('re-run')
            rerun()
        stack.append(hash_key)
        try:
            step_dict = execute_step(func, args, kwargs)
            result = step_dict['result']
            return result
        finally:
            stack.pop()       
    return wrapper