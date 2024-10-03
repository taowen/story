import streamlit as st
from step import step
from execute_step import Step
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode
from streamlit_flow.layouts import TreeLayout
import cache
from step import step, edges, nodes
import numpy as np
import cv2
from quantize_color import quantize_color
from data_types import ColorImage
import typing

@step
def load_image():
    image = cv2.imread('quantized_mountain.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ColorImage(image)
    return image

@step
def demo():
    image = load_image()
    quantize_color(image)

def visualize_value(value: typing.Any):
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
    else:
        st.write(value)

def visualize_step(step: Step):
    st.title('Input')
    visualize_value(step['kwargs'])
    st.title('Output')
    visualize_value(step['result'])

def main():
    st.set_page_config(layout="wide")

    flow_key = cache.get_cache('flow_key')
    should_run = flow_key in st.session_state

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
            if cache.has_cache(selected_id):
                visualize_step(cache.get_cache(selected_id))
            else:
                st.write('result not found')
    
    if should_run and not selected_id:
        demo()

if __name__ == "__main__":
    main()
