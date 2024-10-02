import streamlit as st
from step import step
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode
from streamlit_flow.layouts import TreeLayout
import cache
from step import step, edges, nodes
import numpy as np
import cv2
from quantize_color import quantize_color

@step
def demo_func(i):
    demo_func2(9)

@step
def demo_func2(j):
    pass

def main():
    st.set_page_config(layout="wide")

    image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if image is None:
        return
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                st.write(cache.get_cache(selected_id))
            else:
                st.write('result not found')
    
    if should_run and not selected_id:
        quantize_color(image)    

if __name__ == "__main__":
    main()
