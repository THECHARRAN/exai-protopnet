import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from upload_panel import render_upload
from analysis_panel import render_analysis


st.set_page_config(
    page_title="NeuroVision AI",
    layout="wide"
)

left, right = st.columns([1, 1])

with left:
    uploaded = render_upload()

with right:
    render_analysis(uploaded)