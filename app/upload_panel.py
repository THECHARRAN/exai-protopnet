import streamlit as st
from PIL import Image

def render_upload():

    st.subheader("MRI Scan")

    file = st.file_uploader(
        "Upload MRI",
        type=["png","jpg","jpeg"]
    )

    if file:

        img = Image.open(file)

        st.image(img,use_container_width=True)

        return img

    return None