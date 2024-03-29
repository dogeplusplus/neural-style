import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow_serving.apis import prediction_service_pb2_grpc

from client import style_transfer_serving
import grpc


def main():
    options = [
        ('grpc.max_send_message_length', 200 * 1024 * 1024),
        ('grpc.max_receive_message_length', 200 * 1024 * 1024)
    ]
    channel = grpc.insecure_channel('localhost:8500', options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    st.title("Neural Style-Transfer App")
    col1, col2 = st.beta_columns(2)
    content_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
    style_file = st.sidebar.file_uploader('Upload Style', type=['jpg', 'jpeg', 'png'])
    style_options = st.sidebar.selectbox(label='Example Styles', options=os.listdir('assets/template_styles'))
    col1.subheader('Content Image')
    col2.subheader('Style Transfer')
    st.sidebar.subheader('Style Image')
    show_image = col1.empty()
    show_style = col2.empty()


    style = None
    content = None

    if content_file:
        content = content_file.getvalue()
        show_image.image(content, use_column_width=True)

    if style_file:
        style = style_file.getvalue()
        st.sidebar.image(style, use_column_width=True)
    elif style_options is not None:
        with open(os.path.join('assets/template_styles', style_options), 'rb') as f:
            style = f.read()
        st.sidebar.image(style, use_column_width=True)

    if content_file is not None and style_file is not None:
        content = np.array(Image.open(content_file))
        style = np.array(Image.open(style_file))
        content = cv2.cvtColor(content, cv2.COLOR_BGRA2BGR)
        style = cv2.cvtColor(style, cv2.COLOR_BGRA2BGR)

        content_image = tf.convert_to_tensor(content)
        style_image = tf.image.resize(tf.convert_to_tensor(style), (256, 256))
        with st.spinner('Generating style transfer...'):
            style_transfer = style_transfer_serving(stub, content_image, style_image)
            show_style.image(style_transfer, use_column_width=True)


if __name__ == "__main__":
    main()
