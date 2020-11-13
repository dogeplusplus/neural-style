import os
import streamlit as st
import tensorflow as tf
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
    col2.subheader('Style Image')
    show_image = col1.empty()
    show_style = col2.empty()

    st.subheader('Style Transfer')
    show_transfer = st.empty()

    style = None
    content = None

    if content_file:
        content = content_file.getvalue()
        show_image.image(content, use_column_width=True)

    if style_file:
        style = style_file.getvalue()
        show_style.image(style, use_column_width=True)
    elif style_options is not None:
        with open(os.path.join('assets/template_styles', style_options), 'rb') as f:
            style = f.read()
        show_style.image(style, use_column_width=True)

    if content is not None and style is not None:
        content_image = tf.io.decode_image(content)
        style_image = tf.image.resize(tf.io.decode_image(style), (256, 256))
        with st.spinner('Generating style transfer...'):
            style_transfer = style_transfer_serving(stub, content_image, style_image)
            show_transfer.image(style_transfer, use_column_width=True)


if __name__ == "__main__":
    main()
