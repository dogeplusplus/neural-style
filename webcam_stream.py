import os
import cv2
import time
import numpy as np
import streamlit as st
import tensorflow as tf
from threading import Thread


class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return

            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


def style_transfer_direct(image, style_image, model, resize=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(np.array(image, dtype=np.float32), (512, 512))
    else:
        image = np.array(image, dtype=np.float32)
    image = tf.convert_to_tensor(image[np.newaxis, ...] / 255.)
    resp = model(image, style_image)
    stylized_image = resp[0]
    return stylized_image.numpy()[0]


def get_style_dictionary():
    style_lookup = {}
    for file in os.listdir('assets/template_styles'):
        with open(os.path.join('assets/template_styles', file), 'rb') as f:
            style = f.read()
        style_image = tf.io.decode_image(style)
        style_image = np.array(style_image[np.newaxis, ...], dtype=np.float32) / 255.
        style_image = tf.image.resize(style_image, (256, 256))
        style_image = tf.convert_to_tensor(style_image)
        style_lookup[file] = style_image
    return style_lookup

@st.cache(persist=True)
def get_custom_style(image_bytes):
    style_image = tf.io.decode_image(image_bytes)
    style_image = np.array(style_image[np.newaxis, ...], dtype=np.float32) / 255.
    style_image = tf.image.resize(style_image, (256, 256))
    style_image = tf.convert_to_tensor(style_image)
    return style_image


def main():
    model = tf.saved_model.load('style/1')
    st.title("Neural Style-Transfer Webcam")
    st.subheader('Style Transfer')
    webcam_flag = st.sidebar.checkbox('Enable Webcam', value=False)
    style_flag = st.sidebar.checkbox('Enable Style Transfer', value=False)
    with open('assets/bonk.png', 'rb') as f:
        default_bytes = f.read()
    placeholder_image = st.image(default_bytes)

    style_dictionary = get_style_dictionary()
    style_options = st.sidebar.selectbox(label='Example Styles', options=list(style_dictionary.keys()))
    custom_style = st.sidebar.file_uploader('Upload Style:', type=['.jpg', '.jpeg', '.png'])
    frame_rate = st.text(f'Frames per second: 0')

    if webcam_flag:
        video_capture = WebcamVideoStream(0)
        start = time.time()
        total_frames = 0
        try:
            video_capture.start()
            if custom_style is not None:
                custom_style_bytes = custom_style.getvalue()
                style_image = get_custom_style(custom_style_bytes)
            else:
                style_image = style_dictionary[style_options]

            st.sidebar.subheader("Style Image:")
            st.sidebar.image(np.array(style_image.numpy()[0] * 255., dtype=np.uint8), use_column_width=True)

            while webcam_flag:
                content_image = video_capture.read()
                if style_flag:
                    transfer = style_transfer_direct(content_image, style_image, model)
                    placeholder_image.image(transfer)
                else:
                    placeholder_image.image(content_image, channels='BGR')
                total_frames += 1
                end = time.time()
                frame_rate.text(f'Frames per second: {total_frames / (end - start)}')
        finally:
            video_capture.stop()
            del video_capture


if __name__ == "__main__":
    main()
