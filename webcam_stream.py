import cv2
import tensorflow as tf
import numpy as np
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
    if resize:
        image = cv2.resize(np.array(image, dtype=np.float32), (512, 512))
    else:
        image = np.array(image, dtype=np.float32)
    resp = model(tf.convert_to_tensor(image[np.newaxis, ...] / 255.), tf.convert_to_tensor(style_image))
    stylized_image = resp[0]
    return stylized_image.numpy()[0]


def main():
    model = tf.saved_model.load('style/1')
    video_capture = WebcamVideoStream('http://192.168.0.10:4747/video')
    video_capture.start()

    style_image = np.array(cv2.imread('/home/albert/Downloads/pattern.jpeg'), dtype=np.float32)[np.newaxis, ...] / 255.

    while True:
        content_image = video_capture.read()
        transfer = style_transfer_direct(content_image, style_image, model)
        cv2.imshow('style camera', transfer)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    video_capture.stop()


if __name__ == "__main__":
    main()
