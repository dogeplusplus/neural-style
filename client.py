import os
import cv2
import grpc
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
import time


if __name__ == "__main__":
    options = [
        ('grpc.max_send_message_length', 200 * 1024 * 1024),
        ('grpc.max_receive_message_length', 200 * 1024 * 1024)
    ]
    channel = grpc.insecure_channel('localhost:8500', options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    file = tf.io.read_file('/home/albert/Downloads/pebbles.jpg')
    style = tf.io.decode_image(file)

    style_image = cv2.resize(np.array(style, dtype=np.float32), (64, 64))[np.newaxis, ...] / 255.
    style_proto = tf.make_tensor_proto(np.array(style, dtype=np.float32)[np.newaxis, ...] / 255.)

    def style_transfer(stub, image):
        request.model_spec.name = 'style'
        request.model_spec.signature_name = 'serving_default'
        image = cv2.resize(np.array(image, dtype=np.float32), (512, 512))
        image_proto = tf.make_tensor_proto(image[np.newaxis, ...] / 255.)

        request.inputs['placeholder'].CopyFrom(image_proto)
        request.inputs['placeholder_1'].CopyFrom(style_proto)
        resp = stub.Predict(request)
        stylized_image = tf.make_ndarray(resp.outputs['output_0'])[0]
        return stylized_image

    video = cv2.VideoCapture('/home/albert/Downloads/cat_yelling.mp4')
    while video.isOpened():
        ret, frame = video.read()
        styled_image = style_transfer(stub, frame)
        cv2.imshow('cheese', styled_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cv2.destroyAllWindows()
