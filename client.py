import cv2
import grpc
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

def style_transfer_serving(stub, content, style, resize=None):
    content = np.array(content, dtype=np.float32)
    style = np.array(style, dtype=np.float32)

    if resize:
        content = cv2.resize(content, (512, 512))
        style = cv2.resize(style, (512, 512))

    image_proto = tf.make_tensor_proto(content[np.newaxis, ...] / 255.)
    style_proto = tf.make_tensor_proto(style[np.newaxis, ...] / 255.)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'style'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['placeholder'].CopyFrom(image_proto)
    request.inputs['placeholder_1'].CopyFrom(style_proto)
    resp = stub.Predict(request)
    stylized_image = tf.make_ndarray(resp.outputs['output_0'])[0]
    return stylized_image

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

    file = tf.io.read_file('/home/albert/Downloads/sam_and_nyx/sam_faces/sam_kitchen.jpg')
    content = tf.io.decode_image(file)

    style_transfer_serving(stub, content, style)