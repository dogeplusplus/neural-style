import cv2
import grpc
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def style_transfer_serving(stub, content, style, resize=None):
    content = np.array(content, dtype=np.float32) / 255.
    style = np.array(style, dtype=np.float32) / 255.

    if resize:
        content = cv2.resize(content, (512, 512))
        style = cv2.resize(style, (512, 512))


    image_proto = tf.make_tensor_proto(content[np.newaxis, ...] / 255.)
    style_proto = tf.make_tensor_proto(style[np.newaxis, ...] / 255.)

    stylized_image = hub_module(tf.constant(content[np.newaxis, ...]), tf.constant(style[np.newaxis, ...]))
    # request = predict_pb2.PredictRequest()
    # request.model_spec.name = 'style'
    # request.inputs['placeholder'].CopyFrom(image_proto)
    # request.inputs['placeholder_1'].CopyFrom(style_proto)
    # resp = stub.Predict(request)
    # stylized_image = tf.make_ndarray(resp.outputs['output_0'])[0]
    stylized_image = stylized_image[0] * 255
    stylized_image = np.array(stylized_image, dtype=np.uint8)
    stylized_image = stylized_image
    return stylized_image

if __name__ == "__main__":
    options = [
        ('grpc.max_send_message_length', 200 * 1024 * 1024),
        ('grpc.max_receive_message_length', 200 * 1024 * 1024)
    ]
    # channel = grpc.insecure_channel('localhost:8500', options=options)
    # stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    file = tf.io.read_file('/home/albert/github/neural-style/assets/template_styles/pebbles.jpg')
    style = tf.io.decode_image(file)

    file = tf.io.read_file('/home/albert/Downloads/sam_and_nyx/sam_stairs.jpeg')
    content = tf.io.decode_image(file)

    stub = None
    result = style_transfer_serving(stub, content, style)
    import matplotlib.pyplot as plt
    plt.imshow(result[0])
    plt.show()