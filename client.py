import grpc
import tensorflow as tf
import numpy as np
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

if __name__ == "__main__":
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    channel = grpc.insecure_channel('localhost:8500', options=options)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()

    file = tf.io.read_file('C:\\Users\\doge\\Downloads\\sam.jpg')
    image = tf.io.decode_image(file)

    request.model_spec.name = 'style'
    request.model_spec.signature_name = 'serving_default'
    image_proto = tf.make_tensor_proto(np.array(image, dtype=np.float32)[np.newaxis, ...])
    request.inputs['placeholder'].CopyFrom(image_proto)
    request.inputs['placeholder_1'].CopyFrom(image_proto)
    resp = stub.Predict(request)
    print(resp)




