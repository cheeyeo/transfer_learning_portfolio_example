from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import imagenet_utils
import tensorflow as tf
import numpy as np
import grpc
import argparse

# TODO: Save this during training into pickle file ?
LABELS = [
    'bluebell',
    'buttercup',
    'coltsfoot',
    'cowslip',
    'crocus',
    'daffodil',
    'daisy',
    'dandelion',
    'fritillary',
    'iris',
    'lilyvalley',
    'pansy',
    'snowdrop',
    'sunflower',
    'tigerlily',
    'tulip',
    'windflower'
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    input_name = "input_1"
    output_name = "dense_1"

    # Process input image
    # img_path = "datasets/images/bluebell/image_0241.jpg"
    img_path = args["image"]
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = imagenet_utils.preprocess_input(img)
    # print(img.shape)

    # Create new GRPC request
    request = PredictRequest()
    request.model_spec.name = "flowers17"
    request.model_spec.signature_name = "serving_default"
    request.inputs[input_name].CopyFrom(tf.make_tensor_proto(img))

    # Send request to server
    channel = grpc.insecure_channel("localhost:8500")
    predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    response = predict_service.Predict(request, timeout=10.0)
    # print(response)

    res = response.outputs[output_name].float_val
    print("[INFO] Raw Prediction Labels: {}".format(res))
    prediction = LABELS[np.argmax(res)]
    print("[INFO] Predicted Label: {}".format(prediction))