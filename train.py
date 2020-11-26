from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from imutils import paths
from pathlib import Path
from nn.conv.fcheadnet import FCHeadNet
from preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from preprocessing.simpledatasetloader import SimpleDatasetLoader
import numpy as np
import argparse
import os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to dataset")
    ap.add_argument("-m", "--model", required=True, help="path to output model")

    args = vars(ap.parse_args())

    # create image augmentation
    img_aug = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    print("[INFO] Loading dataset...")
    img_paths = list(paths.list_images(args["dataset"]))
    class_names = [Path(p).parts[-2] for p in img_paths]
    class_names = np.unique(class_names)
    print(class_names)
    # print(len(img_paths))

    # init image preprocessing
    aap = AspectAwarePreprocessor(224, 224)
    iap = ImageToArrayPreprocessor()

    sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
    data, labels = sdl.load(img_paths, verbose=-1)
    data = data.astype("float32") / 255.0

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
    print(trainY[0])
    print(testY[0])

    trainY = LabelBinarizer().fit_transform(trainY)
    testY = LabelBinarizer().fit_transform(testY)
    # print(trainY[0])
    # print(testY[0])

    # building model
    base_model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    new_model = FCHeadNet.build(base_model, len(class_names), 256)

    model = Model(inputs=base_model.input, outputs=new_model)

    model.summary()
    plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)

    # freeze base_model layers
    for layer in base_model.layers:
        layer.trainable = False

    print("[INFO] Compiling model...")
    opt = RMSprop(lr=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    
    print("[INFO] Training starts...")
    model.fit(
        img_aug.flow(trainX, trainY, batch_size=32),
        validation_data=(testX, testY),
        epochs=25,
        steps_per_epoch=len(trainX) // 32,
        verbose=1
    )

    print("[INFO] Evaluation after warm up...")
    preds = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=class_names))

    # unfreeze final set of CONV layers in VGG and make it trainable
    # i.e. allow `block_5_conv1` to `block_5_pool` to be trainable
    for layer in base_model.layers[15:]:
        layer.trainable = True

    print("[INFO] Re-compiling model...")
    opt = SGD(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    print("[INFO] Fine tuning model...")
    model.fit(
        img_aug.flow(trainX, trainY, batch_size=32),
        validation_data=(testX, testY),
        epochs=100,
        steps_per_epoch=len(trainX) // 32,
        verbose=1   
    )

    print("[INFO] Evaluation after fine tuning...")
    preds = model.predict(testX)
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=class_names))

    print("[INFO] Saving model...")
    version = "0001"
    model_path = os.path.sep.join([args["model"], version])
    # model.save(args["model"])
    tf.keras.models.save_model(
        model,
        model_path,
        overwrite=True,
        include_optimizer=True,
        signatures=None
    )