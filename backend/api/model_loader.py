import tensorflow as tf
import numpy as np
import os

from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2


MAX_SEQ = 20
IMG_SIZE = 128

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "video_classification_model.h5")


def build_model():
    feature_extractor = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    for layer in feature_extractor.layers:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(MAX_SEQ, IMG_SIZE, IMG_SIZE, 3))

    x = layers.TimeDistributed(feature_extractor)(inputs)
    x = layers.TimeDistributed(layers.GlobalAveragePooling2D())(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


model = None

def load_model():
    global model
    if model is None:
        model = build_model()
        model.load_weights(MODEL_PATH)  
    return model


def predict_sequence(sequence):
    model = load_model()
    prediction = model.predict(sequence)
    return float(prediction[0][0])