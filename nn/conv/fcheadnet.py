from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class FCHeadNet:
    @staticmethod
    def build(base_model, classes, D):
        new_head = base_model.output
        new_head = Flatten(name="flatten")(new_head)
        new_head = Dense(D, activation="relu")(new_head)
        new_head = Dropout(0.5)(new_head)

        # add softmax layer
        new_head = Dense(classes, activation="softmax")(new_head)

        return new_head