import tensorflow as tf
from tensorflow import keras
import numpy as np
class TestModel(keras.Model):
    def __init__(self):
        super(TestModel, self).__init__()
        self.optimizer = keras.optimizers.Adam()
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.dense1 = keras.layers.Dense(64, activation='relu')
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.dense3 = keras.layers.Dense(10)

    def call(self, inputs):
        output = self.dense1(inputs)
        output = self.dense2(output)
        return self.dense3(output)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)


        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}



(x_tr, y_tr), (x_te, y_te) = keras.datasets.mnist.load_data()
x_tr = x_tr.reshape(60000,784).astype(np.float32) / 255
x_te = x_te.reshape(10000,784).astype(np.float32) / 255


model = TestModel()
model.compile(loss=model.loss, optimizer=model.optimizer, metrics=["acc"], )
model.fit(x_tr,y_tr, epochs=5, batch_size=32, validation_split=0.2)




# ===Test for no shortcuts===

import numpy as np
import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam
# -> optimizer만 이상함... 참조위치가 바뀐듯.. 이럴 땐 그냥 shortcut이 낫겟다.

(x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
x_tr = x_tr.reshape(60000,784).astype(np.float32) / 255
x_te = x_te.reshape(10000,784).astype(np.float32) / 255

input = Input((784))
output = Dense(64, activation=relu)(input)
output = Dense(64, activation=relu)(output)
output = Dense(10)(output)
model = Model(inputs=input, outputs=output, name='test')
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(), metrics=[SparseCategoricalAccuracy()])
model.summary()
model.fit(x_tr, y_tr, epochs=3, batch_size=32, validation_split=0.2)


import threading

a = 0
def add1():
    global a
    a += 1

def add2():
    global a
    a += 10
add1()
a
add2()
a

threading.Thread(target=add1, args=)
