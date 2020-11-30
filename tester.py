from tensorflow import keras
import numpy as np
import random
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model("Saved models/2layerNet.h5")
x = np.load("data/preprocessedInputs.npy")
y = np.load("data/outputs.npy")
index = int(random.random()*400)
inputs = x[index]
inputs = np.reshape(inputs, (24, 1)).T
expected = y[index]
expected = int(expected)

time1 = time.process_time()
prob = float(model.predict(inputs))
time2 = time.process_time()
print(time2-time1)

if prob>.5:
    print(f"The model believes that patient {index} has CKD wp {prob}")
else:
    print(f"The model believes that patient {index} does not have CKD wp {1-prob}")

if expected == 1:
    print(f"Patient {index} has CKD")
else:
    print(f"Patient {index} does not have CKD")