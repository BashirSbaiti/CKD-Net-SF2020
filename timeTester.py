from tensorflow import keras
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model("Saved models/2layerNet.h5")
x = np.load("data/preprocessedInputs.npy")
y = np.load("data/outputs.npy")


def getAvgPredTime(size, timesToSample=15):  # TODO: maybe add performance tracking back in
    runs = list()
    for i in range(timesToSample):
        index = int(random.random() * (400 - (size - 1)))
        inputs = x[index: index + size]

        time1 = time.process_time()
        prob = model.predict(inputs)
        time2 = time.process_time()
        runs.append(((time2 - time1) * 1000) / size)

    total = 0.
    count = 0.
    for tim in runs:
        total += tim
        count += 1.

    return total / count


sizes = list()
avgTimes = list()
for i in range(1, 100):
    avgTimes.append(getAvgPredTime(i))
    sizes.append(i)

fig = plt.figure(figsize=(8, 8))  # TODO: Clean up this matplotlib code (PCR too)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Number of Patients being Processed", fontsize=15)
ax.set_ylabel("Average time to Predict one Patient (ms/patient)", fontsize=15)
ax.set_title("Average time to Predict one Patient vs. Number of Patients being Processed", fontsize=14.5)
ax.scatter(sizes, avgTimes, s=50)
ax.grid()
plt.savefig("timegraphs/timegraph.png")

print(getAvgPredTime(1))
