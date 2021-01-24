from tensorflow import keras
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model("Saved models/2layerNet.h5")
x = np.load("data/preprocessedInputs.npy")
y = np.load("data/outputs.npy")

oosx = np.load("data/testx.npy")
oosy = np.load("data/testy.npy")


def evaluate(x, y):
    c = 0

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for pred in model.predict(x):
        yHat = round(float(pred))
        gtLabel = int(y[c])
        if yHat == gtLabel and yHat == 1:
            tp += 1
        elif yHat == gtLabel and yHat == 0:
            tn += 1
        elif yHat == 1 and gtLabel == 0:
            fp += 1
        else:
            fn += 1
        c += 1

    confMatrix = [[tp, fn], [fp, tn]]

    sens = float(tp) / (tp + fn)
    spec = float(tn) / (tn + fp)
    perc = float(tp) / (tp + fp)
    npv = float(tn) / (tn + fn)
    acc = float((tp) + tn) / (fn+fp+tn+tp)
    f1 = 2 * ((perc * sens) / (perc + sens))

    return [[sens, spec, perc, npv, acc, f1], confMatrix]


print("------------Insample------------")
results = evaluate(x, y)
sens, spec, perc, npv, acc, f1 = results[0]
confMatrix = results[1]
print(f"Confusion matrix: {confMatrix}")
print(
        f"sensitivity: {sens}\nspecificity: {spec}\nprecision: {perc}\nNegative Predictive Value: {npv}\nAccuracy: {acc}\nF1 Score: {f1}")
print("------------Out of Sample------------")
results2 = evaluate(oosx, oosy)
sens, spec, perc, npv, acc, f1 = results2[0]
confMatrix = results2[1]
print(f"Confusion matrix: {confMatrix}")
print(
        f"sensitivity: {sens}\nspecificity: {spec}\nprecision: {perc}\nNegative Predictive Value: {npv}\nAccuracy: {acc}\nF1 Score: {f1}")
