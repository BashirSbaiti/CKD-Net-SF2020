import numpy as np
import pandas as pd
from tensorflow import keras as k
from sklearn.utils import shuffle
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def findMax(df, column):
    max = -2147483648
    for val in df[column]:
        val = float(val)
        if val != -1:
            if val > max:
                max = val
    return max


def normalize(df, returnParams=False):  # prepossesses data so that sd=1, mean=0
    means = dict()
    sigmas = dict()
    for column in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']:
        total = 0.
        count = 0.
        for i, val in enumerate(df[column]):
            if float(val) != -1:
                total += float(val)
                count += 1.
        mean = total / count
        sigma2 = 0
        count2 = 0
        for i, val in enumerate(df[column]):
            if float(val) != -1:
                df[column].iloc[i] = float(val) - mean
                sigma2 += (float(val) - mean) * (float(val) - mean)
                count2 += 1
        sigma2 = sigma2 / count2
        for i, val in enumerate(df[column]):
            if float(val) != -1:
                df[column].iloc[i] = float(val) / (sigma2 ** .5)
        means.update({column: mean})
        sigmas.update({column: sigma2**.5})
    if returnParams:
        return df, sigmas, means
    return df


def basicNormalize(df):  # makes every number 0-1 by dividing each column by its max
    for column in df.columns:
        max = findMax(df, column)
        for i, val in enumerate(df[column]):
            if float(val) != -1:
                df[column].iloc[i] = float(val) / max
    return df


def getVector(df, rowNumber):
    arr = np.zeros(len(df.iloc[rowNumber]), dtype=float)
    c = 0
    for val in df.iloc[rowNumber]:
        arr[c] = val
        c += 1
    arr = np.reshape(arr, (1, len(arr)))
    return arr


def getXMatrix(df):
    for i in range(len(df.iloc[:, 0])):
        if i == 0:
            arr = getVector(df, i).T
        else:
            arr = np.append(arr, getVector(df, i).T, axis=1)
    return arr


def colmean(df, col):
    total = 0
    count = 0
    for val in df[col]:
        if (val != -1):
            total += float(val)
            count += 1
    return total / count


def getYVector(df):
    arr = df.to_numpy()
    arr = np.reshape(arr, (len(arr), 1))
    return arr.T


def baseline_model():
    model = k.Sequential()
    model.add(k.layers.InputLayer(input_shape=24))
    model.add(k.layers.Dense(10, kernel_initializer="normal", activation="relu"))
    model.add(k.layers.Dense(1, kernel_initializer="normal", activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])
    return model


def baseline_old():
    model = k.Sequential()
    model.add(k.layers.Dense(10, input_dim=24, kernel_initializer="normal", activation="relu"))
    model.add(k.layers.Dense(1, kernel_initializer="normal", activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])
    return model


def hidden2_model():
    model = k.Sequential()
    model.add(k.layers.InputLayer(input_shape=24))
    model.add(
        k.layers.Dense(16, kernel_initializer="normal", activation="relu", kernel_regularizer=k.regularizers.l2(0.06)))
    model.add(
        k.layers.Dense(6, kernel_initializer="normal", activation="relu", kernel_regularizer=k.regularizers.l2(0.06)))
    model.add(k.layers.Dense(1, kernel_initializer="normal", activation="sigmoid",
                             kernel_regularizer=k.regularizers.l2(0.06)))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", "mse"])
    return model


def runPCA(input, path):
    pca = PCA(n_components=2)
    principC = pca.fit_transform(input)
    pDf = pd.DataFrame(data=principC, columns=["PC 1", "PC 2"])
    pDf = pd.concat([pDf, data["class"]], axis=1)

    # matplotlib code
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principle Component 1", fontsize=15)
    ax.set_ylabel("Principle Component 2", fontsize=15)
    ax.set_title("2 Component PCA", fontsize=20)

    tgts = [0, 1]
    colors = ["r", "b"]

    for tgt, color in zip(tgts, colors):
        keep = pDf["class"] == tgt
        ax.scatter(pDf.loc[keep, "PC 1"], pDf.loc[keep, "PC 2"], c=color, s=50)
    ax.legend(tgts)
    ax.grid()
    plt.savefig(path)

    tot = 0
    for val in pca.explained_variance_ratio_:
        tot += val
    return tot


def train(inputs, labels, usetb, epochs=300, batchSz=32, val_split=.3, tstx=0, tsty=0):
    if (usetb):
        name = f"CKDnet-{int(time.time())}"
        tensorboard = k.callbacks.TensorBoard(log_dir=f"tblogs\{name}")
        model.fit(inputs, labels, epochs=epochs, verbose=0, callbacks=[tensorboard], batch_size=batchSz,
                  validation_split=val_split)
        metrics = [0, 0, 0]
    else:
        model.fit(inputs, labels, epochs=epochs, verbose=0, callbacks=None, batch_size=batchSz)
        metrics = model.evaluate(tstx, tsty)
    return metrics


df = pd.read_csv("data/csv_result-chronic_kidney_disease_full.csv")
data = shuffle(df)
data = data.reset_index(drop=True)
data = data.drop(columns="id")
for col in list(data):
    c = 0
    for val in data[col]:
        if val == "no" or val == "abnormal" or val == "notpresent" or val == "poor" or val == "ckd":
            data[col].iloc[c] = 0
        elif val == "yes" or val == "normal" or val == "present" or val == "good" or val == "notckd":
            data[col].iloc[c] = 1
        elif val == "?":
            data[col].iloc[c] = -1
        c += 1

colmeans = dict()
for col in data.columns:
    colmeans.update({col: colmean(data, col)})

json1 = json.dumps(colmeans)
f = open("data/colmeans.json", "w")
f.write(json1)
f.close()

for col in list(data):
    c = 0
    for val in data[col]:
        if val == -1:
            data[col].iloc[c] = colmeans[col]
        c += 1

# proper slicing
traindf = data.iloc[:280]
testdf = data.iloc[280:]
traindfy = traindf['class']
testdfy = testdf['class']
traindfx = traindf.drop(columns=["class"])
testdfx = testdf.drop(columns=["class"])

trainx = getXMatrix(normalize(traindfx)).T  # (280, 24), Keras like to have m, nx for whatever reason
testx = getXMatrix(normalize(testdfx)).T  # (120, 24)
trainy = getYVector(traindfy).T  # (280, 1)
testy = getYVector(testdfy).T  # (120, 1)
np.save("data/testx", testx)
np.save("data/testy", testy)

# keras automatically does validation split for tb
dfy = data['class']
dfx = data.drop(columns=["class"])

x, sigmas, means = normalize(dfx, True)
x = getXMatrix(x).T
np.save("data/preprocessedInputs", x)
y = getYVector(dfy).T
np.save("data/outputs", y)
json2 = json.dumps(sigmas)
f = open("data/sigmas.json", "w")
f.write(json2)
f.close()

model = hidden2_model()

results = train(trainx, trainy, False, tstx=testx, tsty=testy)
print(f"loss: {results[0]}\taccuracy: {results[1]}\tMSE: {results[2]}")
model.save("Saved models/2layerNet.h5")

print(runPCA(x, "PCA/pcareal.png"))
