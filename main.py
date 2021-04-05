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
        sigmas.update({column: sigma2 ** .5})
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
    model.add(k.layers.Dense(1,
                             kernel_initializer="normal",
                             activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy", "mse"])
    return model


def hidden2_model(l2 = .005):
    model = k.Sequential()
    model.add(k.layers.InputLayer(input_shape=24))
    model.add(
        k.layers.Dense(16, kernel_initializer="normal",
                       activation="relu",
                       kernel_regularizer=k.regularizers.l2(l2)))
    model.add(
        k.layers.Dense(8, kernel_initializer="normal",
                       activation="relu",
                       kernel_regularizer=k.regularizers.l2(l2)))
    model.add(k.layers.Dense(1, kernel_initializer="normal",
                             activation="sigmoid",
                             kernel_regularizer=k.regularizers.l2(l2)))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    return model


def runPCA(input, path, dfy):
    pca = PCA(n_components=2)
    principC = pca.fit_transform(input)
    pDf = pd.DataFrame(data=principC, columns=["PC 1", "PC 2"])
    pDf = pd.concat([pDf, dfy], axis=1)

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


def train(inputs, labels, usetb, epochs=300, batchSz=32, val_split=.3):
    if (usetb):
        name = f"CKDnet-{int(time.time())}"
        tensorboard = k.callbacks.TensorBoard(log_dir=f"tblogs\{name}")
        model.fit(inputs, labels, epochs=epochs, verbose=0, callbacks=[tensorboard], batch_size=batchSz,
                  validation_split=val_split)
    else:
        model.fit(inputs, labels, epochs=epochs, verbose=0, callbacks=None, batch_size=batchSz)
    return


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
    acc = float((tp) + tn) / (fn + fp + tn + tp)
    f1 = 2 * ((perc * sens) / (perc + sens))

    return [[sens, spec, perc, npv, acc, f1], confMatrix]


def splitData(df):
    colmeans = dict()  # first, fill missing values
    for col in df.columns:
        colmeans.update({col: colmean(df, col)})
    json1 = json.dumps(colmeans)
    f = open("data/colmeans.json", "w")
    f.write(json1)
    f.close()

    for col in list(df):  # replace missing values with averages
        c = 0
        for val in df[col]:
            if val == -1:
                df[col].iloc[c] = colmeans[col]
            c += 1
    data = shuffle(df)
    data = data.reset_index(drop=True)
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

    return trainx, trainy, testx, testy, dfy, dfx


df = pd.read_csv("data/csv_result-chronic_kidney_disease_full.csv")
df = df.drop(columns="id")
for col in list(df):
    c = 0
    for val in df[col]:
        if val == "no" or val == "abnormal" or val == "notpresent" or val == "poor" or val == "ckd":
            df[col].iloc[c] = 0
        elif val == "yes" or val == "normal" or val == "present" or val == "good" or val == "notckd":
            df[col].iloc[c] = 1
        elif val == "?":
            df[col].iloc[c] = -1
        c += 1

trainx, trainy, testx, testy, dfy, dfx = splitData(df)

# all calculations for entire dataset, insample and oos
x, sigmas, means = normalize(dfx, True)  # all x's in the dataset
x = getXMatrix(x).T
np.save("data/preprocessedInputs", x)  # all x's in and out of sample
y = getYVector(dfy).T  # same for y's
np.save("data/outputs", y)
json2 = json.dumps(sigmas)  # save for use when evaluating in other files for insample performance
f = open("data/sigmas.json", "w")
f.write(json2)
f.close()



# results = train(trainx, trainy, False, epochs=300, tstx=testx, tsty=testy)
# print(f"loss: {results[0]}\taccuracy: {results[1]}\tMSE: {results[2]}")

epochs = {15}
l2s = {0}

for l2 in l2s:
    model = baseline_model()
    for epoch in epochs:
        results = list()
        for i in range(100):
            train(trainx, trainy, False, epochs=epoch) #TODO: check layer counts
            model.save("Saved models/2layerNet.h5")

            result = evaluate(testx, testy)
            # sens, spec, perc, npv, acc, f1 = result[0]
            confMatrix = result[1]
            resultLine = result[0]
            # print(f"Confusion matrix: {confMatrix}")
            # print(
            #        f"sensitivity: {sens}\nspecificity: {spec}\nprecision: {perc}\nNegative Predictive Value: {npv}\nAccuracy: {acc}\nF1 Score: {f1}")

            # results[i] = resultLine
            results.append(resultLine)
            trainx, trainy, testx, testy, dfy, dfx = splitData(df)

        resultDf = pd.DataFrame(results)
        resultDf.to_csv(f"rawresults/baseline{epoch}.csv")

        #for resultLine in results:
            #print(resultLine)

        print(f"finish epoch {epoch} l2 {l2}")

x = normalize(dfx, False)
print(model.summary())
print(runPCA(x, "PCA/PCA.png", dfy))
