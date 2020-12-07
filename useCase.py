from tensorflow import keras
import numpy as np
import random
import pandas as pd
import os
import json
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = keras.models.load_model("Saved models/2layerNet.h5")
cols = pd.read_csv("data/csv_result-chronic_kidney_disease_full.csv").drop(columns=["id", "class"]).columns
colmeans = json.load(open("data/colmeans.json"))
sigmas = json.load(open("data/sigmas.json"))

data = list()

for col in cols:
    inp = float(input(f"Please input the {col} of the patient. Input -1 for missing data.\n"))
    if inp == -1:
        data.append(colmeans[col])
    else:
        data.append(inp)

for i in range(len(data)):
    if cols[i] in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']:
        data[i] = data[i]-colmeans[cols[i]]
        data[i] = data[i]/float(sigmas[cols[i]])

data = np.array(data)
data = np.reshape(data, [1, 24])

time1 = time.process_time()
prob = float(model.predict(data))
time2 = time.process_time()
print(f"The model took {(time2 - time1) * 1000} ms to make the prediction")

print(data)
print(prob)
if prob > .5:
    print(f"The model believes that the patient has CKD wp {prob}")
else:
    print(f"The model believes that the patient does not have CKD wp {1 - prob}")

