import json
import numpy as np

with open('hist.json') as dt:
    hist = json.loads(dt.read())

X_train = np.array(hist['train']['X'])
y_train = np.array(hist['train']['y'])

X_validation = np.array(hist['validation']['X'])
y_validation = np.array(hist['validation']['y'])

X_test = np.array(hist['test']['X'])
y_test = np.array(hist['test']['y'])
