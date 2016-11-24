import json
import numpy as np

with open('hist.json') as dt:
    hist = json.loads(dt.read())

X_train = np.array(hist['train']['X'])
y_train = np.array(hist['train']['y'])
label_train = np.array(hist['train']['label'])

X_validation = np.array(hist['validation']['X'])
y_validation = np.array(hist['validation']['y'])
label_validation = np.array(hist['validation']['label'])

X_test = np.array(hist['test']['X'])
y_test = np.array(hist['test']['y'])
label_test = np.array(hist['test']['label'])
