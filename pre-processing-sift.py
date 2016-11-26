
# coding: utf-8

# In[ ]:

import cv2
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from sift import *


# In[ ]:

# get the descriptors of all images using SIFT or RootSIFT
images_folder = "./img/cifar-10/test"
class_names = os.listdir(images_folder) # there are a folder for each class

# processing all images
print "PROCESSING TEST FOLDER: "
X = []
y = []
y_label = []
count = 0
for name in class_names:
    files = os.listdir(images_folder+"/"+name)

    # transform each file of image into a list of descriptors
    for file_name in files:
        descs = get_descriptors_from_img(images_folder+"/"+name+"/"+file_name)
        if descs is None:
            continue
        X.append(descs)
        
        y_vec = [0.0] * len(class_names)
        y_vec[class_names.index(name)] = 1.0
        y.append(y_vec)
        
        y_label.append(name)

        count += 1
        print "\r",count, " images processed",


# In[ ]:

# separate in three groups, train, validation and test
seed = 4785
np.random.seed(seed)
np.random.shuffle(X)
np.random.seed(seed)
np.random.shuffle(y)
np.random.seed(seed)
np.random.shuffle(y_label)

total = len(X)

train_size = int(total*0.8) # 80% of data to train
validation_size = int(total*0.1) # 10% of data to validation and test
start_validation = train_size + validation_size

print "train: ", train_size
print "validation: ", validation_size
print "test: ", total - (train_size + validation_size)


data = {
    "train": {
        "X": X[:train_size],
        "y": y[:train_size],
        "label": y_label[:train_size]
    },
    "validation":{
        "X": X[train_size: start_validation],
        "y": y[train_size: start_validation],
        "label": y_label[train_size: start_validation]
    },
    "test":{
        "X": X[start_validation: ],
        "y": y[start_validation: ],
        "label": y_label[start_validation: ]
    }
}


# In[ ]:

# train all descriptors of group of train using k-means
number_of_features = 20 # this value can be change to improve the accuracy
kmeans = KMeans(n_clusters=number_of_features)
descriptors = []
count = 0
for image in data['train']['X']:
    for descriptor in image:
        descriptors.append(descriptor)
        count += 1
        print "\r",count," added",
descriptors = np.array(descriptors)
print "\ntraining k-means with the descriptors"
kmeans.fit(descriptors)
print "done!"


# In[ ]:

def std(data):
    mean = np.mean(data)
    std = np.std(data)
    out = (data - mean)/std
    return out.tolist()


# In[ ]:

# classify all descriptors of all groups using k-means and make a histogram of all groups
for key in data:
    X = []
    for image in data[key]['X']:
        clusters = kmeans.predict(image)
        image_features = [0]*number_of_features
        for cluster in clusters:
            image_features[cluster] += 1
        X.append(std(image_features))
    data[key]['X'] = X


# In[ ]:

# save all groups in a json
with open("hist.json","w+") as out:
    out.write(json.dumps(data))
    out.close()


# In[ ]:



