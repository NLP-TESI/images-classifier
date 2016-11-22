
# coding: utf-8

# In[8]:

from generate_hist import *

import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import json


# In[9]:

def myFilter(img):
    img_gauss1 = cv2.GaussianBlur(img,(5,5), 1)
    return cv2.GaussianBlur(img_gauss1,(5,5), 1)


# In[10]:

# 5) processing image dataset to generate a feature vec
test_folder = "./img/cifar-10/test"
class_names = os.listdir(test_folder) # there are a folde for each class

# processing train folder
print "PROCESSING TEST FOLDER: "
X = []
y = []
count  = 0
for name in class_names:
    files = os.listdir(test_folder+"/"+name)

    # transform each file into a feature vector
    for file_name in files:
        vec = generate_vector(test_folder+"/"+name+"/"+file_name, filter=myFilter)
        X.append(vec.tolist())

        y_vec = [0.0] * len(class_names)
        y_vec[class_names.index(name)] = 1.0
        y.append(y_vec)

        count += 1

        if count % 1000 == 0:
            print "\r",count, " images processed",

save_as_json(X,y)


# In[ ]:



