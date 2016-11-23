import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import json

# retorna uma image com apenas um dos canais
def get_img_channel(img, channel):

    img_copy = np.copy(img)

    if channel == "r":
        img_copy[:,:,1] = 0
        img_copy[:,:,2] = 0

    elif channel == "g":
        img_copy[:,:,0] = 0
        img_copy[:,:,2] = 0

    elif channel == "b":
        img_copy[:,:,0] = 0
        img_copy[:,:,1] = 0

    return img_copy


# generate histogram as a np.array
def hist(img):

    R = get_img_channel(img, "r").flatten()
    G = get_img_channel(img, "g").flatten()
    B = get_img_channel(img, "b").flatten()

    hist_R,_ = np.histogram(R, bins=256)
    hist_G,_ = np.histogram(G, bins=256)
    hist_B,_ = np.histogram(B, bins=256)

    return hist_R, hist_G, hist_B

def hist_gray(img):
    hist,_ = np.histogram(np.array(img).flatten(), bins=256)
    return hist


# plot images
def draw_hist(img):

    fig, subs = plt.subplots(4,2)
    subs[0][0].imshow(img)
    subs[0][1].axis('off')

    R = get_img_channel(img,'r')
    G = get_img_channel(img,'g')
    B = get_img_channel(img,'b')

    subs[1][0].imshow(R)
    subs[1][1].hist(img[:,:,0].flatten(),np.arange(0,256))
    subs[1][1].set_xlim([0,256])


    subs[2][0].imshow(G)
    subs[2][1].hist(img[:,:,1].flatten(),np.arange(0,256))
    subs[2][1].set_xlim([0,256])


    subs[3][0].imshow(B)
    subs[3][1].hist(img[:,:,2].flatten(),np.arange(0,256))
    subs[3][1].set_xlim([0,256])
    plt.show()

# standardize a list
def standardize(data):
    data[0] = 0
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean)/std

# create a feature vector concatenating each image
def generate_vector(img_path, filter=None, togray=False):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if filter is not None:
        img = filter(img)
    if not togray:
        hist_R, hist_G, hist_B = hist(img)
        feature_vec = np.hstack( [standardize(hist_R), standardize(hist_G), standardize(hist_B)])
        return feature_vec.tolist()
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return standardize(hist_gray(img)).tolist()

def save_as_json(X, y, seed=42):
    # randomizing positions
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)


    # spliting the dataset in thee groups
    outputjson = {
        "train": {
            "X": X[:8000],
            "y": y[:8000]
        },
        "validation":{
            "X": X[8000: 9000],
            "y": y[8000: 9000]
        },
        "test":{
            "X": X[9000: ],
            "y": y[9000: ]
        }
    }

    with open("hist.json","w+") as out:
        out.write(json.dumps(outputjson))
        out.close()
