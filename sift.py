# import the necessary packages
import numpy as np
import cv2

class RootSIFT:
    def __init__(self):
        # initialize the SIFT feature extractor
        self.extractor = cv2.xfeatures2d.SIFT_create()

    def compute(self, image, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, descs) = self.extractor.compute(image, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and taking the
        # square-root
        descs /= (descs.sum(axis=1, keepdims=True) + eps)
        descs = np.sqrt(descs)
        #descs /= (np.linalg.norm(descs, axis=1, ord=2) + eps)

        # return a tuple of the keypoints and descriptors
        return (kps, descs)

def get_descriptors_from_img(image_file_name, root_sift=True, filter_img=None):
    # load the image we are going to extract descriptors from and convert
    image = cv2.imread(image_file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if filter_img is not None:
        img = filter_img(image)
        if img is not None:
            image = img
    
    if root_sift:
        # it to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # detect Difference of Gaussian keypoints in the image
        detector = cv2.xfeatures2d.SIFT_create()
        kps = detector.detect(gray)

        # extract RootSIFT descriptors
        rs = RootSIFT()
        (_,descs) = rs.compute(gray, kps)
    else:
        extractor = cv2.xfeatures2d.SIFT_create()
        (_, descs) = extractor.detectAndCompute(image, mask=None)
    
    return descs
