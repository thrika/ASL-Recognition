import numpy as np
import cv2
from skimage.filters import gabor
from math import pi
import shutil, random, os


def get_gabor_feature(image, name):
    classify = np.array([])
    label = np.array([])
    for i in range(0, 5, 1):

        for j in range(0, 8, 1):
            filt_real = gabor(image, frequency=(i + 1) / 10.0, theta=j * pi / 8)[0]
            filt_imag = gabor(image, frequency=(i + 1) / 10.0, theta=j * pi / 8)[1]
            res = filt_real * filt_real + filt_imag * filt_imag
            res_mean = np.mean(res)
            classify = np.append(classify, res_mean)
            label = np.append(label, name)
    allVectors.append(classify)
    allLabels.append(name)


dirpath = './randomTrain/'
subdirs = [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, o))]
i = 0

j = 0

while (i < len(subdirs)):
    allVectors = []
    allLabels = []
    print(i)
    res = subdirs[i][subdirs[i].rindex('/') + 1:]
    oldDir = dirpath + res
    filenames = os.listdir(oldDir)
    for fname in filenames:
        if (fname != '.DS_Store'):
            print(j)
            srcpath = os.path.join(oldDir, fname)
            print(srcpath)
            img = cv2.imread(srcpath, 0)
            get_gabor_feature(img, res)
            j += 1

        np.save('gabor_features/gabor'+res+'.npy', np.asarray(allVectors))
        np.save('gabor_features/label'+res+'.npy', np.asarray(allLabels))

    i += 1