import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
import shutil, random, os


def do_this(filePath,name,dest):
    
    img=cv2.imread(filePath)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17,170,255)) 
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
    #converting from gbr to YCbCr color space
#    img_YCrCb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#    #skin color range for hsv color space 
#    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135)) 
#    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    
#    #merge skin detection (YCbCr and hsv)
#    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
#    global_mask=cv2.medianBlur(global_mask,3)
#    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4,4), np.uint8))
    
    
    HSV_result = cv2.bitwise_not(HSV_mask)
#    YCrCb_result = cv2.bitwise_not(YCrCb_mask)
#    global_result=cv2.bitwise_not(global_mask)
    
    fullpath= dest+'/'+name
    cv2.imwrite(fullpath,HSV_result)
#    some_string = output + "_ycrcb.jpg"
#    cv2.imwrite(some_string,YCrCb_result)
#    some_string = output + "_global.jpg"
#    cv2.imwrite(some_string,global_result)
    
    
#Set source directory and destination directories
dirpath = './half'
destDirectory = './ra/'

#Get List of sub directories
subdirs = [os.path.join(dirpath, o) for o in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath,o))]
i=0
while(i<len(subdirs)):
    res=subdirs[i][subdirs[i].rindex('/')+1:]
    newDest = destDirectory+res
    oldDir = dirpath+'/'+res
#    print(oldDir)
    os.mkdir(newDest)
    filenames = os.listdir(oldDir)
    for fname in filenames:
        srcpath = os.path.join(oldDir, fname)
#        print(srcpath)
        do_this(srcpath,fname,newDest)
#        print(srcpath)
#        shutil.copy(srcpath, newDest)
    i+=1
print(len(subdirs))

