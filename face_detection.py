#!/usr/bin/env python
# coding: utf-8

from src import detect_faces, show_bboxes
from PIL import Image


import cv2
import numpy as np
import skimage
from skimage import io,transform
from PIL import Image
import matplotlib.pyplot as plt
import os


def video2frame(folder, path):
    cap = cv2.VideoCapture(path)
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")


    i = 0
    stride = 5

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if (i+1) % stride == 0:
                frame = np.rot90(frame,3)
               
                frame = frame[:,:,::-1]
                img = Image.fromarray(frame)
                try:
                    bboxs, landmarks = detect_faces(img)
                    if len(bboxs)> 0:
                        print(bboxs[0][:4])
                        crop = img.crop((bboxs[0][:4]))
                        crop.save(f"{folder}/frame_{i}.jpg")
                except:
                    pass
            i+=1
            # Break the loop
        else: 
            break

ROOT_DIR = "/home/bach/Downloads/face_recognition-20190320T023152Z-001/data_raw/"

list_file = os.listdir(ROOT_DIR)

for i, file in enumerate(list_file):
    folder = str(i)
    if not os.path.exists(folder):
        os.mkdir(folder)
    path = os.path.join(ROOT_DIR, file)
    video2frame(folder, path)
    
    




