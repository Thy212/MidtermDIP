#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import os
# Read the original image
img = cv2.imread(r'C:\Users\zhang\Desktop\New folder\input1.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#name
def saveimg(filename, image):
    directory = r'C:\Users\zhang\Desktop\New folder'
    os.chdir(directory)
    cv2.imwrite(filename, image)
    
def ay():
    lower_y = np.array([25, 155, 155])
    upper_y = np.array([50, 255, 255])
    masky = cv2.inRange(hsv, lower_y, upper_y)
    resulty = cv2.bitwise_and(img, img, mask = masky)
    filenamey = 'outputyellow.jpg'
    cv2.imshow('yellow', resulty)
    saveimg(filenamey, resulty) 
    
def ao():
    lower_o = np.array([0, 155, 155])
    upper_o = np.array([25, 255, 255])
    masko = cv2.inRange(hsv, lower_o, upper_o)
    resulto = cv2.bitwise_and(img, img, mask = masko)
    filenameo = 'outputorange.jpg'
    cv2.imshow('orange', resulto)
    saveimg(filenameo, resulto) 
    
def ar():
    lower_r = np.array([155, 155, 155])
    upper_r = np.array([180, 255, 255])
    maskr = cv2.inRange(hsv, lower_r, upper_r)
    resultr = cv2.bitwise_and(img, img, mask = maskr)
    filenamer = 'outputred.jpg'
    cv2.imshow('red', resultr)
    saveimg(filenamer, resultr) 

def ab():
    lower_b = np.array([100, 155, 155])
    upper_b = np.array([155, 255, 255])
    maskb = cv2.inRange(hsv, lower_b, upper_b)
    resultb = cv2.bitwise_and(img, img, mask = maskb)
    filenameb = 'outputblue.jpg'
    cv2.imshow('blue', resultb)
    saveimg(filenameb, resultb) 

def ag():
    lower_g = np.array([55, 155, 100])
    upper_g = np.array([100, 255, 255])
    maskg = cv2.inRange(hsv, lower_g, upper_g)
    resultg = cv2.bitwise_and(img, img, mask = maskg)
    filenameg = 'outputgreen.jpg'
    cv2.imshow('green', resultg)
    saveimg(filenameg, resultg) 
    
def ap():
    lower_v = np.array([100, 155, 100])
    upper_v = np.array([155, 255, 155])
    maskv = cv2.inRange(hsv, lower_v, upper_v)
    resultv = cv2.bitwise_and(img, img, mask = maskv)
    filenamev = 'outputviolet.jpg'
    cv2.imshow('violet', resultv)
    saveimg(filenamev, resultv)
    
def b1():
    img[np.where((img==[1,237,254]).all(axis=2))]=[0,255,0]
    filename1 ='Image1.jpg'
    cv2.imshow('Image1', img)
    saveimg(filename1, img)
    
def b2():
    img[np.where((img==[33,105,229]).all(axis=2))]=[0,255,0]
    filename2 ='Image2.jpg'
    cv2.imshow('Image2', img)
    saveimg(filename2, img)
    
def b3():
    img[np.where((img==[83,0,223]).all(axis=2))]=[0,255,0]
    filename3 ='Image3.jpg'
    cv2.imshow('Image3', img)
    saveimg(filename3, img)
    
def b4():
    img[np.where((img==[193,113,0]).all(axis=2))]=[0,255,0]
    filename4 ='Image4.jpg'
    cv2.imshow('Image4', img)
    saveimg(filename4, img)
    
def b5():
    img[np.where((img==[60,154,0]).all(axis=2))]=[0,255,0]
    filename5 ='Image5.jpg'
    cv2.imshow('Image5', img)
    saveimg(filename5, img)
    
def b6():
    img[np.where((img==[123,32,111]).all(axis=2))]=[0,255,0]
    filename6 ='Image6.jpg'
    cv2.imshow('Image6', img)
    saveimg(filename6, img)
    
def c():
    img[np.where((img==[1,237,254]).all(axis=2))]=[0,255,0]
    img[np.where((img==[33,105,229]).all(axis=2))]=[0,255,0]
    img[np.where((img==[83,0,223]).all(axis=2))]=[0,255,0]
    img[np.where((img==[193,113,0]).all(axis=2))]=[0,255,0]
    img[np.where((img==[60,154,0]).all(axis=2))]=[0,255,0]
    img[np.where((img==[123,32,111]).all(axis=2))]=[0,255,0]
    filenamec ='Imagec.jpg'
    cv2.imshow('Imagec', img)
    saveimg(filenamec, img)
    
def d():
    imageray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,thresh=cv2.threshold(imageray,220,255, 0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(""+str(len(contours)))
    print(contours[0])

    cv2.drawContours(img,contours, -1,(0,0,0),15)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    filenamed = 'outputgray.jpg'
    saveimg(filenamed, gray)
    
def e(img):  
    imageray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(imageray,220,255, 0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img,contours, -1,(0, 0, 0),2)
    h, w, num_c = img.shape
    segmask = np.zeros((h, w, num_c), np.uint8)
    stencil = np.zeros((h, w, num_c), np.uint8)
    for c in contours:
        cv2.drawContours(segmask, [c], 0, (255,255,255), -1)
        cv2.drawContours(stencil, [c], 0, (255, 0, 0), -1)
        stencil[np.where((stencil==[255,0,0]).all(axis=2))] = [0, 0, 0]
        stencil[np.where((stencil==[0,0,0]).all(axis=2))] = [255,255,255]
    mask = cv2.bitwise_xor(stencil, segmask)
    img = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    cv2.imshow('e', img)
    filenameh = 'e.jpg'
    saveimg(filenameh, img)
    
def f():
    imageray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret,thresh=cv2.threshold(imageray,220,255, 0)
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(""+str(len(contours)))
    print(contours[0])

    cv2.drawContours(img,contours, -1,(0,0,255),2)
    filenamef = 'contours.jpg'
    cv2.imshow('contours', img)
    saveimg(filenamef, img) 

def g(): 
    start_point = (38, 48)
    end_point = (316, 331)
    color = (0,255, 0)
    thickness = 2
    imgf = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (55, 61)
    end_point = (300, 316)
    color = (0,255, 0)
    thickness = 2
    image3 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (346, 48)
    end_point = (620, 330)
    color = (0,255, 0)
    thickness = 2
    image3 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (360, 64)
    end_point = (610, 316)
    color = (0,255, 0)
    thickness = 2
    image4 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (666, 48)
    end_point = (942, 330)
    color = (0,255, 0)
    thickness = 2
    image5 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (680,61)
    end_point = (930, 318)
    color = (0,255, 0)
    thickness = 2
    image6 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (38,380)
    end_point = (316, 660)
    color = (0,255, 0)
    thickness = 2
    image7 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (55, 393)
    end_point = (302, 645)
    color = (0,255, 0)
    thickness = 2
    image8 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (346, 380)
    end_point = (620, 659)
    color = (0,255, 0)
    thickness = 2
    image9 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (360, 398)
    end_point = (610, 961)
    color = (0,255, 0)
    thickness = 2
    image10 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (666, 380)
    end_point = (942, 659)
    color = (0,255, 0)
    thickness = 2
    image11 = cv2.rectangle(img, start_point, end_point, color, thickness)
    start_point = (680,397)
    end_point = (930, 961)
    color = (0,255, 0)
    thickness = 2
    image12 = cv2.rectangle(img, start_point, end_point, color, thickness)
    filenameg = 'square.jpg'
    cv2.imshow('square', img)
    saveimg(filenameg, img) 

def h():
    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (15, 700)
    fontScale = 1
    color = (0, 255, 255)
    thickness1 = 2
    sharpened = cv2.putText(img, '51900239-Nguyen Le Bao Thy', org, font, fontScale, color, thickness1, cv2.LINE_AA)
    cv2.imshow('Image Sharpening', sharpened)
    filenameh = 'Name.jpg'
    saveimg(filenameh, img) 

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




