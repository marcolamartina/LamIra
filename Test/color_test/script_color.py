#!/usr/bin/env python
import matplotlib.pyplot as plt
import ast
import numpy as np
import cv2
import sys

def color_read():
    color_dict={}

    with open("/Users/marco/Desktop/colors.txt","r") as f:
        for line in f.readlines():
            label,colors=line.strip().split("\t")
            colors=ast.literal_eval(colors)
            color_dict[label]=colors


    labels,values=zip(*color_dict.items())
    l=[v[0] for v in values]
    a=[v[1] for v in values]
    b=[v[2] for v in values]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(l, a, b)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (a[i], b[i]))

    plt.show() 

def color_print():    
    lista=[]
    for a in range(-128,128):
        temp=[]
        for b in range(-128,128):
            temp.append([170,a+128,b+128])
        lista.append(temp)    
    img=np.array(lista)
    img=img.astype(np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    cv2.imshow('image',rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

color_read()    

