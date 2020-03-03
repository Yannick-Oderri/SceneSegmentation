#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:14:51 2020

@author: ynki9
"""

import numpy as np
import cv2 


def calculate_surface_normals(depth_img):
    shape = (*depth_img.shape[:2], 3)
    normal_data = np.zeros(shape, np.uint8)
    depth_img = depth_img / 5

    for y in range(1, shape[0] - 1):
        for x in range(1, shape[1] - 1):
            try:
                dzdx = (depth_img[y, x+1] - depth_img[y, x-1]) / 2.0
                dzdy = (depth_img[y+1, x] - depth_img[y-1, x]) / 2.0
            except Warning: 
                print(f"y:{y},x:{x}")

            if depth_img[y, x] != 0:
                d = np.asarray([-dzdx, -dzdy, 1.0])
                n = ((d / np.linalg.norm(d))*0.5 + 0.5) * 255
            else:
                n = np.asarray([0, 0, 0])
            
            
            normal_data[y, x] = n.astype(np.int)

    return normal_data


def calculate_surface_normals2(depth_img):
    shape = (*depth_img.shape[:2], 3)
    normal_data = np.zeros(shape, np.uint8)
    depth_img = depth_img / 10

    for y in range(1, shape[0]):
        for x in range(1, shape[1]):
            t = np.asarray([y-1, x, depth_img[y-1, x]])
            l = np.asarray([y, x-1, depth_img[y, x-1]])
            c = np.asarray([y, x, depth_img[y, x]])
            
            if c[2] == 0:
                n = np.asarray([0, 0, 0])
            else:
                d = np.cross((t-c), (l-c))
                n = (((d / np.linalg.norm(d))*0.5+0.5) * 255).astype(np.uint8)
            
            
            normal_data[y, x] = n

    return normal_data

if __name__ == "__main__":
    depth_img = cv2.imread("./data/img/test0.png", -1)
    normal_map = calculate_surface_normals(depth_img)


    cv2.imshow("normal img", normal_map)
    cv2.waitKey(0)