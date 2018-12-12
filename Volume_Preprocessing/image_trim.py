# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 10:21:44 2018

@author: user
"""

def rectangle_trim(img, x1 = 0, x2 = 0,
                   y1 = 0, y2 = 0):
    res = img[x1:x2, y1:y2]
    return res

def rectangle_trim_volume(volume, x1 = 0, x2 = 0,
                   y1 = 0, y2 = 0):
    res = volume[:,x1:x2, y1:y2]
    return res