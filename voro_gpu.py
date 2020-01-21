# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:39:14 2020

@author: Jeffrey
"""

import numpy as np

def gen_points(N, mins=None, maxs=None):
    if mins is None:
        mins = [-1.0] * 3
    if maxs is None:
        maxs = [1.0] * 3
    pts = np.random.uniform(mins, maxs, size=(N, len(mins)))
    return pts

def interleave_rows(a, b):
    c = np.vstack([np.vstack(ab) for ab in zip(a,b)])
    return c
    
def get_bounding_planes(pts):
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    
    mineqs = np.hstack((np.eye(3), -mins.reshape(*mins.shape, 1)))
    maxeqs = np.hstack((-np.eye(3), maxs.reshape(*maxs.shape, 1)))
    
    P = interleave_rows(mineqs, maxeqs)
    return P
    

def main():
    N = 1000
    pts = gen_points(N)
    bound_eqs = get_bounding_planes(pts)
    
    

if __name__ == '__main__':
    main()