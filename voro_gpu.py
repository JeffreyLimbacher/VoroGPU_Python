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
    
def init_problem(pts):
    return get_bounding_planes(pts), [(2,5,0), (5,3,0), (1,5,2), (5,1,3), (4,2,0), (4,0,3), (2,4,1), (4,3,1)]

def within_halfspace(point, plane):
    pt_eq = np.concatenate((point, (1)))
    dot_val = np.dot(pt_eq, plane) 
    if abs(dot_val) < 1e-10:
        raise
    return dot_val > 0


def intersect_planes(pu, pv, pw):
    b = -np.array((pu[-1], pv[-1], pw[-1]))
    Arows = (pu[:-1], [pv[:-1], [pw[:-1]]])
    A = np.vstack(Arows)
    x = np.linalg.solve(A, b)
    return x
    
def compute_boundary(R):
    delta_R = []
    while len(R) != 0:
        found = False
        for i, r in enumerate(R):
            if is_simple_cycle(delta_R, r):
                delta_R.append(r)
                R.pop(i)
                found = True
                break
        if not found:
            raise
    return delta_R
            
def is_simple_cycle(delta_R, r):
    if len(delta_R) == 0:
        return True
    first = delta_R[0]
    last = delta_R[-1]
    setr = set(r)
    if len(set(first).intersection(setr)) != 2:
        return False
    elif len(set(last).intersection(setr)) != 2:
        return False
    return True

def edges(delta_R):
    for i in range(len(delta_R)):
        r1 = delta_R[i]
        r2 = delta_R[i-1]
        res = set(r1).intersection(set(r2))
        return list(res)

def clip_by_plane(T, P, p):
    R = []
    i = 0
    while i < len(T):
        plane_inds = T[i]
        planes = [P[p_i] for p_i in plane_inds]
        intersection_pt = intersect_planes(*planes)
        if within_halfspace(intersection_pt, p):
            R.append(plane_inds)
            T.pop(i)
        else:
            i += 1
    if len(R) > 0:
        P.push(p)
        delta_R = compute_boundary(R)
        for edge in edges(delta_R):
            T.append((*edge, p))
    return T, P

def median_plane(a, b):
    n = a - b
    n = n / np.linalg.norm(n)
    med = (a + b) / 2
    d = med.dot(n)
    return np.hstack((n, -d))
    

def main():
    N = 1000
    pts = gen_points(N)
    P, T = init_problem(pts)
    
    

if __name__ == '__main__':
    main()