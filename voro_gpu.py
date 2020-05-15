# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 18:39:14 2020

@author: Jeffrey
"""

import numpy as np
from scipy.spatial import cKDTree
from collections import Counter

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

def init_connectivity(offset=0):
    return [(2,5,0), (5,3,0), (1,5,2), (5,1,3), (4,2,0), (4,0,3), (2,4,1), (4,3,1)]

def point_dot_plane(point, plane):
    assert(len(point)+1 == len(plane))
    dot_val = (point * plane[:len(point)]).sum() + plane[-1]
    return dot_val

def within_halfspace(point, plane):
    dot_val = point_dot_plane(point, plane)
    if abs(dot_val) < 1e-10:
        raise
    return dot_val > 0


def calc_norm(p):
    return (p**2).sum(axis=-1)

def intersect_planes(pu, pv, pw):
    b = -np.array((pu[-1], pv[-1], pw[-1]))
    Arows = (pu[:-1], pv[:-1], pw[:-1])
    A = np.vstack(Arows)
    x = np.linalg.solve(A, b)
    return x
    
def compute_boundary(R):
    delta_R = []
    while len(R) != 0:
        found = False
        for i, r in enumerate(R):
            if is_simple_cycle_all(delta_R, r):
                delta_R.append(r)
                found = True
                break
            elif is_simple_cycle_front(delta_R, r):
                delta_R.insert(0, r)
                found = True
                break
        if not found:
            raise
        else:
            R.pop(i)
    return delta_R

def intersect_lists(r1, r2):
    res = [r for r in r1 if r in r2]
    return res

def share_edge(r1, r2):
    return len(intersect_lists(r1, r2)) == 2

def get_dual_edges(r):
    for i in range(len(r)):
        yield (r[i-1], r[i]) if r[i-1] < r[i] else (r[i], r[i-1])

def is_simple_cycle_all(delta_R, r):
    for i in range(len(delta_R)):
         if is_simple_cycle_pos(delta_R, r, i):
             return True
    return False
            
def is_simple_cycle_back(delta_R, r):
    return is_simple_cycle_pos(delta_R, r, -1)

def is_simple_cycle_front(delta_R, r):
    return is_simple_cycle_pos(delta_R, r, 0)

def is_simple_cycle_pos(delta_R, r, pos):
    return True if len(delta_R) == 0 else share_edge(delta_R[pos], r)

def dual_boundary(delta_R):
    dual_edges = [e for r in delta_R for e in get_dual_edges(r)]
    c = Counter(dual_edges)
    for edge in c: 
        if c[edge] == 1:
            yield edge
        
def clip_by_plane(T, P, p, pt):
    R = []
    i = 0
    P = list(P)
    T = list(T)
    while i < len(T):
        plane_inds = T[i]
        planes = [P[p_i] for p_i in plane_inds]
        intersection_pt = intersect_planes(*planes)
        if not within_halfspace(intersection_pt, p):
            R.append(plane_inds)
            T.pop(i)
        else:
            i += 1
    if len(R) > 0:
        P.append(p)
        delta_R = compute_boundary(R)
        for edge in dual_boundary(delta_R):
            if len(edge) != 2:
                raise
            T.append((*edge, len(P)-1))
    return np.array(T), np.array(P)


def median_plane(a, b):
    # Assumes points are along last dimension, must have number of dims
    n = a - b
    n = n / np.linalg.norm(n,axis=-1,keepdims=True)
    med = (a + b) / 2
    d = (med*n).sum(axis=-1,keepdims=True)
    eqs = np.concatenate((n, -d),axis=-1)
    return eqs



def get_knn(pts, k=1):
    tree = cKDTree(pts)
    distances, knn = tree.query(pts, k+1)
    knn = knn[:,1:]
    return knn

def build_median_plane_matrix(pts, knn):
    # gather the coordinates in a Nxkx3 array
    npt_arr = pts[knn]
    arr = median_plane(pts[:,np.newaxis,:], npt_arr)
    return arr

# Project a point onto a plane where the plane is given by an equation (4-tuple)
def project_onto_plane(pt, plane):
    err = point_dot_plane(pt, plane)
    norm = plane[:len(pt)]
    new_pt = pt - norm*err
    return new_pt

def line_from_planes(p1, p2):
    n1 = p1[:len(p1)-1]
    n2 = p2[:len(p2)-1]
    line = np.cross(n1, n2)
    if(calc_norm(line) < 1e-10):
        raise
    # create random plane from line and intersect it with the other to get a point
    random_plane = np.concatenate((line, [0]))
    random_pt = intersect_planes(p1, p2, random_plane)
    random_pt2 = random_pt+line
    return (random_pt, random_pt2)
    
def project_onto_line(p, p1, p2):
    line_pt1, line_pt2 = line_from_planes(p1, p2)
    line_to_point = p - line_pt1
    line = line_pt2 - line_pt1
    proj_vector = line_to_point.dot(line)*line
    return proj_vector + line_pt1
    

def barycenters_and_volumes(pt, planes, voronoi_verts):
    for vert in voronoi_verts:
        plane_pts = [project_onto_plane(pt, planes[t]) for t in vert]
        print(plane_pts)


def main():
    N = 10
    np.random.seed(0)
    pts = gen_points(N)
    final_Ps = []
    final_Ts = []
    init_P = get_bounding_planes(pts)
    init_T = init_connectivity()
    knn = get_knn(pts, 9)
    eqs = build_median_plane_matrix(pts, knn)
    for i,pt in enumerate(pts):
        P = init_P.copy()
        T = init_T.copy()
        for eq in eqs[i]:
            T, P = clip_by_plane(T, P, eq, pt)
        barycenters_and_volumes(pt, P, T)
        final_Ps.append(P)
        final_Ts.append(T)
    
        
        

if __name__ == '__main__':
    main()