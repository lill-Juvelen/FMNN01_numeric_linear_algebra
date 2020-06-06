# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:39:51 2019

@author: lill-Juvelen
"""

# Task 1 

# Compute the rank of a matrix 
# qr
# lu
# svd

import math
import numpy as np 
import scipy
from numpy import random, linalg, array, matrix
from scipy import linalg
import matplotlib 
from matplotlib import *
"""
 Computes the svd of A and returning number of non zero
 singular values with respect to a threshold
 """
def compute_rank_svd(A, thres):
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    temp = abs(s)
    r = len(temp[temp>thres])
    return r

"""
Computes the qr-decomposition of A and returns number of non zero
diagonal element in R with respect to a threshold
"""
def compute_rank_qr(A, thres):
    Q,R = np.linalg.qr(A)
    d = R.diagonal()
    temp = abs(d)
    return len(temp[temp>thres])

"""
Computes the qr-decomposition of A and returns number of non zero
diagonal element in U with respect to a threshold
"""
def compute_rank_lu(A,thres):
    P,L,U = scipy.linalg.lu(A)
    d = U.diagonal()
    temp = abs(d)
    return len(temp[temp>thres])


def kahan_matrix(n,theta):
    s = math.sin(theta)
    c = math.cos(theta)
    r = np.identity(n)
    r[np.triu_indices(n,1)] = -c
    np.fill_diagonal(r,1)
    
    S = np.identity(n)
    for i in range(0,n):
        S[i,i] = s**i
    
    Kahan = (S @ r).T @ (S @ r)
    return Kahan
        
    
    # construct kahan matrix
    

#%%
    # main
    
m = 1000
n = 1000
thres = 0.000000000005
theta = 4

A = random.rand(m,n)
B  = np.array([[0, 0, 0],[0,0,0],[0,8,0]]).T
C = np.array([[0,1,2,3,4,5], [0,0,0,6,5,6],[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,0,3]]).T
D = np.array([[0,0,0],[2,0,0],[3,0,0]])

n = 90
K = kahan_matrix(n,theta)






print('Computed svd rank of matrix B:' , compute_rank_svd(B,thres))
print('Computed qr rank of matrix B:' , compute_rank_qr(B,thres))
print('Computed lu rank of matrix B:' , compute_rank_lu(B,thres))
print('Computed python rank matric B:', np.linalg.matrix_rank(B))

print('Computed svd rank of matrix A:' , compute_rank_svd(A,thres))
print('Computed qr rank of matrix A:' , compute_rank_qr(A,thres))
print('Computed lu rank of matrix A:' , compute_rank_lu(A,thres))
print('Computed python rank matric A:', np.linalg.matrix_rank(A))

print('Computed svd rank of matrix C:' , compute_rank_svd(C,thres))
print('Computed qr rank of matrix C:' , compute_rank_qr(C,thres))
print('Computed lu rank of matrix C:' , compute_rank_lu(C,thres))
print('Computed python rank matric C:', np.linalg.matrix_rank(C))

print('Computed svd rank of matrix D:' , compute_rank_svd(D,thres))
print('Computed qr rank of matrix D:' , compute_rank_qr(D,thres))
print('Computed lu rank of matrix D:' , compute_rank_lu(D,thres))
print('Computed python rank matric D:', np.linalg.matrix_rank(D))


print('Computed svd rank of matrix K:' , compute_rank_svd(K,thres))
print('Computed qr rank of matrix K:' , compute_rank_qr(K,thres))
print('Computed lu rank of matrix K:' , compute_rank_lu(K,thres))
print('Computed python rank matric K:', np.linalg.matrix_rank(K))

#%% kladd


matplotlib.pyplot(pyplot.imshow(K))