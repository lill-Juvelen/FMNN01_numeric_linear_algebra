# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 19:50:37 2019

@author: lill-Juvelen
"""

import numpy as np
from numpy import linalg
import scipy
from scipy import linalg.lu

class compute_rank:
    
    def __init__(self):
        
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
    Q,R = numpy.linalg.qr(A)
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
