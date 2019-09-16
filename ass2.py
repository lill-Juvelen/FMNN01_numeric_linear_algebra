# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:12:44 2019

@author: lill-Juvelen
"""

# gram schmidt

import numpy as np
from scipy.linalg import qr


B = np.array([[3, 2, 4, 1],[5, 6, 7, 6], [2, 3, 8, 2], [5, 4, 3, 2]])
A = B.transpose()


def Classical_Gram_Schmidt(A):
    m,n = A.shape
    R = np.zeros(m,n)
    Q = np.zeros(m,n)
    for j in range(0,n):
        v = A[:,j]
        for i in range(0,j):
            R[i,j] = np.dot(Q[:,i], v)
            v = np.subtract(v,R[i,j]*Q[:,i])
        R[j,j] = np.linalg.norm(v)
        Q[:,j] = v/R[j,j]
    return Q, R

#Q, R = Classical_Gram_Schmidt(A)

#print(Q)
#print(np.dot(Q[:,0], Q[:,1]))


def Householder_simple(A):
    m,n = A.shape
    R = np.zeros((m,n))
    Q = np.eye(m)
    A_decomp = np.zeros((m,n))
    # we overwrite the columns of A in each iteration
    
    #for each column of the matrix A
    for k in range(0,n):
        # pick column to reflect
        a = A[k:,k]
        x1 = a[0]
        ahat = np.sign(x1) * np.array([np.linalg.norm(a)] + (m-1)*0.)
        print('this is my a_hat')
        print(ahat)
        
        # compute v
        v = a-ahat
        v = v / np.linalg.norm(v)
        
        Q = np.eye()
        q = np.eye(len(a)) - np.outer(v,v)
        Q[k:,k:] = q;
        
        


def Householder(A):
    m,n = A.shape
    R = np.zeros((m,n))
    Q_final = np.eye(m)
    
    A_next = A
    
    # compute H matrix and introduce zeros to j-th column of A
    for j in range(0,n): 
        # seperate actual column
        a_col = A_next[:,j]
        a_col = a_col[j:]
        
        # perform householder step on part of array that forms v. 
        q = Householderstep(a_col)
        
        # form Q with right dimensions
        Q = np.eye(m)
        Q[j:,j:] = q
        
        # save instances of Q
        Q_final = Q @ Q_final
        
        A_next = Q @ A_next
        
    R = A_next
    return Q_final, R


# input parameter 
def Householderstep(a):
    m = len(a)
    Q = np.eye(m)
    # compute the 2-norm of first column of A
    norm_a = np.linalg.norm(a)
    
    """compute reflection. Sign of first element of 
    column decides sign of reflection"""
    ahat= np.sign(a[0]) * np.array([norm_a]+(m-1)*[0.])
    
    if (np.sign(a[0]) < 0 ):
        print('a was negative')
        print(a)
    
    # compute v
    v = a - ahat
    v = v / np.linalg.norm(v)

    # compute Q/H matrix
    Q=np.eye(m)-2*np.outer(v,v)
    return Q

A = np.array([[1,2],[4,5], [7,8]])
print(A)
Qfin, R = Householder(A)

# results
qTq = Qfin.T @ Qfin
print(qTq)


Adec = Qfin.T @ R
print('Our QR-decomposition')
print(Adec)

q,r = qr(A)

print('Scipy QR-decomposition')
print(q @ r)


q01 = np.dot(Qfin[:,0], Qfin[:,1])
q02 = np.dot(Qfin[:,0], Qfin[:,2])
q12 = np.dot(Qfin[:,1], Qfin[:,2])


print(q01, q02, q12)




         
