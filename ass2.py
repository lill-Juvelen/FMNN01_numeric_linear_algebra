# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 10:12:44 2019

@author: lill-Juvelen
"""

# gram schmidt orthogonalization

import numpy as np
from scipy.linalg import qr
from numpy import random, array, linalg

def Classical_Gram_Schmidt(A):
    m,n = A.shape
    R = np.zeros((m,n))
    Q = np.zeros((m,m))
    for j in range(0,n):
        v = A[:,j]
        for i in range(0,j):
            R[i,j] = np.dot(Q[:,i], v)
            v = np.subtract(v,R[i,j]*Q[:,i])
        R[j,j] = np.linalg.norm(v)
        Q[:,j] = v/R[j,j]
    return Q, R


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
        Q_final = Q_final @ Q
        
        A_next = Q @ A_next
        
    R = A_next
    return Q_final, R


# input parameter a is in the k-th step the (m-k)x1 column  k = 0,1,...,n
# "the part of A that forms v"
def Householderstep(a):
    m = len(a)
    Q = np.eye(m)
    # compute the 2-norm of first column of A
    norm_a = np.linalg.norm(a)
    
    """compute reflection. Sign of first element of 
    column decides sign of reflection"""
    ahat= np.sign(a[0]) * np.array([norm_a]+(m-1)*[0.])
    
    # compute v
    v = a - ahat
    v = v / np.linalg.norm(v)

    # compute Q/H matrix
    Q=np.eye(m)-2*np.outer(v,v)
    return Q


def validate_QR(A, method):
    m,n = A.shape
    if method == 0:
        print("Results from  Gram-Schmidt")
        Q,R = Classical_Gram_Schmidt(A)
    if method == 1:
        print("Results from Householder")
        Q,R = Householder(A)
    QQ = Q.T @ Q
    eye_dev = np.linalg.norm(np.eye(m) -QQ, 2)
    
    print()
    ev = np.linalg.eigvals(QQ)
    print("2-norm of (I-QTQ) is : ", eye_dev)
    # 2 norm of I-QTQ
    # 2 norm of Q
    # eigenvalues of QTQ
    # check if all columns are orthogonal
    
#%% testing Gram-Schmidt
    
print('testing')
m = 101
n = 100
A = random.rand(m,n)
validate_QR(A,0)


#%%
n = np.array([1,10,100,1000])
evQ = []
normQ = np.array([.0,.0,.0,.0])
dev = np.array([.0,.0,.0,.0])
detQ = np.array([.0,.0,.0,.0])

for i in range(0,len(n)):
    print('Testing for matrix size')

    print(n[i]+2)
    print(n[i])
    
    
    B = random.rand(n[i]+2, n[i])
    Q,R = Classical_Gram_Schmidt(B)
    I = np.eye(n[i]+2)
    QQ = Q.T @ Q
    
    ##% is the 2 norm of QTQ = 1 ?
    
    normQ[i] = linalg.norm(Q,2)
    
    print('2-norm of Q matrix: ')
    print(normQ[i])
    
    # deviation I-Q^tQ
    temp = linalg.norm(I - QQ,2)
    dev[i] = temp
    
    print('Deviation from identity matrix')
    print(dev[i])
    # eigenvalues of Q
    
    ev = linalg.eigvals(QQ)
    evQ.append(ev)
    
    # 
    detQ[i] =linalg.det(QQ)
    
    print('determinant of QTQ')
    print(detQ[i])        
# %%
    
A = np.random.rand(1001,1000)

Qfin, R = Householder(A)
I = np.identity(1001)


# results
qTq = Qfin.T @ Qfin
temp = linalg.norm(I - qTq,2)
print('2 norm of deviation from identity matrix')
print(temp)

print(qTq)


Adec = Qfin @ R
print('Our QR-decomposition')
print(Adec)

q,r = qr(A)

print('Scipy QR-decomposition')
print(q @ r)

q01 = np.dot(Qfin[:,0], Qfin[:,1])
q02 = np.dot(Qfin[:,0], Qfin[:,2])
q12 = np.dot(Qfin[:,1], Qfin[:,2])


print(q01, q02, q12)




         
