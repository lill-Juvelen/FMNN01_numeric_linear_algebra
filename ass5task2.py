# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:33:25 2019

@author: lill-Juvelen
"""

#%% TASK 2: implement QR 

from numpy import random
import numpy as np
from numpy import linalg as LA
import scipy 

K = []
#%%
n = 100
A = random.rand(n,n)
A = 1/2 * (A + A.T) 
A = 3* A
evals, evec = np.linalg.eig(A)
print(evals[n-1])

H = scipy.linalg.hessenberg(A)
ev = np.zeros((n,1))
k = 0


# m: size of current matrix
# first, m = n. 
# next step: m = n-1
Continue = True

for m in range(n, 1, -1):
    Continue = True
    print('Current size of H is ', m)
    k = 0
    while Continue:
        k = k+1
        sigma = H[m-1][m-1] # choose diagonal element as shift
        sh = sigma * np.identity(m)
        Q, R = np.linalg.qr(H - sh)
        Hnew = R @ Q + sh
        H = Hnew
        
        if Hnew[m-1][m-2] < 1e-8:
            ev[m-1] = Hnew[m-1][m-1]
            print('converging eigenvalue is ', ev[m-1])
            H = Hnew[:m-1,:m-1]
            if m == 2:
                ev[m-2] = H
                print('last converging eigenvalue ', H )
            Continue = False
print(ev)
#%%
def eig_QRshift(A):
    H = scipy.linalg.hessenberg(A)
    ev = np.zeros((n,1))
    k = 0
    kvec = np.zeros((n,1))
    # m: size of current matrix
    # first, m = n. 
    # next step: m = n-1
    Continue = True
    for m in range(n, 1, -1):
        Continue = True
       # print('Current size of H is ', m)
        k = 0
        while Continue:
            k = k+1
            sigma = H[m-1,m-1] # choose diagonal element as shift
            sh = sigma * np.identity(m)
            Q, R = np.linalg.qr(H - sh)
            Hnew = R @ Q + sh
            H = Hnew
            if Hnew[m-1][m-2] < 1e-8:
                ev[m-1] = Hnew[m-1][m-1]
                #rint('converging eigenvalue is ', ev[m-1])
                H = Hnew[:m-1,:m-1]
                kvec[m-1] = k
                if m == 2:
                    ev[m-2] = H
                    #rint('last converging eigenvalue ', H )
                    kvec[m-2] = k
                    Continue = False
                Continue = False
    return ev, kvec

n = 4
A = random.rand(n,n)
A = 1/2 * (A + A.T) 
A = 3* A
ev, kvec = eig_QRshift(A)
evals, evec = np.linalg.eig(A)
print(kvec)
temp1 = -np.sort(-evals, axis = 0)
temp = -np.sort(-ev, axis = 0)
print(temp[0:10], temp1[0:10])
print()

print(np.mean(kvec))
tt = abs((temp- temp1)**2)
t = tt.diagonal()
mm = np.mean(t)
print(mm)

K.append(np.mean(kvec))

#%% funnkar för en iteration
while k < 10:
    sigma = H[m-1][m-1] # choose diagonal element as shift
    sh = sigma * np.identity(m)
    Q, R = np.linalg.qr(H - sh)
    Hnew = R @ Q + sh
    H = Hnew
    
    if Hnew[m-1][m-2] < 1e-8:
        ev[m-1] = Hnew[:m-1,:m-1]
    k = k+1    
