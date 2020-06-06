# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:30:14 2019

@author: lill-Juvelen
"""

# compute eigenvalues of matrix

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import animation

from pylab import *

#%%
"""
 TASK 1: demonstrate the Gerschgorin theorem with Gerschgorin disks and 
 diagonal dominant matrix. we start with a matrix A that is -not- 
 diagonally dominant, and for every iteration we diminish the not diagonal
 elements with a factor p : {1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}
 -> when the matrix is diagonally dominant the gregorin discs cover the 
 true eigenvalues. 
 
 Demonstration by plotting
 TODO: make animation :)
"""
def A_p(p):
    Ad = np.array([5,0,-2,-3])
    Ap = p* np.array([[0,0,0,-1], [1,0,-1,1],[-1.5, 1, 0, 1], [-1,1,3,0]])
    np.fill_diagonal(Ap, Ad)
    return Ap

# collect p factors in decreasing array
p = np.linspace(1, 0, 10, endpoint=False)

"""
# for different p we compute eigenvalues using eig and plot the eigenvalues in 
complex plane
"""
r = []
for j in range(0,10):
    print('p = ', p[j])
    
    # create matrix A(p)
    A = A_p(p[j])
    eigvals, evec = LA.eig(A)
    n = len(A)
    figure(figsize=(8,8))
    ax=subplot(aspect='equal')
    ax.set_xlim((-8, 8))
    ax.set_ylim((-8, 8))
    
    for i in range(0,n):
        xii = np.real(eigvals[i])
        yii = np.imag(eigvals[i])
        xi = np.real(A[i,i])
        yi = np.imag(A[i,i])
        ri = np.sum(np.abs(A[i,:])) - np.abs(A[i,i])
        r.append(ri)
        circle1=plt.Circle((xi,yi),ri, alpha=0.2, edgecolor='r')
        plt.gcf().gca().add_artist(circle1)
        grid(linestyle='-', linewidth=1)
        plt.plot(xi, yi,'o', color='r')
        plt.plot(xii,yii, '.', color='b')
    
    red_patch = mpatches.Patch(color='red', label='Diagonal elements of A(p)')
    blue_patch = mpatches.Patch(color='blue', label='Eigenvalues of A(p)')
    plt.legend(handles=[red_patch, blue_patch])
    plt.title('Eigenvalues and Gerschgorin discs of matrix A(p), p = %.1f' % p[j])
    plt.xlabel('Re' )
    plt.ylabel('Im')
    plt.show()
    
# how make animation of the frames?
    



        
    