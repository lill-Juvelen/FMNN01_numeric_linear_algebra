# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:40:39 2019

@author: lill-Juvelen
"""

import numpy as np
import matplotlib
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from numpy import linalg as LA

def demoGerschgorin(A):
    """plots gerschgorin discs to demonstrate Gerschgorin's theorem.
    
    Greschgorin’s theorem says that for any m×m matrix A, every eigenvalue
    can be found in at least one of m circular discs in the complex plane 
    that have their centers corresponding to the diagonal elements of A, and 
    their radius being the sum of the absolute value of the corresponding 
    row elements.  The theorem also states that if n discs are connected
    but disjoint from the remainderm−ndiscs, precisely n eigenvalues are found
    within this connected domain. 
    """

    n = len(A)
    eval, evec = LA.eig(A)

    patches = []
    
    # draw discs
    
    for i in range(n):
        xi = np.real(A[i,i])
        yi = np.imag(A[i,i])
        ri = np.sum(np.abs(A[i,:])) - np.abs(A[i,i]) 
        
        circle = Circle((xi, yi), ri)
        patches.append(circle)

    fig, ax = plt.subplots()

    p = PatchCollection(patches, cmap=matplotlib.cm.jet, alpha=0.1)
    ax.add_collection(p)
    plt.axis('equal')
    
    for xi, yi in zip(np.real(eval), np.imag(eval)):
        plt.plot(xi, yi,'o')
    
    plt.show()
    

def A_p(p):
    """ Returns a matrix. Off diagonal elements are multiplied with p.
    small p yeilds diagonal-dominant matrix. 
    """
    Ad = np.array([5,0,-2,-3])
    Ap = p* np.array([[0,0,0,-1], [1,0,-1,1],[-1.5, 1, 0, 1], [-1,1,3,0]])
    np.fill_diagonal(Ap, Ad)
    return Ap

p = 0.1
A = A_p(p)
print(A)
demoGerschgorin(A)