"""Implementation of loss/score functions for multi-label evaluation."""

import numpy as np

def pairwise_rank_loss(Z, Y): #truth(Z), prediction(Y)
    """
    Z and Y should be the same size 2-d matrix
    """
    rankloss = ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie0 = 0.5 * ((Z==0) & (Y==0)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie1 = 0.5 * ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==1)).sum(axis=1)
    return -(rankloss + tie0 + tie1)

def pairwise_f1_score(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    # calculate F1 by sum(2*y_i*h_i) / (sum(y_i) + sum(h_i))
    Z = Z.astype(int)
    Y = Y.astype(int)
    up = 2*np.sum(Z & Y, axis=1).astype(float)
    down1 = np.sum(Z, axis=1)
    down2 = np.sum(Y, axis=1)

    down = (down1 + down2)
    down[down==0] = 1.
    up[down==0] = 1.

    return up / down
