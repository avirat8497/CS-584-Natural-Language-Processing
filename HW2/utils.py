#!/usr/bin/env python

import numpy as np

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """
    ### YOUR CODE HERE

    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= norm
    ### END YOUR CODE
    return x

def softmax(x):
    """Compute the softmax function for each row of the input x.
    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. 

    Arguments:
    x -- A D dimensional vector or N x D dimensional numpy matrix.
    Return:
    x -- You are allowed to modify x in-place
    """
    ### YOUR CODE HERE

    orig_shape = x.shape

    c = np.max(x, axis=-1, keepdims=True)
    exps = np.exp(x - c)
    sum_exps = np.sum(exps, axis=-1, keepdims=True)
    x = exps / sum_exps

    assert x.shape == orig_shape
    return x

    ### END YOUR CODE
    return x

