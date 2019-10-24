"""Code used (with minor modification) from https://arxiv.org/pdf/math-ph/0609050.pdf"""
from scipy import * 
from scipy import linalg

def random_orthogonal(n):
    """A Random Orthogonal matrix distributed with Haar measure"""
    z = randn(n,n)
    q,r = linalg.qr(z)
    d = diagonal(r)
    ph = d/absolute(d)
    q = multiply(q,ph,q)
    return q
