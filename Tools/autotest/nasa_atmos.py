# -*- coding: utf-8 -*-
"""
Created on Tue May 20 09:20:24 2025

@author: thoma
"""

import numpy as np

def T(h):
    if h < 11000:
        return 288.14 - 0.00649 * h
    elif h < 25000:
        return 216.64
    else:
        return 141.89 + 0.00299 * h

def _P(h, t):
    if h < 11000:
        return 101290 * np.pow(t / 288.08, 5.256)
    elif h < 25000:
        return 22650 * np.exp(1.73 - 0.000157 * h)
    else:
        return 2488 * np.pow(t / 216.6, -11.388)

def P(h):
    return _P(h, T(h))

def _Rho(p, t):
    return p / (286.9 * t)

def Rho(h):
    t = T(h)
    p = _P(h, t)
    return _Rho(p, t)


if __name__ == '__main__':
    
    tol = 1e-5
    
    T5000 = T(5000)
    T15000 = T(15000)
    T30000 = T(30000)
    print('T5000 test:' + str(T5000 > 255.69 - tol and T5000 < 255.69 + tol) + ', T = ' + str(T5000))
    print('T15000 test:' + str(T15000 > 216.64 - tol and T15000 < 216.64 + tol) + ', T = ' + str(T15000))
    print('T30000 test:' + str(T30000 > 231.59 - tol and T30000 < 231.59 + tol) + ', T = ' + str(T30000))
    
    P5000 = P(5000)
    P15000 = P(15000)
    P30000 = P(30000)
    print('P5000 test:' + str(P5000 > 54113.934 - tol and P5000 < 54113.934 + tol) + ', P = ' + str(P5000))
    print('P15000 test:' + str(P15000 > 12123.67136 - tol and P15000 < 12123.67136 + tol) + ', P = ' + str(P15000))
    print('P30000 test:' + str(P30000 > 1161.180455 - tol and P30000 < 1161.180455 + tol) + ', P = ' + str(P30000))
    
    Rho5000 = Rho(5000)
    Rho15000 = Rho(15000)
    Rho30000 = Rho(30000)
    print('Rho5000 test:' + str(Rho5000 > 0.7376745768 - tol and Rho5000 < 0.7376745768 + tol) + ', Rho = ' + str(Rho5000))
    print('Rho15000 test:' + str(Rho15000 > 0.1950585359 - tol and Rho15000 < 0.1950585359 + tol) + ', Rho = ' + str(Rho15000))
    print('Rho30000 test:' + str(Rho30000 > 0.01747629497 - tol and Rho30000 < 0.01747629497 + tol) + ', Rho = ' + str(Rho30000))