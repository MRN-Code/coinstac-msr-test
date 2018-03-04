#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 08:22:47 2018

@author: Harshvardhan
"""

def adam(w, X, y, steps=10000, eta=1e-2, tol=1e-6):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    wc = w
    mt, vt = 0, 0

    i = 0
    while True:
        i = i + 1
        grad = gradient(wc, X, y)
        print(grad)

        mt = beta1 * mt + (1 - beta1) * grad
        vt = beta2 * vt + (1 - beta2) * (grad**2)

        m = mt / (1 - beta1**i)
        v = vt / (1 - beta2**i)

        wp = wc
        wc = wc - eta * m / (np.sqrt(v) + eps)

        if np.linalg.norm(wp - wc) <= tol:
            print(wc)
            break

    return wc, i


# Example 1 - corresponds to one inputspec.json
X = [[2, 3], [3, 4], [7, 8], [7, 5], [9, 8]]
y = [6, 7, 8, 5, 6]
lamb = 0.

# Example 2 - another inputspec.json
# X = [20.1, 7.1, 16.1, 14.9, 16.7, 8.8, 9.7, 10.3, 22, 16.2, 12.1, 10.3]
# y = [31.5, 18.9, 35, 31.6, 22.6, 26.2, 14.1, 24.7, 44.8, 23.2, 31.4, 17.7]

# https://onlinecourses.science.psu.edu/stat462/node/101
#df = pd.read_excel('test(1).xlsx')
#X = np.array(df['X'])
X = np.array([1, 2, 3, 4, 5])
X = np.array([[1] * len(X), X]).T
#y = np.array(df['Y']).T
y = np.array([1, 2, 3, 4, 5])

w = np.array([0, 0])

w_adam, count_adam = adam(w, X, y, steps=10000, eta=0.05, tol=.01)
print('{:<25}{}{}'.format('Adagrad:', w_adam, count_adam))
