#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:53:47 2018

@author: Harshvardhan
@notes: Contains code to independently verify the results of the single-shot
        regression where 1 client implies pooled regression
"""

import statsmodels.api as sm

# Example 1 - corresponds to one inputspec.json
X = [[2, 3], [3, 4], [7, 8], [7, 5], [9, 8]]
y = [6, 7, 8, 5, 6]
lamb = 0.

# Example 2 - another inputspec.json
# X = [20.1, 7.1, 16.1, 14.9, 16.7, 8.8, 9.7, 10.3, 22, 16.2, 12.1, 10.3]
# y = [31.5, 18.9, 35, 31.6, 22.6, 26.2, 14.1, 24.7, 44.8, 23.2, 31.4, 17.7]

X = sm.add_constant(X)


# model = sm.OLS(y, X).fit() is equivalent to the following line
# when lamb=0. and is basic regression
model1 = sm.OLS(y, X).fit_regularized(alpha=lamb, L1_wt=0.)
print(model1.params)
