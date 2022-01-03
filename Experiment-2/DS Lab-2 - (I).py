#
import pandas as pd
import numpy as np
import math

DF = pd.read_csv("kc_house_data.csv")
print(DF.head())

x = DF.price
y = DF.sqft_living

#Covariance

print("\n")
def covariance(x,y):
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    s=sum((a - mean_x) * (b - mean_y) for (a,b) in zip(x,y)) / len(x)
    return s
print("Covariance: ",covariance(x,y))


#Correlation
def correlation_pr(x, y):
    n = len(x)
    sum_x = float(sum(x))
    sum_y = float(sum(y))
    sum_x_sq = sum(xi*xi for xi in x)
    sum_y_sq = sum(yi*yi for yi in y)
    psum = sum(xi*yi for xi, yi in zip(x, y))
    num = psum - (sum_x * sum_y/n)
    den = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
    if den == 0:
        return 0
    return num / den
print("Correlation: ",correlation_pr(x,y))

ls={'Values':[covariance(x,y),correlation_pr(x,y)]}
matrix=pd.DataFrame(data=ls, index=['Covariance','Correlation_pr'])
print("\nMatrix: \n",matrix)