#
import pandas as pd
import numpy as np
import math
file = open("Customer_Behaviour.csv", "r")

Data = pd.read_csv(file, sep = ",")
gender = {'Male': 1,'Female': 0}
Data.Gender = [gender[item] for item in Data.Gender]
print(Data)
print(Data.head())

def Point_Biserial_Correlation(a,b, Data):
        bd_unique = Data[a].unique()
        g0 = Data[Data[a] == bd_unique[0]][b]
        g1 = Data[Data[a] == bd_unique[1]][b]
        SD = np.std(Data[b])
        n = len(Data[a])
        n0 = len(g0)
        n1 = len(g1)
        m0 = g0.mean()
        m1 = g1.mean()
        return (m0 - m1)* math.sqrt((n0 * n1) / n ** 2) / SD
print("\n")
print("Point Biserial Correlation Value: ",Point_Biserial_Correlation("Gender", "Salary", Data))
