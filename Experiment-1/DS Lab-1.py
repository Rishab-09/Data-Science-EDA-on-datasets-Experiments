
import numpy as np
import pandas as pd
data_set = pd.read_csv("Data Set.csv")
print(data_set.head())

'''data1 = list(data_set['SepalLengthCm'])
l_ds = len(data1)
print(sum(data1))

# Mean
get_mean=sum(data1)/l_ds
print("Mean of SepalLengthCm: ", get_mean)


#Median
data1.sort()
if l_ds % 2 == 0:
    value1 = data1[l_ds // 2 ]
    value2 = data1[l_ds // 2 - 1]
    Median = (value1+ value2) / 2
else:
    Median = data1[l_ds // 2]

print("Median is: ", Median)


# Mode
mode = max(data1, key= data1.count)
print("Mode is: ",mode)


# Variance
import statistics
Variance = statistics.variance(data1)
print("Variance is: ", Variance)

# Standard Deviation
SD = Variance ** 0.5
print("Standard Deviation is: ", SD)


# Quartile Range
Q1 = np.median(data1[:50])
Q2 = np.median(data1[50:])

IQR = Q2 - Q1           # Interquartile range (IQR)

print("Quartile Range: ",IQR)


# Nominal Data
ND=list(data_set['Species'])
species = []
for i in ND:
    if i not in species:
        species.append(i)
print("Nominal Data is: ",species)


# Categorical or Numerical

name=input('Enter name of Variable: ')
print('The entered variable is',name)

data=list(data_set[name])
result = []
for i in data:
    if i not in result:
        result.append(i)

if len(result)>15:
    print('The variable is numerical')
else:
    print('The variable is categorical')'''


from prettytable import PrettyTable

Contigency_Table = PrettyTable(["Species","SepalLengthCm","PetalLengthCm"])

Contigency_Table.add_row(["Iris-setosa", "4.3-5.8", "1.0-1.9"])
Contigency_Table.add_row(["Iris-versicolor", "4.9-7.0", "3.0-5.1"])
Contigency_Table.add_row(["Iris-virginica", "4.9-7.9", "4.5-6.9"])

print("Contigency Table of Two Different Categorical Variables: \n",Contigency_Table)

