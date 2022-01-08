#!/usr/bin/env python
# coding: utf-8

# <center><h1> DATA SCIENCE

# <center><h3>Lab-4

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


DS = pd.read_csv("Titanic_Dataset.csv")
DS.head(5)


# In[3]:


DS.shape


# In[4]:


DS.describe()


# <h3>Handling of Missing Values

# In[5]:


Missing_values_count=print(DS.isnull().sum())
Missing_values_count

print("\n")
null_values=[features for features in DS.columns if DS[features].isnull().sum()>1]
for feature in null_values:
    print(feature)


# <h4>1. For Numerical Variables

# In[6]:


# numerical variables containing missing values

num_null_values=[feature for feature in DS.columns if DS[feature].isnull().sum()>1 and DS[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values

for feature in num_null_values:
    print("{}: {}% missing value".format(feature,np.around(DS[feature].isnull().mean(),3)))


# In[7]:


# Replacing num_null_values

for feature in num_null_values:
    median_value=DS[feature].median()

    DS[feature]=np.where(DS[feature].isnull(),1,0)
    DS[feature].fillna(median_value,inplace=True)
    
DS[num_null_values].isnull().sum()


# <h4>2. For Categorical Variables

# In[9]:


# categorical variables containing missing values
cat_null_values=[feature for feature in DS.columns if DS[feature].isnull().sum()>1 and DS[feature].dtypes=='O']

for feature in cat_null_values:
    print("{}: {}% missing values".format(feature,np.round(DS[feature].isnull().mean(),3)))


# In[10]:


# Replace missing value with a new label

def replace_cat_values(DS,cat_null_values):
    data=DS.copy()
    data[cat_null_values]=data[cat_null_values].fillna('NULL')
    return data

DS=replace_cat_values(DS,cat_null_values)

DS[cat_null_values].isnull().sum()


# <b>we can observe the missing values in the data set present in the columns where Age is numerical variable, Cabin and Embarked are categorical variable. We have handle the missing values and replace it with some other values in place of null value. 

# <h3>Handling of Outliers at least for 1 variable.

# In[12]:


numerical_variables = [feature for feature in DS.columns if DS[feature].dtypes != 'O']
print('Number of Numerical Variables: ', len(numerical_variables))
discrete_variables=[feature for feature in numerical_variables if len(DS[feature].unique())<10]
print("Discrete Variables Count: {}".format(len(discrete_variables)))
continuous_variables=[feature for feature in numerical_variables if feature not in discrete_variables]
print("Continuous Variables Count: {}".format(len(continuous_variables)))


# In[13]:


print(continuous_variables)
DS[continuous_variables].head()


# In[14]:


sns.boxplot(DS['Fare'])


# In[15]:


outlier=np.where(DS['Age']>66)
print(outlier)


# In[16]:


import numpy as np
Q1 = np.quantile(DS['Fare'],0.25)
print("Q1 = ",Q1)
Q3 = np.quantile(DS['Fare'],0.75)
print("Q3 = ",Q3)
IQR = Q3 - Q1
print("IQR = ",IQR)

# For finding out the Outlier using IQR we have to define a multiplier which is 1.5

print("\nRange of Outliers:")
Q1=Q1 - 1.5*IQR 
print("Min_Value = ",Q1)
Q3=Q3 + 1.5*IQR
print("Max_Value = ",Q3)


# <b>Hence, any value below Min_Value or Max_Value will be considered as an Outlier

# <h3> Handling of Rare Variables

# In[17]:


cat_rare_variables=[feature for feature in DS.columns if DS[feature].dtype=='O']
cat_rare_variables


# In[18]:


DS.head()


# In[19]:



for feature in cat_rare_variables:
    temp=DS.groupby(feature)['Fare'].count()/len(DS)
    temp_df=temp[temp>0.01].index
    DS[feature]=np.where(DS[feature].isin(temp_df),DS[feature],'Rare_var')


# In[20]:


DS.head()


# In[21]:


for feature in cat_rare_variables:
    labels_ordered=DS.groupby([feature])['Fare'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    DS[feature]=DS[feature].map(labels_ordered)


# In[22]:


DS.head()


# <b>----------------------------------------------------------------------------------------------------------------------------

# NAME: **Rishab Jha**

# *PRN*: **20190802072**
