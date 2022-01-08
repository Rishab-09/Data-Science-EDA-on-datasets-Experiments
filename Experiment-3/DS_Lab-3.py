#!/usr/bin/env python
# coding: utf-8

# <center><h1> DATA SCIENCE

# <center><h3>Lab-3

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


DS = pd.read_csv("Titanic_Dataset.csv")
DS.head(5)


# In[4]:


DS.shape


# In[5]:


DS.describe()


# <h2>Exploratory Data Analysis

# <b>1. Identification of Missing Values

# In[6]:


Missing_values_count=print(DS.isnull().sum())
Missing_values_count

print("\n")
features_with_na=[features for features in DS.columns if DS[features].isnull().sum()>1]
for feature in features_with_na:
    print(feature)


# we can observe the missing values in the data set are in the columns Age, Cabin and Embarked. Also we can understand whether the data set contains missing value using .info() since it provide the count of no  null values. 

# <b>2. Identification of All The Numerical Variables

# In[7]:



# list of numerical variables
numerical_variables = [feature for feature in DS.columns if DS[feature].dtypes != 'O']
print('Number of Numerical Variables: ', len(numerical_variables))

# visualise the numerical variables
DS[numerical_variables].head()


# Hence, we have identify all the numerical variables present in the dataset and there are total 7 columns of numerical variables.

# <b>3. Distribution Of the Numerical Variables

# In[8]:


# Numerical variables are of 2 types:
# 1. Discrete Variabls

discrete_variables=[feature for feature in numerical_variables if len(DS[feature].unique())<10]
print("Discrete Variables Count: {}".format(len(discrete_variables)))


# In[9]:


print(discrete_variables)
DS[discrete_variables].head()


# In[10]:


# 2. Continous Variables

continuous_variables=[feature for feature in numerical_variables if feature not in discrete_variables]
print("Continuous Variables Count: {}".format(len(continuous_variables)))


# In[11]:


print(continuous_variables)
DS[continuous_variables].head()


# Hence, we have distributed Numerical Variables into two types i.e., Discrete variables which includes columns such as 'Survived', 'Pclass', 'SibSp', 'Parch' and Continuous Variable which include column such as 'PassengerId','PassengerId', 'Age', 'Fare'.

# <b>4. Identification of Categorical Variables

# In[12]:


# list of categorical variables
categorical_variables = [feature for feature in DS.columns if DS[feature].dtypes == 'O']
print('Number of Categorical Variables: ', len(categorical_variables))

# visualise the categorical variables
DS[categorical_variables].head()


# Hence, we have identify the categorical variables present in the dataset and there are total 5 columns of categorical variables.

# <b>5. Cardinality of Categorical Variables

# In[13]:


Cardinality_Value=DS[categorical_variables].nunique()
print(Cardinality_Value)


# In[14]:


UV= {'Columns':['Name','Sex','Ticket','Cabin','Embarked'],
     'Unique Values':[891,2,681,147,3]}
DF = pd.DataFrame(UV)
print(DF)

DF.plot.bar('Columns','Unique Values')


# Hence, we have a found the cardinality of a categorical variable i.e., the number of unique values present inside the particular column.

# <b>6. Identification of Outliers

# In[20]:


for feature in continuous_variables:
    data=DS.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        sns.boxplot(data[feature])
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
sns.boxplot(DS['Fare'],DS['Age'])


# Hence, we have identified the outliers present in the numerical variables columns which are aslo a continuous variables.

# <b>7. Relationship between independent and dependent variables.

# In[ ]:


for feature in discrete_variables:
    data=DS.copy()
    data.groupby(feature)['Age'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Age')
    plt.title(feature)
    plt.show()


# Hence, we have found the relationship between the discrete variable considering as independent variable and Age as dependent variable. We can also select continuous variable and observe the result.

# <b>8. Correlation between the variables

# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(DS.corr(), annot=True, cmap="gist_yarg_r", linewidths=1)


# Hence, we plot a heatmap for showing the correlation between the variable whether the variable in a dataset is strongly, positively and negatively correlated.

# <b>In this lab, we have performed EDA on the given dataset. Also, we identify the numerical and categorical variables and its types. Along, with that we have also found the cardinality i.e., uniques values and identify the outliers present inside the numerical variables. At the end we implemted an heatmap graph which shows the correlation between the variables in the dataset.

# NAME: **Rishab Jha**

# *PRN*: **20190802072**
