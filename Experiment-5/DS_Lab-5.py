#!/usr/bin/env python
# coding: utf-8

# <center><h1> DATA SCIENCE

# <center><h3>Lab-5

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


DS = pd.read_csv("Titanic_Dataset.csv")
DS.head()


# In[3]:


DS.shape


# In[4]:


DS.info()


# In[5]:


DS.describe()


# In[29]:


# Identification of Categorical Variables
categorical_variables = [feature for feature in DS.columns if DS[feature].dtypes == 'O']
print('Number of Categorical Variables: ', len(categorical_variables))
DS[categorical_variables].head(2)


# In[7]:


# Drop columns
DS[categorical_variables].drop(['Name','Ticket','Cabin'],axis=1).head()


# In[8]:


# Storing a columns of a dataset in new variable for performing Feature Encoding
New_DS=pd.read_csv("Titanic_Dataset.csv",usecols=['Sex','Embarked'])
New_DS.head()


# In[9]:


New_DS.shape


# In[10]:


New_DS.info()


# In[11]:


New_DS.count()


# In[12]:


# Finding number of unique values in columns
for i in New_DS.columns:
    print(i, ':', len(New_DS[i].unique()),'labels')


# In[13]:


# Creating dummie values of dataset
pd.get_dummies(New_DS,drop_first=False).shape


# In[14]:


# Printing dummie values
pd.get_dummies(New_DS,drop_first=False).head(10)


# <b> Feature Encoding on column Sex

# In[15]:


# Counting number of values for unique values in Sex columns
New_DS.Sex.value_counts().sort_values(ascending=False).head()


# In[16]:


# Creating dummie value for column Sex
dummies_1=pd.get_dummies(New_DS['Sex'])
dummies_1.head()


# In[17]:


# Selecting top 2 unique values
top_2=[x for x in New_DS.Sex.value_counts().sort_values(ascending=False).head().index]
top_2


# In[18]:


# Coverting values in binary
for label in top_2:
    New_DS[label]=np.where(New_DS['Sex']==label,1,0)


# In[19]:


# Comparing original values with dummie value
New_DS[['Sex']+top_2].head()


# <b> Feature Encoding on column Embarked

# In[20]:


# Counting number of values for unique values in Embarked columns
New_DS.Embarked.value_counts().sort_values(ascending=False).head()


# In[21]:


# Creating dummie value for column Embarked
dummies_2=pd.get_dummies(New_DS['Embarked'])
dummies_2.head()


# In[22]:


# Selecting top 3 unique values
top_3=[x for x in New_DS.Embarked.value_counts().sort_values(ascending=False).head().index]
top_3


# In[23]:


# Coverting values in binary
for label in top_3:
    New_DS[label]=np.where(New_DS['Embarked']==label,1,0)


# In[24]:


# Comparing original values with dummie value
New_DS[['Embarked']+top_3].head(6)


# <b>In this lab, we have performed Feature Encoding on Titanic dataset considering the columns Sex and Embarked. Also, we learned to categories the unique values from a single column present in dataset and find the total count of number of values present in each unique values. We have also created dummies of the data and and compare with the original columns of the dataset and observe and compare the result.

# <div style="text-align: right">Name: <b>Rishab Jha</b> <br>PRN:<b>20190802072</div>
